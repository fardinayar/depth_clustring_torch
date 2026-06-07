// Batched range-image angle clustering on GPU.
//
// Reproduces the connected-components produced by PRBonn/depth_clustering's
// LinearImageLabeler BFS, but uses a parallel Union-Find (Playne-Hawick style)
// label-equivalence solver instead of a sequential BFS.
//
// The link predicate between two neighbouring pixels is the original
// AngleDiff::GetBeta:
//     d1 = max(d_a, d_b); d2 = min(d_a, d_b);
//     beta = |atan2(d2*sin(alpha), d1 - d2*cos(alpha))|
//     linked  <=>  beta > threshold
// Because beta is symmetric in (d_a, d_b), the "linked" relation is symmetric,
// so connected components are identical to the reference BFS result.
//
// Connectivity: 4-neighbourhood (up/down/left/right). Each pixel only needs to
// process its DOWN and RIGHT edges; UP/LEFT are covered by the neighbour's pass.
// Columns optionally wrap (cylindrical range image).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// --------------------------------------------------------------------------
// device helpers
// --------------------------------------------------------------------------

__device__ __forceinline__ float get_beta(float alpha, float d_a, float d_b) {
    float d1 = fmaxf(d_a, d_b);
    float d2 = fminf(d_a, d_b);
    return fabsf(atan2f(d2 * sinf(alpha), d1 - d2 * cosf(alpha)));
}

// Follow the parent chain to the current root.
__device__ __forceinline__ int find_root(const int* __restrict__ L, int idx) {
    int p = L[idx];
    while (p != idx) {
        idx = p;
        p = L[idx];
    }
    return idx;
}

// Atomic union of two valid nodes (Playne & Hawick). Lower index wins as root.
__device__ __forceinline__ void merge(int* __restrict__ L, int a, int b) {
    while (true) {
        a = find_root(L, a);
        b = find_root(L, b);
        if (a < b) {
            int old = atomicMin(&L[b], a);
            if (old == b) break;     // b was a root; link committed
            b = old;
        } else if (b < a) {
            int old = atomicMin(&L[a], b);
            if (old == a) break;
            a = old;
        } else {
            break;                   // already same component
        }
    }
}

// --------------------------------------------------------------------------
// kernels
// --------------------------------------------------------------------------

__global__ void init_kernel(int* __restrict__ L,
                            const bool* __restrict__ valid,
                            long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    L[idx] = valid[idx] ? (int)idx : -1;
}

__global__ void merge_kernel(int* __restrict__ L,
                             const float* __restrict__ range,
                             const bool* __restrict__ valid,
                             const float* __restrict__ row_alphas,
                             const float* __restrict__ col_alphas,
                             float threshold,
                             bool wrap,
                             int B, int H, int W) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)B * H * W;
    if (idx >= n) return;
    if (!valid[idx]) return;

    int hw = H * W;
    int rem = (int)(idx % hw);
    int b = (int)(idx / hw);
    int r = rem / W;
    int c = rem % W;
    float d = range[idx];

    // DOWN neighbour (r+1, c)
    if (r + 1 < H) {
        long nidx = (long)b * hw + (long)(r + 1) * W + c;
        if (valid[nidx]) {
            float beta = get_beta(row_alphas[r], d, range[nidx]);
            if (beta > threshold) merge(L, (int)idx, (int)nidx);
        }
    }

    // RIGHT neighbour (r, c+1) with optional column wrap
    int nc = -1;
    float alpha_c = 0.f;
    if (c + 1 < W) {
        nc = c + 1;
        alpha_c = col_alphas[c];
    } else if (wrap) {
        nc = 0;
        alpha_c = col_alphas[W - 1];   // wrap alpha stored at last entry
    }
    if (nc >= 0) {
        long nidx = (long)b * hw + (long)r * W + nc;
        if (valid[nidx]) {
            float beta = get_beta(alpha_c, d, range[nidx]);
            if (beta > threshold) merge(L, (int)idx, (int)nidx);
        }
    }
}

// Resolve every valid pixel to its component root (final path compression).
__global__ void flatten_kernel(int* __restrict__ L, long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (L[idx] < 0) return;          // invalid / background
    L[idx] = find_root(L, (int)idx);
}

// --------------------------------------------------------------------------
// launcher
// --------------------------------------------------------------------------

torch::Tensor cluster_cuda(torch::Tensor range,
                           torch::Tensor valid,
                           torch::Tensor row_alphas,
                           torch::Tensor col_alphas,
                           double threshold,
                           bool wrap) {
    TORCH_CHECK(range.is_cuda(), "range must be a CUDA tensor");
    TORCH_CHECK(range.dim() == 3, "range must be (B, H, W)");
    TORCH_CHECK(valid.sizes() == range.sizes(), "valid must match range shape");

    range = range.contiguous();
    valid = valid.to(torch::kBool).contiguous();
    row_alphas = row_alphas.to(range.device(), torch::kFloat32).contiguous();
    col_alphas = col_alphas.to(range.device(), torch::kFloat32).contiguous();

    int B = range.size(0), H = range.size(1), W = range.size(2);
    TORCH_CHECK(row_alphas.numel() == H, "row_alphas must have length H");
    TORCH_CHECK(col_alphas.numel() == W, "col_alphas must have length W");

    long n = (long)B * H * W;
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(range.device());
    torch::Tensor labels = torch::empty({B, H, W}, opts);

    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    init_kernel<<<blocks, threads, 0, stream>>>(
        labels.data_ptr<int>(), valid.data_ptr<bool>(), n);

    merge_kernel<<<blocks, threads, 0, stream>>>(
        labels.data_ptr<int>(),
        range.data_ptr<float>(),
        valid.data_ptr<bool>(),
        row_alphas.data_ptr<float>(),
        col_alphas.data_ptr<float>(),
        (float)threshold, wrap, B, H, W);

    flatten_kernel<<<blocks, threads, 0, stream>>>(
        labels.data_ptr<int>(), n);

    return labels;  // global root index per valid pixel, -1 for invalid
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cluster", &cluster_cuda,
          "Batched range-image angle clustering (CUDA union-find)",
          py::arg("range"), py::arg("valid"), py::arg("row_alphas"),
          py::arg("col_alphas"), py::arg("threshold"), py::arg("wrap"));
}
