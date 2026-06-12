# Depth Clustering Torch

**Depth Clustering Torch** is a highly optimized, batched **PyTorch + CUDA** library for angle-based depth clustering.

It provides extremely fast GPU-accelerated connected-component segmentation for:

* Stereo disparity images
* Organized LiDAR range images
* Arbitrary 3D point clouds

The entire pipeline operates directly inside PyTorch, making it suitable for deep learning workflows, large-scale batched processing, and real-time applications.

---

# Why Depth Clustering Torch?

*  Fully GPU-accelerated CUDA implementation
*  Native PyTorch tensors as input/output
*  Batched processing support
*  Stereo disparity clustering
*  Organized range image clustering
*  Unorganized point cloud clustering
*  Parallel GPU Union-Find connected components
*  Optional ground removal
*  360° LiDAR wrap-around support
*  Deep-learning friendly API

---

# Acknowledgements & Lineage

This library is fundamentally inspired by the excellent C++ implementation:

**PRBonn/depth_clustering**

developed by the Photogrammetry and Robotics Lab at the University of Bonn.

While a Python wrapper based on PyBind11 is available at:

https://github.com/ArianKheir/Depth-clustring-Python-Library

this repository is a complete native PyTorch/CUDA reimplementation.

Unlike wrapper-based solutions, all major computations are executed directly on the GPU using a parallel label-equivalence solver based on the Playne-Hawick Union-Find algorithm, eliminating CPU bottlenecks and enabling efficient batched processing inside neural network pipelines.

---

# Visual Results

## Stereo Disparity Clustering

The clustering algorithm can be directly applied to stereo disparity maps without explicitly reconstructing a 3D point cloud.

| Original RGB | Input Disparity | Clustered Output |
| ------------ | --------------- | ---------------- |
| ![](example/aachen_000000_000019_leftImg8bit.png)    | ![](example/aachen_000000_000019_disparity.png)      | ![](example/Disparity_clustered_output.png)        |

---

## Point Cloud / LiDAR Clustering

Unorganized 3D points are first projected to a spherical range image, clustered in image space, and then labels are mapped back to the original points.

| Range Projection | Clustered Projection |
| ---------------- | -------------------- |
| ![](example/pointcloud_raw_2d.png)        | ![](example/pointcloud_cluster_image.png)            |

---

# Theory

The algorithm groups neighboring depth measurements according to the angle formed between adjacent sensor rays.

Given two neighboring depth measurements $d_a$ and $d_b$ with an angular separation of $\alpha$, the inclination angle $\beta$ is computed as:

$$
\beta = \left| \arctan \left( \frac{d_2 \sin(\alpha)}{d_1 - d_2 \cos(\alpha)} \right) \right|
$$

where:
* $d_1 = \max(d_a, d_b)$
* $d_2 = \min(d_a, d_b)$

If $\beta > \theta$, the two measurements are considered part of the same object and are merged into the same cluster.

Typical values:

| Sensor Type        | Threshold |
| ------------------ | --------- |
| Stereo             | 7°        |
| LiDAR              | 5°        |
| Dense Range Images | 5–10°     |

---

## Installation

To set up the environment, follow these steps:

1. **Environment Setup**
   ```bash
   conda create -n myenv python=3.10 -y
   conda activate myenv
   ```    
2. **Install core dependencies**
   ```bash
   conda install -c conda-forge numpy scipy matplotlib opencv pyyaml -y
   ```
3. **PyTorch & CUDA**
   ```bash
   # Install PyTorch with CUDA 12.1 support
   pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
   ```    
4. **Build & Install Library**
   ```bash
   git clone https://github.com/fardinayar/depth_clustring_torch
   
   cd depth_clustring_torch
   
   pip install -e . --no-build-isolation
   ```

**Note:** Ensure you have the CUDA Toolkit installed on your system that is compatible with your NVIDIA GPU drivers. You can verify your CUDA version with
   ```bash
   nvcc --version
   ```
also Ensure that `nvcc` is accessible through your PATH or that `CUDA_HOME` is correctly defined.

## Requirements
* Python ≥ 3.8
* PyTorch ≥ 1.12
* CUDA Toolkit
* NVIDIA GPU
---
The installation process automatically performs Ahead-of-Time (AOT) compilation of the CUDA extension.
---

## JIT Compilation

If the package is simply copied into a project without installation, the CUDA extension will automatically be compiled during the first import:

```python
import depth_clustering_torch
```

---

# API Overview

Three high-level APIs are provided:

```python
cluster_disparity(...)
cluster_point_cloud(...)
cluster_range_image(...)
```

All APIs return integer label tensors where:

* `0,1,2,...` represent cluster IDs
* `-1` indicates invalid/background/removed points

---

# Stereo Disparity Clustering

Clusters disparity images directly.

```python
import torch
from depth_clustering_torch import cluster_disparity

labels = cluster_disparity(
    disparity=disparity_tensor,
    fx=2262.52,
    fy=2265.30,
    cx=1096.98,
    cy=513.13,
    baseline=0.209,
    theta_deg=7.0,
    min_size=20,
    ground=True,
    ground_thresh_deg=15.0,
)
```

## Input

### disparity

Shape:

```python
(H, W)
```

or

```python
(B, H, W)
```

Disparity values in pixels.

---

## Parameters

| Parameter         | Description               |
| ----------------- | ------------------------- |
| fx                | Camera focal length (x)   |
| fy                | Camera focal length (y)   |
| cx                | Principal point (x)       |
| cy                | Principal point (y)       |
| baseline          | Stereo baseline in meters |
| theta_deg         | Clustering threshold      |
| min_size          | Minimum cluster size      |
| ground            | Enable ground removal     |
| ground_thresh_deg | Ground slope threshold    |

---

# Point Cloud Clustering

Clusters arbitrary 3D point clouds.

```python
from depth_clustering_torch import cluster_point_cloud

labels = cluster_point_cloud(
    points=points_tensor,
    n_rows=64,
    n_cols=1024,
    fov_up_deg=2.0,
    fov_down_deg=-24.9,
    theta_deg=5.0,
    min_size=50,
    wrap=True,
    ground=True,
    ground_thresh_deg=5.0,
)
```

---

## Input

Shape:

```python
(N,3)
```

or

```python
(B,N,3)
```

Coordinate convention:

```text
x = forward
y = left
z = up
```

---

## Parameters

| Parameter         | Description                |
| ----------------- | -------------------------- |
| n_rows            | Vertical sensor resolution |
| n_cols            | Horizontal projection bins |
| fov_up_deg        | Upper FOV                  |
| fov_down_deg      | Lower FOV                  |
| theta_deg         | Clustering threshold       |
| min_size          | Minimum cluster size       |
| wrap              | Enable 360° wrap-around    |
| ground            | Enable ground removal      |
| ground_thresh_deg | Ground slope threshold     |

---

# Range Image Clustering

Clusters organized range images.

```python
from depth_clustering_torch import cluster_range_image

labels = cluster_range_image(
    range_img=range_tensor,
    row_angles=row_angles,
    col_angles=col_angles,
    theta_deg=7.0,
    wrap=True,
    min_size=20,
    ground=True,
    ground_thresh_deg=5.0,
)
```

---

## Input

### range_img

Shape:

```python
(H,W)
```

or

```python
(B,H,W)
```

Range values in meters.

### row_angles

```python
(H,)
```

Vertical angles.

### col_angles

```python
(W,)
```

Horizontal angles.

---

# Ground Removal

All APIs optionally support vectorized ground filtering.

```python
ground=True
```

Points classified as ground are assigned:

```python
-1
```

before clustering.

---

# Performance

The implementation is designed for large-scale batched processing and leverages:

* CUDA parallelism
* Shared-memory optimization
* GPU Union-Find label equivalence
* Batched tensor operations

Compared to CPU-based pipelines, significant speedups can be achieved, especially for large LiDAR scans and high-resolution stereo inputs.

---

# Output Format

All clustering functions return integer labels:

```python
labels.shape == input.shape
```

Example:

```text
-1  -1   0   0   0
-1   1   1   1   0
 2   2   1   3   3
```

where:

* `-1` = background / invalid / ground
* `0,1,2,...` = cluster IDs

---

# Citation

If you use this implementation in academic research, please cite:

```bibtex
@article{placeholder,
  title   = {placeholder},
  author  = {placeholder},
  journal = {placeholder},
  year    = {placeholder}
}
```

---

# Original Depth Clustering Papers

```bibtex
@InProceedings{bogoslavskyi16iros,
  title     = {Fast Range Image-Based Segmentation of Sparse 3D Laser Scans for Online Operation},
  author    = {I. Bogoslavskyi and C. Stachniss},
  booktitle = {Proc. of The International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2016},
  url       = {http://www.ipb.uni-bonn.de/pdfs/bogoslavskyi16iros.pdf}
}
```

```bibtex
@Article{bogoslavskyi17pfg,
  title   = {Efficient Online Segmentation for Sparse 3D Laser Scans},
  author  = {I. Bogoslavskyi and C. Stachniss},
  journal = {PFG -- Journal of Photogrammetry, Remote Sensing and Geoinformation Science},
  year    = {2017},
  pages   = {1--12}
}
```

---

# License

This project is released under the MIT License.

Please also respect the licensing terms of the original PRBonn depth clustering implementation when using derived ideas or datasets.
