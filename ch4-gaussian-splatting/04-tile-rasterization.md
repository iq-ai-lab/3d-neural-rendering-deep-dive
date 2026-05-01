# 04. Tile-based Rasterization 과 Alpha-Compositing

## 🎯 핵심 질문

- Tile-based rasterization 은 무엇이고, 왜 naive pixel-by-pixel splatting 대신 tile 단위로 처리하는가?
- Tile 내에서 Gaussian 들을 **depth 순서로 정렬** 한 후 alpha-compositing 을 어떻게 적용하는가?
- Alpha-compositing 공식 $C = \sum_i \alpha_i T_i c_i$ 에서 transmittance $T_i = \prod_{j<i}(1-\alpha_j)$ 는 어떻게 계산되는가?
- CUDA kernel 에서 이 모든 과정이 **병렬화** 되고, **미분 가능** 하게 구현되려면?

---

## 🔍 왜 Tile-based Rasterization 인가

Naive approach: 각 Gaussian 마다 모든 픽셀 iteration → O(N_gaussian × W × H).

**3DGS 의 최적화**:
- 한 화면에 보이는 Gaussian 은 매우 많음 (100k+)
- 하지만 대부분의 픽셀은 소수의 Gaussian 만 영향 (local support)
- **Tile-based 구조**:
  1. Image 를 16×16 (또는 32×32) tile 로 분할
  2. 각 tile 에 영향을 미치는 Gaussian 들만 수집 (frustum culling)
  3. Tile 별로 depth sort + alpha-compositing
  4. **병렬 처리**: 각 tile 을 독립적 GPU kernel 로 처리

이는 **order-independent transparency** 구현과 **fast rendering** (30 FPS+) 를 동시에 실현합니다.

---

## 📐 수학적 선행 조건

- 기본 확률론: alpha compositing 의 개념 (transmittance, blending)
- 선형대수: 행렬식, determinant (Gaussian 의 contribution 계산)
- CUDA 기초: thread hierarchy, shared memory, atomic operations
- 문서 02-03 참조: 2D Gaussian covariance 와 evaluation

---

## 📖 직관적 이해

### Alpha-Compositing: 투명도와 혼합

한 픽셀에서 **앞뒤 Gaussian** 의 색을 섞는 방법:
- Gaussian A 가 opacity α_A 로 색 c_A 를 기여
- Gaussian B 가 Gaussian A 뒤에 있고, opacity α_B 로 색 c_B
- **Blending**: C = α_A · c_A + (1 - α_A) · c_B
  - α_A = 1 이면 B 는 보이지 않음 (완전 불투명)
  - α_A = 0 이면 A 는 무영향 (완전 투명)

### Transmittance 의 누적

여러 Gaussian 이 쌓여있으면:
$$C = \sum_{i=1}^{N} \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j) \cdot \mathbf{c}_i$$

여기서:
- $\alpha_i$ = Gaussian i 의 opacity (×  2D Gaussian value at pixel)
- $T_i = \prod_{j<i} (1-\alpha_j)$ = transmittance (앞의 모든 Gaussian 을 통과한 광량)
- $\mathbf{c}_i$ = Gaussian i 의 색

**Depth order** 중요: **가까운 것부터** 정렬해야 올바른 결과.

### Tile-based 의 효율

```
┌────────────────────────────────┐
│ Image (e.g., 512×512)          │
├────┬────┬────┬────┬────┬────┬──┤
│    │ T1 │ T2 │    │    │    │  │
├────┼────┼────┼────┼────┼────┼──┤
│    │ T3 │ T4 │    │    │    │  │
├────┼────┼────┼────┼────┼────┼──┤
│    │    │    │    │    │    │  │
└────┴────┴────┴────┴────┴────┴──┘

각 Tile (16×16) → 독립적 CUDA kernel
한 kernel 에서:
  1. Gaussian collect (frustum culling)
  2. Depth sort
  3. Alpha-composite
  4. Per-pixel output
```

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Tile Grid

Image resolution W × H 를 tile size T (e.g., 16) 로 분할:
- Number of tiles: $(⌈W/T⌉, ⌈H/T⌉)$
- Tile (i, j) 는 pixel coordinate $[i·T, (i+1)·T) × [j·T, (j+1)·T)$

### 정의 1.2 — Alpha-Compositing (Over Operator)

Depth order 된 Gaussian list $\{(\alpha_k, \mathbf{c}_k)\}_{k=1}^{K}$ (가까운→먼 순) 에 대해:

$$\mathbf{C}_{\text{final}} = \sum_{k=1}^{K} \alpha_k \left( \prod_{j=1}^{k-1} (1 - \alpha_j) \right) \mathbf{c}_k$$

**재귀 형태** (더 계산 효율):
$$\mathbf{C}_k = \alpha_k \mathbf{c}_k + (1 - \alpha_k) \mathbf{C}_{k-1}$$
$$T_k = (1 - \alpha_k) T_{k-1}$$

초기: $\mathbf{C}_0 = \mathbf{0}$, $T_0 = 1$ (배경).

### 정의 1.3 — Gaussian Opacity at Pixel

픽셀 (u, v) 에서 Gaussian i 의 opacity:
$$\alpha_i(u, v) = \text{opacity}_i \times \exp\left( -\frac{1}{2} (\mathbf{u} - \mathbf{u}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{u} - \mathbf{u}_i) \right)$$

여기서:
- $\text{opacity}_i$ = Gaussian i 의 scalar parameter α
- exponential term = 2D Gaussian value (문서 03)

### 정의 1.4 — Depth Sort

Gaussian i, j 를 비교하는 key: projected z-depth (screen space 또는 world space Z).

**In-tile sorting**: 각 tile 에서, 그 tile 에 기여하는 Gaussian 들을 depth 별로 정렬.

---

## 🔬 정리와 증명

### 정리 1.1 — Alpha-Compositing 의 Associativity

$$\sum_{i=1}^{N} \alpha_i T_i \mathbf{c}_i = \overline{\mathbf{c}}$$

where $T_i = \prod_{j=1}^{i-1} (1-\alpha_j)$.

**증명**: 귀납법.

Base: $\overline{\mathbf{c}}_1 = \alpha_1 \mathbf{c}_1$ ✓.

Step: $\overline{\mathbf{c}}_k = \alpha_k T_k \mathbf{c}_k + \overline{\mathbf{c}}_{k-1} = \alpha_k \mathbf{c}_k + (1-\alpha_k) \overline{\mathbf{c}}_{k-1}$.

따라서 pixel color 는 일관된 blending order 에 따라 결정됨. ∎

### 정리 1.2 — Depth Sort 가 필수인 이유

만약 Gaussian 들이 **역순** (먼→가까운) 으로 정렬되면:
$$\overline{\mathbf{c}}_{\text{wrong}} = \sum_{i=N}^{1} \alpha_i T_i' \mathbf{c}_i$$

where $T_i' = \prod_{j=N}^{i+1} (1-\alpha_j)$.

이는 **$\overline{\mathbf{c}}_{\text{correct}}$ 와 다름** (앞의 불투명한 Gaussian 이 뒤의 색을 완전히 가림) → 부정확한 렌더링.

### 정리 1.3 — Transmittance 의 수렴

$$T_k = \prod_{j=1}^{k-1} (1 - \alpha_j)$$

만약 $\sum \alpha_j \geq 1$, 그러면 $T_K \to 0$ (충분히 많은 Gaussian 후).

**의미**: Transmittance 가 1e-3 이하면 뒤의 Gaussian 은 무시 가능 → early termination 으로 최적화.

---

## 💻 실전 구현 (PyTorch + CUDA 개념)

### 실험 1 — PyTorch 에서의 Alpha-Compositing

```python
import torch

def alpha_composite(alphas, colors, depths):
    """
    Alpha-compositing 계산 (단일 픽셀)
    
    Args:
        alphas: (N,) opacity values [0, 1]
        colors: (N, 3) RGB colors
        depths: (N,) depth values (작을수록 가까움)
    
    Return:
        final_color: (3,) RGB
        final_alpha: () transmittance (배경이 보이는 정도)
    """
    # Depth sort: 작은 depth 부터 (가까운→먼)
    sorted_indices = torch.argsort(depths)
    
    alphas_sorted = alphas[sorted_indices]
    colors_sorted = colors[sorted_indices]
    
    # Alpha-composite (forward pass)
    final_color = torch.zeros(3, dtype=colors.dtype)
    transmittance = 1.0
    
    for i in range(len(alphas_sorted)):
        alpha_i = alphas_sorted[i]
        c_i = colors_sorted[i]
        
        final_color = final_color + transmittance * alpha_i * c_i
        transmittance = transmittance * (1.0 - alpha_i)
        
        # Early termination (선택)
        if transmittance < 1e-3:
            break
    
    return final_color, transmittance

# Test
N = 5
alphas = torch.tensor([0.3, 0.5, 0.2, 0.4, 0.1])
colors = torch.tensor([
    [1.0, 0.0, 0.0],  # Red
    [0.0, 1.0, 0.0],  # Green
    [0.0, 0.0, 1.0],  # Blue
    [1.0, 1.0, 0.0],  # Yellow
    [1.0, 0.0, 1.0],  # Magenta
])
depths = torch.tensor([5.0, 2.0, 8.0, 1.0, 3.0])

final_c, final_t = alpha_composite(alphas, colors, depths)
print(f"Final color: {final_c}")
print(f"Final transmittance: {final_t:.6f}")

# Sanity check: depth sort order
sorted_idx = torch.argsort(depths)
print(f"Depth sort order: {sorted_idx.tolist()}")
print(f"Sorted depths: {depths[sorted_idx].tolist()}")
```

### 실험 2 — Differentiable Alpha-Compositing

```python
def alpha_composite_diff(alphas, colors, depths):
    """
    Differentiable version (requires grad)
    """
    sorted_indices = torch.argsort(depths)
    alphas_sorted = alphas[sorted_indices]
    colors_sorted = colors[sorted_indices]
    
    # Accumulate (no loop, vectorized)
    # T_i = prod(1 - alpha_j, j < i)
    one_minus_alpha = 1.0 - alphas_sorted
    
    # Cumulative product
    cumprod = torch.cumprod(one_minus_alpha, dim=0)
    T = torch.cat([torch.ones(1), cumprod[:-1]])  # Transmittance for each position
    
    # Color contribution
    contrib = (T * alphas_sorted).unsqueeze(-1) * colors_sorted  # (N, 3)
    final_color = contrib.sum(dim=0)
    
    return final_color

# Test with gradients
alphas = torch.tensor([0.3, 0.5, 0.2], requires_grad=True)
colors = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], requires_grad=True)
depths = torch.tensor([5.0, 2.0, 8.0])

final_c = alpha_composite_diff(alphas, colors, depths)

# Loss (render vs. target)
target = torch.tensor([0.2, 0.3, 0.4])
loss = torch.nn.functional.mse_loss(final_c, target)

loss.backward()

print(f"Loss: {loss.item():.6f}")
print(f"∇alphas: {alphas.grad}")
print(f"∇colors:\n{colors.grad}")
print("✓ Gradients computed successfully")
```

### 실험 3 — Tile-based Rendering Simulation

```python
def tile_based_render(gaussians, image_size=(256, 256), tile_size=16):
    """
    Tile-based rendering simulation (간단화)
    
    Args:
        gaussians: list of {pos_2d, cov_2d, color, opacity, depth}
        image_size: (H, W)
        tile_size: 16 or 32
    
    Return:
        image: (H, W, 3)
    """
    H, W = image_size
    image = torch.zeros(H, W, 3)
    
    # Tile grid
    num_tiles_h = (H + tile_size - 1) // tile_size
    num_tiles_w = (W + tile_size - 1) // tile_size
    
    for tile_h in range(num_tiles_h):
        for tile_w in range(num_tiles_w):
            # Tile bounds
            h_min, h_max = tile_h * tile_size, min((tile_h+1) * tile_size, H)
            w_min, w_max = tile_w * tile_size, min((tile_w+1) * tile_size, W)
            
            # Collect Gaussian 들 that affect this tile (frustum culling)
            tile_gaussians = []
            for g in gaussians:
                # Crude: if center is within 3σ of tile
                u, v = g['pos_2d']
                sig = torch.sqrt(torch.diag(g['cov_2d']))
                
                if (w_min - 3*sig[0] < u < w_max + 3*sig[0] and
                    h_min - 3*sig[1] < v < h_max + 3*sig[1]):
                    tile_gaussians.append(g)
            
            # Depth sort
            tile_gaussians.sort(key=lambda g: g['depth'])
            
            # Render each pixel in tile
            for h in range(h_min, h_max):
                for w in range(w_min, w_max):
                    pixel_pos = torch.tensor([float(w), float(h)])
                    
                    # Alpha-composite
                    final_color = torch.zeros(3)
                    transmittance = 1.0
                    
                    for g in tile_gaussians:
                        diff = pixel_pos - g['pos_2d']
                        mahal = (diff @ torch.linalg.inv(g['cov_2d']) @ diff).item()
                        gaussian_val = torch.exp(torch.tensor(-0.5 * mahal))
                        
                        alpha = g['opacity'] * gaussian_val
                        final_color = final_color + transmittance * alpha * g['color']
                        transmittance = transmittance * (1.0 - alpha)
                        
                        if transmittance < 1e-3:
                            break
                    
                    image[h, w] = final_color
    
    return image

# Test
gaussians = [
    {
        'pos_2d': torch.tensor([64.0, 64.0]),
        'cov_2d': torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
        'color': torch.tensor([1.0, 0.0, 0.0]),
        'opacity': 0.8,
        'depth': 5.0,
    },
    {
        'pos_2d': torch.tensor([80.0, 80.0]),
        'cov_2d': torch.tensor([[10.0, 0.0], [0.0, 10.0]]),
        'color': torch.tensor([0.0, 1.0, 0.0]),
        'opacity': 0.6,
        'depth': 10.0,
    },
]

image = tile_based_render(gaussians, (128, 128), tile_size=16)
print(f"Rendered image shape: {image.shape}")
print(f"Image min/max: {image.min():.4f} / {image.max():.4f}")
print("✓ Tile-based rendering complete")
```

---

## 🔗 실전 활용

### 1. CUDA Kernel Structure (Pseudocode)

```cuda
__global__ void tile_based_rasterize_kernel(
    int num_gaussians,
    float* gaussian_centers_2d,    // (N, 2)
    float* gaussian_covs_2d,       // (N, 4) for 2x2 symmetric
    float* gaussian_colors,        // (N, 3)
    float* gaussian_opacities,     // (N,)
    float* gaussian_depths,        // (N,)
    int* tile_gaussian_indices,    // Pre-sorted by depth per tile
    int* tile_gaussian_counts,     // Count per tile
    float* output_image,           // (H, W, 3)
    int H, int W, int tile_size
) {
    // blockIdx = (tile_h, tile_w)
    // threadIdx.x = pixel within tile
    
    int tile_h = blockIdx.y;
    int tile_w = blockIdx.x;
    int pixel_idx = threadIdx.x;
    
    int h_min = tile_h * tile_size;
    int w_min = tile_w * tile_size;
    int h = h_min + (pixel_idx / tile_size);
    int w = w_min + (pixel_idx % tile_size);
    
    if (h >= H || w >= W) return;
    
    float color[3] = {0, 0, 0};
    float transmittance = 1.0;
    
    int count = tile_gaussian_counts[tile_h * gridDim.x + tile_w];
    int* indices = &tile_gaussian_indices[...];
    
    for (int i = 0; i < count; i++) {
        int g_idx = indices[i];
        
        float u = w - gaussian_centers_2d[g_idx * 2 + 0];
        float v = h - gaussian_centers_2d[g_idx * 2 + 1];
        
        // Mahalanobis distance (2D)
        float mahal = ... // compute using cov_2d_inv
        float gauss_val = exp(-0.5f * mahal);
        
        float alpha = gaussian_opacities[g_idx] * gauss_val;
        
        color[0] += transmittance * alpha * gaussian_colors[g_idx*3+0];
        color[1] += transmittance * alpha * gaussian_colors[g_idx*3+1];
        color[2] += transmittance * alpha * gaussian_colors[g_idx*3+2];
        
        transmittance *= (1.0f - alpha);
        
        if (transmittance < 1e-3f) break;
    }
    
    output_image[(h * W + w) * 3 + 0] = color[0];
    output_image[(h * W + w) * 3 + 1] = color[1];
    output_image[(h * W + w) * 3 + 2] = color[2];
}
```

### 2. Depth Sort 최적화

- **Radix sort** (per tile, device memory)
- **Pre-sort** during training updates (sparse, 일부만 변경)

### 3. Differentiability

각 단계가 미분 가능:
- Forward: alpha-composite 계산
- Backward: ∂loss/∂color, ∂loss/∂opacity, ∂loss/∂depth
- Depth sort gradient: "relaxed" sort (near-identical values 는 gradient 무시)

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|------------|
| Depth order 명확 | Z-fighting (매우 가까운 Gaussian) 시 instability. ε separation 추가 |
| Tile size 고정 | 매우 큰 Gaussian (e.g., >100px) 은 여러 tile 에 분산 → 중복 계산 |
| Early termination safe | Transmittance < 1e-3 후 gradient 흐름 끊김. 정확한 gradient 는 모든 Gaussian 사용 필요 |
| Per-pixel rendering | Multiple samples (MSAA, jittering) 은 추가 비용 |
| Screen-space depth | Object-space depth 로 정렬하면 다른 결과 가능 (rare) |

---

## 📌 핵심 정리

$$\boxed{\mathbf{C} = \sum_{i=1}^{K} \alpha_i \left( \prod_{j=1}^{i-1} (1-\alpha_j) \right) \mathbf{c}_i}$$

| 단계 | 계산 | 병렬화 |
|------|------|--------|
| **Tile division** | Image → 16×16 tiles | Per-tile kernel |
| **Gaussian collect** | Frustum culling | Shared memory reduction |
| **Depth sort** | In-tile Gaussian depth order | Radix sort |
| **Alpha-composite** | Transmittance accumulation | Per-pixel thread |
| **Output** | Final RGB to image buffer | Parallel write |

**핵심 성질**:
1. ✓ O(K) per pixel (K = tile 내 Gaussian 수, ~10-100)
2. ✓ Order-dependent transparency (맞는 순서 필수)
3. ✓ Fully differentiable (역전파 가능)
4. ✓ GPU 친화적 (shared memory, coalesced access)

---

## 🤔 생각해볼 문제

**문제 1** (기초): 3개 Gaussian alpha=[0.5, 0.5, 0.5] 순서로 pixel 에 기여할 때, 최종 transmittance 는?

<details>
<summary>해설</summary>

$$T_4 = (1-0.5)(1-0.5)(1-0.5) = 0.125$$

배경이 12.5% 보임, 전체 87.5% 가려짐.

</details>

**문제 2** (심화): Alpha-compositing 에서 depth sort 를 **거꾸로** 하면 어떤 문제가 생기는가?

<details>
<summary>해설</summary>

가까운 불투명한 Gaussian 이 먼저 나오면, 그것이 배경 C=0 을 "가림" → 뒤의 Gaussian 색이 완전히 transmittance 에 의해 약해짐 → 최종 색이 어두워짐 (부정확).

정확한 rendering: 먼→가까운 순.

</details>

**问题 3** (논문 비평): Tile size 를 너무 작게 (e.g., 4×4) 또는 너무 크게 (e.g., 64×64) 설정하면?

<details>
<summary>해설</summary>

**너무 작음 (4×4)**: 
- Tile 수 많음 → kernel launch overhead 증가
- Frustum culling 부정확 (작은 영역, 많은 중복)

**너무 큼 (64×64)**:
- 큰 Gaussian 이 한 tile 에 몰려 → thread contention, register spill
- Shared memory 부족

**최적**: 16×16 또는 32×32 (GPU architecture 와 typical scene 에 최적화)

</details>

---

<div align="center">

[◀ 이전](./03-ewa-projection.md) | [📚 README](../README.md) | [다음 ▶](./05-adaptive-density-control.md)

</div>
