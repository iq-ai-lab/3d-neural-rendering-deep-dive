# 06. 3DGS 학습 · 재현 · NeRF 와의 비교

## 🎯 핵심 질문

- 3DGS 의 loss function 은 무엇인가? NeRF 의 L2 loss 와 어떤 차이가 있는가?
- COLMAP point cloud 로부터 초기화하는 방법과 그 중요성은?
- Mip-NeRF 360 같은 벤치마크에서 3DGS 는 어떤 성능을 보이는가? PSNR, 학습 시간, 렌더링 속도는?
- NeRF vs 3DGS 의 장단점 표는?

---

## 🔍 왜 이 문서가 중요한가

지금까지 우리는 **개별 기법들** (Gaussian parameterization, SH, EWA, tiling, densification) 을 다루었습니다. 이 문서는:

1. **통합 학습 루프**: 모든 기법이 어떻게 함께 작동하는가
2. **실제 재현**: public dataset (Mip-NeRF 360) 에서의 결과
3. **정량 비교**: NeRF 와 3DGS 의 정확한 차이점
4. **실전 조언**: hyperparameter, trick, 주의점

3DGS 논문 (Kerbl et al. 2023) 에서 가장 임팩트 있는 결과들을 재현하고 해석합니다.

---

## 📐 수학적 선행 조건

- Loss functions: L1, D-SSIM, PSNR 정의
- Optimization: Adam, learning rate scheduling
- Evaluation metrics: image quality assessment
- 문서 01-05 참조: 모든 기법

---

## 📖 직관적 이해

### Loss Function: L1 + D-SSIM

NeRF 는 **L2 loss** (pixel-wise MSE) 를 사용:
$$\mathcal{L}_{\text{NeRF}} = \sum_{\text{rays}} \|\hat{\mathbf{c}} - \mathbf{c}_{\text{gt}}\|_2^2$$

3DGS 는 **하이브리드** loss:
$$\mathcal{L}_{3DGS} = (1-\lambda) \mathcal{L}_{L1} + \lambda \mathcal{L}_{D\text{-SSIM}}$$

여기서:
- **$\mathcal{L}_{L1}$** = L1 distance (pixel color 차이)
- **$\mathcal{L}_{D\text{-SSIM}}$** = 1 - SSIM (structural similarity, local 구조 고려)
- **λ = 0.2** (보통)

**왜 이 혼합인가?**
- **L1**: local pixel accuracy (색 맞추기)
- **SSIM**: global structure (edge, gradient 보존)
- **L2 대신 L1**: outlier 에 덜 민감 (noise 있는 사진)
- 결과: **perceptually better images** (보기에 더 좋음)

### Initialization: COLMAP Point Cloud

```
Raw images → COLMAP → sparse point cloud (관점별 correspondence)
                    ↓
                  SfM points ~5k-50k
                    ↓
          3DGS 초기 Gaussian 위치 (μ)
          3DGS 초기 크기 (s, PCA 로부터)
```

**Why COLMAP?**
- Geometric prior: 3D 구조 대략 알 수 있음
- Initialization quality: 임의 초기화보다 ~10배 빠른 수렴
- Alternative: random init (가능하지만 느림)

### Mip-NeRF 360 Benchmark

**Dataset**: Google 에서 공개한 unbounded outdoor scenes (360도).

예:
- **bicycle**: 자전거, tree 배경, ~150 images
- **garden**: 정원, ~150 images
- **kitchen**: 실내, ~150 images
- ... (총 8개 scenes)

**Metric**:
- **PSNR** (Peak Signal-to-Noise Ratio): dB, higher = better
  $$\text{PSNR} = 10 \log_{10} \frac{1}{\text{MSE}}$$
- **SSIM**: 0-1, higher = better
- **LPIPS**: perceptual distance, lower = better (human-aligned)

### 성능: 3DGS vs NeRF

**Mip-NeRF 360 에서 (대략)**:

| Metric | NeRF (Mildenhall 2020) | Mip-NeRF (Barron 2021) | 3DGS (Kerbl 2023) |
|--------|----------------------|----------------------|-------------------|
| **PSNR (dB)** | ~24 | ~26 | **27-28** ✓ |
| **SSIM** | ~0.72 | ~0.76 | **~0.78** ✓ |
| **LPIPS** | ~0.22 | ~0.18 | **~0.15** ✓ |
| **Train time** | 100+ hrs | 50+ hrs | **30 min** (V100) ✓✓ |
| **Render FPS** | 1-5 | 1-5 | **100+** ✓✓✓ |

**핵심 관찰**:
- Quality: 3DGS ≈ Mip-NeRF (약간 더 좋음)
- Speed: 3DGS **100배 빠름** (training), **20배 빠름** (rendering)

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Loss Function (정량화)

$$\mathcal{L} = (1-\lambda) \left( \frac{1}{n} \sum_{i=1}^n |\hat{c}_i - c_i^{\text{gt}}| \right) + \lambda (1 - \text{SSIM}(\hat{\mathbf{I}}, \mathbf{I}^{\text{gt}}))$$

여기서:
- **$\hat{c}_i$** = rendered RGB at pixel i
- **$c_i^{\text{gt}}$** = ground truth RGB
- **SSIM** = structural similarity (window-based)
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
  (window 11×11, c₁, c₂ numerical stability)
- **λ = 0.2** (0.3 도 사용 가능)

### 정의 1.2 — COLMAP Initialization

COLMAP output: point cloud {**p_i** ∈ ℝ³, visibility set V_i}.

각 point 에 대해:
1. **Position μ** = **p_i** (point cloud 좌표)
2. **Scale s**: 
   - Nearby points 까지의 평균 거리 (또는 k-NN 의 PCA)
   - 보통 0.01-0.1 범위로 normalize
3. **Rotation q** = identity (또는 near point 들의 PCA 주축)
4. **Color c** = COLMAP point color 또는 near image 에서 sample
5. **Opacity α** = 0.1 (작게 시작, 학습하면서 증가)

### 정의 1.3 — Training Loop

Iteration t = 0 to T (보통 30k iterations):

```
1. Forward render (tile-based)
2. Compute loss L
3. Backward pass
4. Optimizer step (Adam, with schedules)
5. If t % 100: Densification (clone/split/prune)
6. If t % 3000: Opacity reset
```

### 정의 1.4 — Evaluation Metrics

$$\text{PSNR} = 10 \log_{10} \frac{1}{\text{MSE}}, \quad \text{MSE} = \frac{1}{n} \sum_i (\hat{c}_i - c_i^{\text{gt}})^2$$

$$\text{SSIM} \in [0, 1], \quad \text{higher is better}$$

$$\text{LPIPS} = \mathbb{E}[\|\mathbf{f}^l_{\text{synth}} - \mathbf{f}^l_{\text{gt}}\|^2 \text{ normalized}], \quad \text{lower is better}$$

(LPIPS: AlexNet feature distance, human perceptual alignment)

---

## 🔬 정리와 증명

### 정리 1.1 — L1 + D-SSIM Loss 의 정당성

**명제**: L1 + D-SSIM loss 는 L2 loss 보다:
1. **Noise 에 robust** (outlier pixels)
2. **Structural detail** 보존
3. **Perceptual quality** 향상

**증명 스케치**:

**1. Robustness to outliers**:
- L2: $\mathcal{L} = (e)^2$ — e=1 → cost 1, e=10 → cost 100 (급증)
- L1: $\mathcal{L} = |e|$ — e=1 → cost 1, e=10 → cost 10 (선형)

outlier 1개 에 L2 는 과도하게 반응, L1 은 절제됨.

**2. SSIM 의 구조성**:
SSIM 는 local window 에서:
- Mean 비교 (brightness)
- Variance 비교 (contrast)
- Correlation (structure)

따라서 edge, gradient 등 **structure** 가 보존되도록 유도.

**3. Perceptual alignment**:
user study 와 LPIPS 계산 → L1+SSIM loss 의 결과가 human preference 더 높음. ∎

### 정리 1.2 — Convergence 속도

3DGS 는 **매우 빠르게 수렴**:
- **0-5k iteration**: loss 급감 (대부분 high-level structure)
- **5k-15k iteration**: detail 개선 (finer Gaussian 추가)
- **15k-30k iteration**: fine-tuning, burnout

NeRF 와 비교하면:
- NeRF: spectral bias 때문에 low-freq 부터 slow convergence
- 3DGS: explicit points → immediate placement

### 정리 1.3 — Rendering Speed

3DGS tile-based rasterization:
- Per-pixel 계산: O(K_tile) 여기 K_tile = tile 내 Gaussian ~50
- No NN forward pass (NeRF 의 MLP 평가 없음)
- GPU parallelization 완벽함
- **결과**: 512×512 이미지 **~10ms** (V100)

NeRF:
- Per-ray calculation: O(samples × MLP depth) = O(64 × 8)
- Ray marching: sequential (병렬화 어려움)
- **결과**: 512×512 이미지 **~200-500ms**

---

## 💻 실전 구현 (PyTorch 최소 예제)

### 실험 1 — Loss Function 구현

```python
import torch
import torch.nn.functional as F
from kornia.losses import ssim

def hybrid_loss(rendered_image, gt_image, lambda_ssim=0.2):
    """
    3DGS loss: (1-λ)L1 + λ D-SSIM
    
    Args:
        rendered_image: (B, 3, H, W) or (H, W, 3)
        gt_image: same shape
        lambda_ssim: float in [0, 1]
    
    Return:
        loss: scalar
    """
    # Shape normalize
    if rendered_image.ndim == 3:
        rendered_image = rendered_image.permute(2, 0, 1).unsqueeze(0)
        gt_image = gt_image.permute(2, 0, 1).unsqueeze(0)
    elif rendered_image.ndim == 4 and rendered_image.shape[1] != 3:
        rendered_image = rendered_image.permute(0, 3, 1, 2)
        gt_image = gt_image.permute(0, 3, 1, 2)
    
    # L1 loss
    l1_loss = F.l1_loss(rendered_image, gt_image)
    
    # SSIM loss
    ssim_val = ssim(rendered_image, gt_image, window_size=11)  # kornia
    ssim_loss = 1.0 - ssim_val
    
    # Hybrid
    loss = (1 - lambda_ssim) * l1_loss + lambda_ssim * ssim_loss
    
    return loss

# Test
rendered = torch.rand(256, 256, 3)
gt = torch.rand(256, 256, 3)

loss = hybrid_loss(rendered, gt, lambda_ssim=0.2)
print(f"Loss: {loss.item():.6f}")
```

### 실험 2 — COLMAP 초기화 시뮬레이션

```python
import torch
import numpy as np
from sklearn.decomposition import PCA

def initialize_from_point_cloud(points, colors=None, k_neighbors=3):
    """
    COLMAP point cloud 로부터 Gaussian 초기화
    
    Args:
        points: (N, 3) 3D points
        colors: (N, 3) optional, RGB colors
        k_neighbors: int, PCA 를 위한 이웃 수
    
    Return:
        {mu, s, q, c, alpha}
    """
    N = points.shape[0]
    
    # 1. Position μ = point 자체
    mu = points  # (N, 3)
    
    # 2. Scale s: k-NN 거리로부터 추정
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(points)
    distances, indices = nbrs.kneighbors(points)
    # 첫 열은 자신, 나머지는 이웃
    mean_dist = distances[:, 1:].mean(axis=1)  # (N,)
    s = torch.from_numpy(mean_dist).float().unsqueeze(-1).repeat(1, 3) * 0.5
    
    # 3. Rotation q: identity 또는 PCA
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(N, 1)
    
    # 4. Color c: point color 또는 (0.5, 0.5, 0.5)
    if colors is not None:
        c = torch.from_numpy(colors).float()
    else:
        c = torch.ones(N, 3) * 0.5
    
    # 5. Opacity α: 작게 시작
    alpha = torch.ones(N, 1) * 0.1
    
    return {
        'mu': mu,
        's': s,
        'q': q,
        'c': c,
        'alpha': alpha
    }

# Test with synthetic point cloud
np.random.seed(42)
points = np.random.randn(100, 3) * 2 + [0, 0, 5]  # Scene ~5m away
colors = np.random.rand(100, 3)

gaussians = initialize_from_point_cloud(points, colors, k_neighbors=5)
print(f"Initialized {len(gaussians['mu'])} Gaussians")
print(f"  μ shape: {gaussians['mu'].shape}")
print(f"  s shape: {gaussians['s'].shape}, range: [{gaussians['s'].min():.4f}, {gaussians['s'].max():.4f}]")
print(f"  α: {gaussians['alpha'].mean():.4f} (should be ~0.1)")
```

### 실험 3 — 전체 학습 루프 (간단화)

```python
def train_3dgs_simple(
    gt_images, camera_poses, intrinsics,
    num_iterations=30000, 
    densify_interval=100, 
    opacity_reset_interval=3000
):
    """
    간단한 3DGS 학습 루프
    """
    # 1. Initialize (COLMAP points 대신 dummy)
    N_gaussians = 1000
    gaussians = {
        'mu': torch.randn(N_gaussians, 3, requires_grad=True) * 2 + [0, 0, 5],
        's': torch.ones(N_gaussians, 3, requires_grad=True) * 0.1,
        'q': torch.randn(N_gaussians, 4, requires_grad=True),
        'c': torch.rand(N_gaussians, 48, requires_grad=True),  # SH 16×3
        'alpha': torch.ones(N_gaussians, 1, requires_grad=True) * 0.1,
    }
    
    # Optimizer
    params = [v for v in gaussians.values() if v.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    losses = []
    
    for it in range(num_iterations):
        loss_iter = 0.0
        num_images = len(gt_images)
        
        # Mini-batch: sample some images
        image_ids = torch.randperm(num_images)[:4]  # batch size 4
        
        for img_id in image_ids:
            # Render
            rendered = render_image(gaussians, camera_poses[img_id], intrinsics)
            
            # Loss
            loss = hybrid_loss(rendered, gt_images[img_id])
            loss_iter += loss.item()
            
            # Backward
            loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss_iter / num_images)
        
        # Densification (every 100 iter)
        if it % densify_interval == 0 and it > 500:
            print(f"Iteration {it}: densifying...")
            # Clone/split/prune logic (문서 05)
            # ...
        
        # Opacity reset (every 3000 iter)
        if it % opacity_reset_interval == 0 and it > 1000:
            print(f"Iteration {it}: opacity reset...")
            gaussians['alpha'].data.fill_(0.5)
        
        if (it + 1) % 5000 == 0:
            print(f"Iteration {it+1}: Loss = {losses[-1]:.6f}, PSNR = {psnr(losses[-1]):.2f} dB")
    
    return gaussians, losses

def psnr(mse):
    return 10 * np.log10(1.0 / max(mse, 1e-10))
```

---

## 🔗 실전 활용

### 1. Public Implementation

- **Official**: https://github.com/graphdeco-inria/gaussian-splatting (PyTorch + CUDA)
- **gsplat** (Nvidia): optimized CUDA kernels
- **Easy-to-use wrapper**: various frameworks

### 2. Hyperparameter Tuning

| Hyperparameter | Default | Range | Note |
|---|---|---|---|
| λ (SSIM weight) | 0.2 | [0, 0.5] | Higher = more structure-focused |
| Learning rate | 1e-3 | [1e-4, 1e-2] | Decay schedule 추천 |
| Densify interval | 100 | [50, 200] | Too frequent = overhead |
| Gradient threshold | 0.0002 | [1e-4, 5e-4] | Scene 마다 조정 |
| Opacity threshold | 0.005 | [0.001, 0.01] | Lower = more aggressive prune |

### 3. Common Tricks

- **Warmup**: 처음 500 iteration 은 densification 안 함
- **Gradient clipping**: gradient 매우 크면 clipping (e.g., max norm 1.0)
- **LR schedule**: exponential decay (e.g., 1e-3 → 1e-4 over iterations)
- **Multi-resolution**: 낮은 해상도부터 시작해서 점진적 업샘플

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|------------|
| COLMAP initialization available | COLMAP 실패 scene (특이한 texture) → random init (느림) |
| Static scene | 동영상/동적 객체 → chapter 5 deformable/4D GS |
| Known camera pose | pose estimation error → bundle adjustment (COLMAP 재실행) |
| Bounded scene | Unbounded outdoor (sky) → mip-NeRF 360 기법 또는 explicit bg model |
| Sufficient training views | 매우 적은 view (~3-5) → regularization 또는 prior 필요 |

---

## 📌 핵심 정리

### Training Recipe

| 단계 | 무엇 | 상세 |
|------|------|------|
| **Init** | COLMAP + PCA | point cloud, color, scale |
| **Loss** | (1-λ)L1 + λD-SSIM | λ=0.2 |
| **Optimize** | Adam lr 1e-3 + decay | gradient-based |
| **Densify** | Clone/Split/Prune | every 100/3000 iter |
| **Render** | Tile-based splatting | 16×16 tile + depth sort |
| **Result** | PSNR 27-28, 30min, 100 FPS | Mip-NeRF 360 benchmark |

### NeRF vs 3DGS Comparison

| 측면 | NeRF | 3DGS |
|------|------|------|
| **Representation** | Implicit (MLP) | Explicit (Gaussians) |
| **Training time** | 1-2 days (V100) | 30 min (V100) |
| **Render speed** | 1-5 FPS | 100+ FPS |
| **Quality (PSNR)** | ~24-25 dB | **27-28 dB** |
| **Memory** | ~500MB | ~500MB (but faster) |
| **Novel view** | Excellent | Excellent |
| **Mesh extraction** | Marching cubes (indirect) | Limited (point cloud) |
| **Editing** | Hard (implicit) | Easier (explicit) |
| **Dynamic/4D** | NeRF-D, HyperNeRF | 4DGS, GauFRe |
| **Unbounded/360** | Mip-NeRF 360 | requires modifications |

**핵심**: 3DGS = **속도** 최우선, NeRF = **유연성** 최우선

---

## 🤔 생각해볼 문제

**문제 1** (기초): L1 loss 와 L2 loss 의 gradient 를 비교하라. outlier (e=100) 가 있을 때 어느 것이 더 큰 gradient 를 produce 하는가?

<details>
<summary>해설</summary>

**L1**: $\mathcal{L} = |e|$, $\frac{\partial \mathcal{L}}{\partial e} = \text{sign}(e)$ → gradient magnitude = 1 (상수)

**L2**: $\mathcal{L} = e^2$, $\frac{\partial \mathcal{L}}{\partial e} = 2e$ → e=100 에서 gradient = 200 (매우 큼)

L2 가 훨씬 큰 gradient → outlier 에 overly sensitive → unstable training.

L1 이 robust (항상 같은 크기 gradient) → stable.

</details>

**문제 2** (심화): COLMAP initialization 없이 random Gaussian 으로 시작하면, 수렴 속도와 최종 quality 가 어떻게 달라지는가?

<details>
<summary>해설</summary>

**Random init**:
- 처음: ~1000 iteration 은 대부분 garbage (전혀 fitting 안 됨)
- 수렴: 5배 이상 느림 (10k → 50k iteration 필요)
- 최종 quality: 비슷 (수렴하면 같음)
- **주요 이슈**: Gaussian 이 제멋대로 배치 → densification 의 의미 불명확

**결론**: COLMAP geometric prior 가 수렴을 **엄청** 빠르게 함 (critical).

</details>

**문제 3** (논文批評): 왜 3DGS 는 LPIPS 에서 NeRF 보다 더 좋은 점수를 얻는가? 무엇이 "더 자연스러워" 보이는가?

<details>
<summary>해설</summary>

**가설들**:

1. **Explicit points**: local structure (edge, texture) 를 정확히 배치 → artifacts 적음
2. **No blur**: NeRF 의 ray-marching + MLP approximation error → slight blur. 3DGS 는 sharp
3. **Color independence**: SH 계수가 orthogonal → noise 적음 vs NeRF MLP 는 entangled
4. **High frequency**: Gaussian mixture 가 high-freq detail capture 우수

특히 **specular highlights** (반짝이) 와 **edges** (경계) 에서 3DGS 가 visually superior.

LPIPS (AlexNet-based) 도 이러한 detail 을 human-aligned 로 본다 → 점수 높음.

</details>

---

<div align="center">

[◀ 이전](./05-adaptive-density-control.md) | [📚 README](../README.md) | [다음 ▶](../ch5-dynamic-4d/01-deformable-nerf.md)

</div>
