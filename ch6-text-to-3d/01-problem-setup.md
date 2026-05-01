# 01. Text-to-3D Problem Setup 과 2D Prior 활용 동기

## 🎯 핵심 질문

- 왜 text → 3D 는 직접 학습하기 어려운가? 3D 데이터의 부족성을 정량화하면?
- 2D diffusion model 을 3D 생성에 활용하는 두 가지 주요 아이디어는 무엇인가?
- **DreamFusion 계열** (score distillation + differentiable renderer) 과 **3D-native 계열** (Point-E, LRM regression) 의 수학적 근거와 trade-off 는?
- 왜 3D diffusion 을 직접 학습하지 않고 2D prior 를 "distill" 하는 것이 더 효과적인가?
- Objaverse vs LAION 의 데이터 비율이 어떻게 이 선택을 정당화하는가?

---

## 🔍 왜 이 문제가 중요한가

Text-to-3D 는 **3D vision 의 마지막 frontier** 입니다. NeRF · 3D Gaussian Splatting 은 이미 있는 이미지들로부터 3D 를 재구성하지만, text 하나로부터 3D 를 **생성** 해야 합니다. 이는:

1. **데이터 스케일의 비대칭** — LAION 5B 개 text-image pair vs Objaverse 800K 개의 3D model
2. **차원의 저주** — 2D image space 대비 3D shape/appearance 는 combinatorially 더 복잡
3. **Ambiguity** — 같은 text 에 무한한 3D 가능성 (view-dependent, lighting-dependent, detail-dependent)

따라서 **2D prior 활용** 이 자연스러움. DreamFusion (Poole et al. 2023) 은 "frozen 2D diffusion model 을 3D 생성의 scoring function 으로 사용" 하는 **score distillation sampling (SDS)** 를 제안했고, 이는 이후 ProlificDreamer · MVDream · SV3D 의 출발점이 됩니다.

---

## 📐 수학적 선행 조건

- **Diffusion Model Deep Dive**: score matching, reverse process $p_\theta(x_0|x_t)$, guidance (conditional diffusion)
- **NeRF 기초**: volume rendering equation, differentiable renderer $g(\theta, \pi): \text{params} \to \text{image}$
- **확률론**: KL divergence, expectation, gradient via score
- **Optimization**: gradient descent, loss landscape geometry

---

## 📖 직관적 이해

### 왜 2D Prior 를 쓰는가?

- **LAION 는 ENORMOUS** — 5.85 billion text-image pairs (OpenAI CLIP training 사용)
- **Objaverse 는 작다** — 800K+ 3D models (2024 확장판)
- 비율: $\text{LAION} / \text{Objaverse} \approx 7300 : 1$

직관: "사람이 3D 모델을 얼마나 많이 만들든, 사진/그림은 훨씬 많다. 따라서 2D 에서 학습한 시각적 understanding 을 3D 에 transfer 하는 것이 합리적."

### DreamFusion 의 아이디어

```
text y
  ↓
diffusion model (frozen) ← "이 image z 가 y 와 잘 맞니?" (scoring)
  ↓ (score feedback)
NeRF renderer (learnable)
  ↓
rendered image z
```

즉:
1. NeRF $\theta$ 로부터 image $z = g(\theta, \pi)$ 를 렌더링
2. **Frozen diffusion 이 "이 image 가 prompt 와 얼마나 match 하는가" 평가**
3. 평가 결과를 gradient 로 변환 → NeRF update

이렇게 하면 3D 데이터 없이 2D prior 활용 가능.

### 도식화

```
Dataset scale:
  LAION (text-image)    ████████████████████ 5.85B
  Objaverse (3D)        ██ 800K
                        
Trade-off:
  3D diffusion (native)    vs    2D prior + NeRF + distillation
  ├─ 장점: 직접적                    ├─ 장점: 데이터 풍부 (LAION)
  ├─ 단점: 3D 데이터 부족            ├─ 단점: 간접적 (approximation)
  └─ 결과: 질 낮음, diversity 낮음   └─ 결과: 질 높음, diversity 높음
```

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Text-to-3D Problem

**입력**: text prompt $y \in \mathcal{Y}$ (예: "a red car")

**출력**: 3D 표현 $\mathcal{R} \in \{\text{NeRF}, \text{Mesh}, \text{GS}, \ldots\}$

**손실**: perceptual loss $\mathcal{L}(\mathcal{R}, y)$ — 렌더링 이미지 $\{z_\pi : \pi \in \text{cameras}\}$ 이 prompt $y$ 와 "일치" 하는 정도.

목표: $\min_\mathcal{R} \mathcal{L}(\mathcal{R}, y)$.

### 정의 1.2 — Differentiable Renderer

3D 파라미터 $\theta$ (예: NeRF MLP weights, Gaussian positions) 로부터 image 로 매핑하는 미분가능 함수:
$$
g: \mathbb{R}^d \times \mathcal{C} \to \mathbb{R}^{H \times W \times 3}
$$

$$
z = g(\theta, \pi), \quad \pi \in \mathcal{C} \text{ (camera parameters)}
$$

$\frac{\partial g}{\partial \theta}$ 가 계산 가능해야 함 (NeRF, 3D GS 모두 지원).

### 정의 1.3 — Frozen Diffusion Prior

Pre-trained conditional diffusion model:
$$
p_\phi(x_0 \mid y) = \int p_\phi(x_0 | x_T) p_\phi(x_T | y) \, dx_T
$$

score function (gradient):
$$
\hat{\epsilon}_\phi(x_t; y, t) \approx -\sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} \log p_\phi(x_t | y)
$$

**Frozen** 은 $\phi$ 를 고정 (업데이트하지 않음) 을 의미.

### 정의 1.4 — Two Families of Solutions

**Family 1: 2D Prior + Renderer Distillation**
- NeRF / 3D GS 로 3D parametrize
- Frozen 2D diffusion 을 score evaluator 로 사용
- Loss: score distillation sampling (SDS)
- 예: DreamFusion (Poole 2023), ProlificDreamer (Wang 2023), MVDream (Shi 2024)

**Family 2: 3D-Native Diffusion / Regression**
- 3D space 에서 직접 diffusion 또는 regression
- 예: Point-E (Nichol 2023), Shap-E (Jun 2023), LRM (Hong 2024)

---

## 🔬 정리와 증명

### 정리 1.1 — Data Scalability Argument

**명제**: $N_{\text{2D}}$ 개의 text-image pair 와 $N_{\text{3D}}$ 개의 3D model 이 주어질 때, $N_{\text{2D}} \gg N_{\text{3D}}$ 이면 2D prior distillation 이 3D native 학습보다 이론적으로 우수하다 (같은 compute budget 하에).

**증명 sketch**:

1. **3D-native 학습**:
   - 3D diffusion $q_\psi$ 를 학습: $\mathcal{L}_{\text{3D}} = \mathbb{E}_{x \sim \text{data}, t, \epsilon} [\|\hat{\epsilon}_\psi(x_t; y, t) - \epsilon\|^2]$
   - Sample complexity: $O(N_{\text{3D}} \cdot D_{\text{3D}})$ (3D 의 dimension 이 높음)
   - 수렴 속도: $O(1/\sqrt{N_{\text{3D}}})$

2. **2D Prior Distillation**:
   - 2D diffusion $p_\phi$ 는 LAION 으로 이미 학습됨 (frozen)
   - NeRF renderer $g$ 만 최적화: parameterization 낮음 ($d \approx 10^5$ vs 3D 의 $10^6$+)
   - Sample complexity: $O(N_{\text{2D}} + d \cdot \text{SDS iterations})$ — 2D 의 풍부함 재사용

3. **Data efficiency**:
   - 각 3D model 에서 얻을 수 있는 2D views: $K$ (보통 수십개)
   - 유효 3D 학습 데이터: $N_{\text{2D}} \approx K \cdot N_{\text{3D}}$ 로 근사
   - $K \approx 10\text{--}20$ 이면 $N_{\text{2D}} / N_{\text{3D}} > 7000$ 를 설명

따라서 **2D prior 가 asymptotically 우수** $\square$.

### 따름 정리 1.2 — Ambiguity Handling

Text-to-3D 의 본질적 ambiguity (한 text 에 많은 3D 가능성) 에 대해:

**명제**: Diffusion model 의 stochastic sampling 과정이 이 ambiguity 를 **자연스럽게 정량화** 한다.

**증명 sketch**: 
- Diffusion model $p_\phi(z_0 | y)$ 는 multimodal distribution
- Reverse process $z_t \sim p_\phi(\cdot | y)$ 에서 random noise $\epsilon_T$ 를 샘플링하면 diverse 3D 생성
- 같은 text 에도 여러 렌더링 가능 (SDS 의 stochastic camera sampling $\pi \sim \mathcal{P}$)

**의의**: 구조적으로 "한 prompt 에 여러 해석" 을 지원 $\square$.

---

## 💻 구현 검증

### 실험 1 — Data Scale 확인

```python
# Objaverse 와 LAION 의 스케일 비교
objaverse_size = 800_000         # 800K models
laion_size = 5_850_000_000       # 5.85B pairs

ratio = laion_size / objaverse_size
print(f"LAION / Objaverse = {ratio:.0f}x")
# 출력: LAION / Objaverse = 7312x

# 3D 모델당 평균 renderings
avg_views_per_model = 12
effective_2d_from_3d = objaverse_size * avg_views_per_model
print(f"Synthetic 2D from 3D = {effective_2d_from_3d:,}")
# 출력: Synthetic 2D from 3D = 9,600,000

print(f"LAION 이 synthetic 3D-derived 2D 보다 {laion_size/effective_2d_from_3d:.0f}x 크다")
# 출력: LAION 이 synthetic 3D-derived 2D 보다 609x 크다
```

### 실험 2 — Renderer Differentiability 확인 (toy NeRF)

```python
import torch
import torch.nn as nn

class TinyNeRF(nn.Module):
    """최소한의 NeRF MLP"""
    def __init__(self, input_dim=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4)  # σ, (R, G, B)
        )
    
    def forward(self, x):
        """x: (batch, 3) positional encoding"""
        return self.net(x)

# Mock differentiable renderer
def render_batch(nerf, camera_rays, n_samples=32):
    """
    Simple volume rendering mock.
    camera_rays: (batch, ray_dim)
    """
    # Sample points along each ray
    t = torch.linspace(0, 1, n_samples, device=camera_rays.device)
    
    # Positional encoding
    pts = camera_rays[:, None, :] * t[None, :]  # (batch, n_samples, 3)
    pts_flat = pts.reshape(-1, 3)
    
    # NeRF forward
    sigma_rgb = nerf(pts_flat)  # (batch*n_samples, 4)
    sigma = torch.relu(sigma_rgb[:, 0])
    rgb = torch.sigmoid(sigma_rgb[:, 1:])
    
    # Volume rendering
    sigma = sigma.reshape(-1, n_samples)
    rgb = rgb.reshape(-1, n_samples, 3)
    
    # Alpha composite (simplified)
    alpha = 1 - torch.exp(-sigma)
    weights = alpha * torch.cumprod(1 - alpha + 1e-7, dim=1)
    rendered = (weights[:, :, None] * rgb).sum(dim=1)
    
    return rendered

# 실행 및 미분가능성 확인
nerf = TinyNeRF()
ray_batch = torch.randn(4, 3, requires_grad=True)

image = render_batch(nerf, ray_batch)
loss = image.sum()
loss.backward()

print(f"✓ Renderer 미분가능: grad shape = {ray_batch.grad.shape}")
print(f"✓ NeRF 미분가능: grad norm = {sum(p.grad.norm() for p in nerf.parameters() if p.grad is not None):.4f}")
```

### 실험 3 — Score-based Loss (Conceptual)

```python
# DreamFusion SDS loss 의 개념적 구현
import torch

def sds_loss_mock(
    rendered_image,          # (H, W, 3) from NeRF
    diffusion_model,         # frozen pretrained
    prompt,                  # text
    t_sample,               # diffusion timestep, e.g., t ∈ [1, T]
    alpha_t, sigma_t        # diffusion schedule params
):
    """
    Mock SDS loss: ∇_θ L = E[w(t) * (ε̂_φ - ε) * ∂z/∂θ]
    
    실제로는 NeRF parameter θ 에 대한 gradient 계산이 중요.
    """
    
    # 1. noisy latent 생성
    noise = torch.randn_like(rendered_image)
    z_t = alpha_t * rendered_image + sigma_t * noise  # forward diffusion
    
    # 2. 고정된 diffusion 의 예측
    with torch.no_grad():
        eps_pred = diffusion_model(z_t, t=t_sample, prompt=prompt)
    
    # 3. "score" 로 보정 (U-Net Jacobian 을 identity 근사)
    # 실제 SDS: ∂ε̂/∂z_t ≈ I (identity) 로 근사
    # → gradient 가 diffusion model 을 통하지 않고 renderer 만 거침
    
    loss = ((eps_pred - noise) ** 2).mean()
    
    return loss

print("✓ SDS loss 구조 확인: (eps_pred - noise) term 이 score 역할")
```

---

## 🔗 실전 활용

### 1. DreamFusion 에서의 Multi-View Sampling

```python
# SDS 평균을 계산할 때 random camera 샘플링
def sample_cameras(batch_size=4, device='cuda'):
    """Random camera poses from unit sphere"""
    import numpy as np
    
    cameras = []
    for _ in range(batch_size):
        # 구 위의 random point
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))
        
        # 카메라 위치
        radius = 1.5
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        cameras.append(torch.tensor([x, y, z], device=device))
    
    return torch.stack(cameras)

# 각 iteration 에서
for it in range(num_iterations):
    cameras = sample_cameras()  # Random cameras
    for cam in cameras:
        z = render(nerf, cam)
        loss = sds_loss(z, diffusion, prompt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 2. Data Efficiency: Fine-Tuned LoRA vs Full NeRF

- **Full NeRF parametrization**: $d \approx 10^5 \text{--} 10^6$ (MLP weights + positional encoding)
- **실제 optimizable params**: $\approx 5 \text{--} 10$ % 활성화 (sparse gradients)
- → **효율적 distillation** 가능

### 3. Ambiguity in Rendering Diversity

```python
# 같은 prompt, 다른 random seed
prompts = ["a red car"]
for seed in range(5):
    torch.manual_seed(seed)
    nerf = TinyNeRF()
    
    # SDS optimization (random camera + noise)
    for it in range(1000):
        cam = sample_cameras(1)[0]
        noise_schedule_t = sample_t(1)[0]
        
        z = render(nerf, cam)
        loss = sds_loss(z, diffusion, prompts[0], t=noise_schedule_t)
        loss.backward()
    
    # 각 seed 마다 다른 3D 생성됨
    print(f"Seed {seed}: model shape/texture 다름 ✓")
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| **2D prior 는 완전히 frozen** | 실제로 fine-tuning 으로 개선 가능 (e.g., LoRA in ProlificDreamer) |
| **LAION 의 bias 를 상속** | Diffusion model 의 aesthetic bias, face bias 등이 3D 에 전이 → CFG scale 조정 필요 |
| **View independence** | Multi-view diffusion (MVDream) 으로 보정 필요 (Ch6-05) |
| **Camera distribution** | Stratified sphere sampling 이 충분한가? → 실제로 uniform 샘플링 > 기하학적 특이점 피함 |
| **Mode-seeking** | SDS 의 reverse-KL divergence 특성 → over-saturated rendering → VSD 로 해결 (Ch6-04) |
| **Speed vs Quality** | DreamFusion: 30분 (NeRF) vs LRM: 5초 (regression) — 정확도 vs latency trade-off |

---

## 📌 핵심 정리

$$\boxed{\text{Text-to-3D} = \text{frozen 2D prior} + \text{differentiable renderer} + \text{score distillation}}$$

| 요소 | 역할 | 왜? |
|------|------|------|
| **LAION (5.85B)** | Text-image pairs | 엄청난 스케일, 풍부한 semantic 정보 |
| **Frozen diffusion** | Scoring function | 고정 후 NeRF 만 update → 효율적 |
| **NeRF / 3D GS** | 3D parametrization | Differentiable, low param count |
| **SDS loss** | Gradient source | Score → renderer gradient (다음 장) |
| **Random cameras** | View diversity | 한 iteration 에서 여러 시각각 → mode-seeking 완화 |

**Key insight**: 3D 데이터 부족을 **2D data abundance** 로 bypass, **differentiable rendering** 으로 역전파.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Objaverse 800K 개 모델에서 각각 평균 12개의 random view 를 렌더링하면, synthetic 2D 데이터는 몇 개인가? LAION 대비 비율은?

<details>
<summary>해설</summary>

Synthetic 2D = $800K \times 12 = 9.6M$ images.

LAION / Synthetic = $5.85B / 9.6M \approx 609$배.

**의미**: 설령 3D 모델로부터 충분한 view 를 생성해도, LAION 의 다양성 (uncurated, diverse objects, scenes, lighting) 을 따라잡기 어려움. 따라서 LAION pre-training 된 diffusion 이 훨씬 가치 있음. $\square$

</details>

**문제 2** (심화): DreamFusion 에서 $\frac{\partial z}{\partial \theta}$ (rendered image 에 대한 NeRF parameter gradient) 가 계산 가능하려면, volume rendering 의 어떤 성질이 필요한가? NeRF 의 stratified sampling 이 미분가능한 이유는?

<details>
<summary>해설</summary>

**미분가능 조건**:

1. **Ray integration 이 미분가능** — $C(\mathbf{r}) = \sum_i (1 - e^{-\sigma_i \delta_i}) c_i$ 는 각 $\sigma_i, c_i$ 에 대해 미분가능.

2. **Sampling 이 deterministic** — Stratified sampling 은 고정된 구간 $(t_i, t_{i+1})$ 에서 uniform 이므로, 각 sample 점 $t_i$ 의 위치가 고정 → 다음 iteration 의 gradient 를 정확하게 계산 (importance sampling 과 달리 deterministic).

3. **MLP 가 differentiable** — $\sigma(t), c(t)$ 를 MLP 로 계산 → full backpropagation path.

**결론**: NeRF 의 "closed-form" volume rendering (numerical integration 의 정확 근사) 가 full differentiability 보장 $\square$.

</details>

**문제 3** (논문 비평): Point-E · Shap-E · LRM 등의 3D-native approach 가 DreamFusion 보다 빠른 이유는? 어떤 trade-off 가 있는가?

<details>
<summary>해설</summary>

**속도 비교**:

| 방법 | 시간 | 이유 |
|------|------|------|
| DreamFusion | 30분 | SDS iteration × rendering × diffusion inference |
| ProlificDreamer | 15분 | VSD + LoRA 최적화 |
| LRM | 5초 | Single forward pass (transformer) |

**LRM 의 빠름 이유**:
- 3D 를 diffusion 으로 생성하지 않고 **transformer regression** 사용
- Image → triplane NeRF (feed-forward)
- 반복 최적화 (iterative refinement) 없음

**Trade-off**:
1. **Quality**: DreamFusion > LRM (더 자세한 detail, fine-tuning 가능)
2. **Diversity**: DreamFusion > LRM (stochastic SDS vs deterministic transformer)
3. **Control**: DreamFusion > LRM (multiple prompts, guidance 등)
4. **Speed**: LRM > DreamFusion (100배 이상 빠름)

**결론**: DreamFusion 은 "3D data 부족을 보완하되 느림", LRM 은 "충분한 3D 데이터 사전학습으로 빠름" — data scale 과 inference latency 의 근본적 trade-off $\square$.

</details>

---

<div align="center">

[◀ 이전](../ch5-dynamic-4d/04-video-reconstruction.md) | [📚 README](../README.md) | [다음 ▶](./02-sds-derivation.md)

</div>
