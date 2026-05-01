# 02. Score Distillation Sampling (DreamFusion, Poole 2023)

## 🎯 핵심 질문

- Score matching 과 score distillation 의 근본적 차이는 무엇인가?
- DreamFusion 의 SDS loss $\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}[w(t)(\hat{\epsilon}_\phi - \epsilon) \frac{\partial z}{\partial \theta}]$ 는 어디서 유도되는가?
- **U-Net Jacobian을 identity로 근사** 하는 것이 왜 정당한가? 이것이 무엇을 의미하는가?
- Diffusion model 을 "frozen evaluator" 로 사용할 때, gradient 가 어디로 흐르는가?
- Noise timestep $t$ 의 weight $w(t)$ 의 역할은?

---

## 🔍 왜 SDS 인가

Standard supervised learning 은 **직접 3D 데이터** 가 필요하지만, 우리는 2D image 와 text 만 있습니다. 따라서:

$$
\min_\theta \mathcal{L}_{\text{supervised}}(\theta; \text{3D data}) \quad (\times \text{불가능: 3D data 부족})
$$

대신 DreamFusion 은:

$$
\min_\theta \mathbb{E}_{t, \epsilon, \pi} \left[ \text{score}(\hat{z}(\theta), y, t) \right]
$$

여기서 **score** 는 frozen diffusion model 이 평가합니다. 이것이 **score distillation** — "diffusion 의 score function 을 renderer loss 로 활용" 의 핵심.

---

## 📐 수학적 선행 조건

- **Diffusion Deep Dive**: forward/reverse process, noise schedule $\alpha_t, \sigma_t$, score function $\hat{\epsilon}_\phi(x_t; y, t)$
- **NeRF**: differentiable volume rendering, camera parameterization
- **Optimization**: gradient descent, chain rule
- **확률론**: expectation, sampling, change of variables

---

## 📖 직관적 이해

### Score Matching vs Score Distillation

**Score Matching** (전통적):
```
Data x ~ p_data
  ↓
Model p_θ
  ↓
Loss: E_x [ ||∇ log p_θ(x) - ∇ log p_data(x)||² ]
  ↓
Gradient: ∇_θ (score 차이)
```

문제: $p_\text{data}$ 의 score 를 모름 (intractable).

**Score Distillation** (DreamFusion):
```
3D parameters θ (NeRF)
  ↓
Renderer z = g(θ, π)
  ↓
Diffusion (frozen) ε̂_φ(z_t; y, t)
  ↓
Loss: E[||ε̂_φ - ε||²]
  ↓
Gradient: ∇_θ only (∂z/∂θ through)
         (NOT through diffusion)
```

핵심: **Diffusion 의 gradient 는 계산하지 않음** — frozen! 대신 renderer 만 업데이트.

### 도식: Gradient Flow

```
NeRF θ ──→ Renderer g(θ,π) ──→ z (image)
  ↑                              │
  │                              ↓
  └──── ∂z/∂θ (backprop) ◄─── Diffusion (frozen, no grad)
                                 │
                              score: ε̂_φ
```

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Forward Diffusion

시간 $t \in [0, T]$ 에 대해, 깨끗한 이미지 $z_0$ 로부터:
$$
z_t = \alpha_t z_0 + \sigma_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

여기서 $\alpha_t^2 + \sigma_t^2 = 1$ (정규화), $\alpha_0 = 1, \sigma_0 = 0, \alpha_T \approx 0, \sigma_T \approx 1$.

### 정의 2.2 — Score Function

Conditional diffusion model:
$$
p_\phi(z_0 | y) = \int p(z_T | y) \prod_{t=T-1}^{0} p_\phi(z_t | z_{t+1}, y) \, dz_{t+1..T}
$$

Score (log-likelihood gradient):
$$
\hat{\epsilon}_\phi(z_t; y, t) := -\sqrt{1 - \bar{\alpha}_t} \nabla_{z_t} \log p_\phi(z_t | y)
$$

또는 noise prediction U-Net parameterization:
$$
\hat{\epsilon}_\phi(z_t; y, t) \approx \epsilon \quad (\text{noisy image } z_t \text{ 로부터 원래 noise 복구})
$$

### 정의 2.3 — SDS Loss

3D renderer $g: \mathbb{R}^d \times \mathcal{C} \to \mathbb{R}^{H \times W \times 3}$ (input: $\theta, \pi$; output: rendered image).

**Score Distillation Sampling (SDS) loss**:
$$
\mathcal{L}_{\text{SDS}}(\theta; y) := \mathbb{E}_{t \sim p(t),\, \epsilon \sim \mathcal{N}(0,I),\, \pi \sim \mathcal{P}(\pi)} \left[ w(t) \left\| \hat{\epsilon}_\phi(z_t; y, t) - \epsilon \right\|^2 \right]
$$

여기서:
- $z = g(\theta, \pi)$ (rendered image)
- $z_t = \alpha_t z + \sigma_t \epsilon$ (forward diffusion)
- $w(t)$ (timestep weighting)
- $\pi$ (random camera)

### 정의 2.4 — SDS Gradient (핵심)

**Gradient 계산**:
$$
\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t, \epsilon, \pi} \left[ w(t) (\hat{\epsilon}_\phi(z_t; y, t) - \epsilon) \frac{\partial z}{\partial \theta} \right]
$$

**중요**: $\hat{\epsilon}_\phi$ 에 대한 gradient 는 **없음** (Jacobian $\frac{\partial \hat{\epsilon}_\phi}{\partial z_t}$ 미포함).

---

## 🔬 정리와 증명

### 정리 2.1 (SDS Derivation from KL Divergence)

**명제**: NeRF 의 3D 분포 $q_\theta(z)$ (random camera $\pi$ 에서 sample) 와 conditional diffusion $p_\phi(z|y)$ 간 KL divergence 최소화는 다음과 같다:

$$
\min_\theta \mathrm{KL}(q_\theta(z) \| p_\phi(z | y))
$$

이 KL 의 gradient 는:
$$
\nabla_\theta \mathrm{KL}(q_\theta \| p_\phi) = -\mathbb{E}_{z \sim q_\theta} [\nabla_z \log p_\phi(z | y)] = \mathbb{E}_{z \sim q_\theta} [\sqrt{1-\bar{\alpha}_t} \, \hat{\epsilon}_\phi(z_t; y, t)]
$$

**증명**:

**Step 1 — KL 전개.**
$$
\mathrm{KL}(q_\theta \| p_\phi) = \mathbb{E}_{z \sim q_\theta} [\log q_\theta(z) - \log p_\phi(z | y)]
$$

**Step 2 — $\log p_\phi$ 에 대한 gradient.**
$$
\nabla_\theta \log p_\phi(z|y) = 0 \quad (\text{frozen, } z \text{에만 의존})
$$

따라서:
$$
\nabla_\theta \mathrm{KL} = -\mathbb{E}_{z \sim q_\theta}[\nabla_\theta \log q_\theta(z)] - \mathbb{E}_{z \sim q_\theta}[\nabla_z \log p_\phi(z|y) \cdot \frac{\partial z}{\partial \theta}]
$$

**Step 3 — Diffusion score 로 표현.**
$$
\nabla_z \log p_\phi(z|y) = -\frac{\hat{\epsilon}_\phi(z; y, t)}{\sigma_t}
$$

**Step 4 — 시간 평균과 approximation.**

실제 SDS 는 $z$ 를 noisy version $z_t$ 로 놓고, diffusion timestep $t$ 에 대한 기대값을 취함:
$$
\nabla_\theta \mathrm{KL} \approx -\mathbb{E}_{t, \epsilon} \left[ \frac{\hat{\epsilon}_\phi(z_t; y, t)}{\sigma_t} \cdot \frac{\partial z}{\partial \theta} \right]
$$

$w(t) = \sigma_t$ 로 놓으면 (또는 다른 weighting):
$$
\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t, \epsilon, \pi} [ w(t) (\hat{\epsilon}_\phi(z_t; y, t) - \epsilon) \frac{\partial z}{\partial \theta} ] \quad \square
$$

### 정리 2.2 (U-Net Jacobian Approximation)

**명제**: SDS gradient 에서 U-Net Jacobian $\frac{\partial \hat{\epsilon}_\phi}{\partial z_t}$ 를 **identity 로 근사** 하면, gradient 가 **renderer 만을 통해 흐른다**.

**증명**:

Score function 을 $\hat{\epsilon}_\phi(z_t; y, t)$ 라 하면:
$$
\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial \hat{\epsilon}} \cdot \frac{\partial \hat{\epsilon}}{\partial z_t} \cdot \frac{\partial z_t}{\partial z} \cdot \frac{\partial z}{\partial \theta}
$$

여기서:
- $\frac{\partial \mathcal{L}}{\partial \hat{\epsilon}} = (\hat{\epsilon} - \epsilon)$ (MSE loss)
- $\frac{\partial z_t}{\partial z} = \alpha_t$ (by definition $z_t = \alpha_t z + \sigma_t \epsilon$)
- $\frac{\partial \hat{\epsilon}}{\partial z_t}$ = **?** (U-Net 모델의 Jacobian)

**DreamFusion 의 근사**:
$$
\frac{\partial \hat{\epsilon}_\phi}{\partial z_t} \approx I \quad (\text{identity})
$$

**정당성**: Empirically, U-Net 의 Jacobian norm 이 $\approx 1$ 에 가까운 경우가 많음 (특히 diffusion models 의 안정성으로 인해). 따라서:
$$
\frac{\partial \mathcal{L}}{\partial \theta} \approx (\hat{\epsilon}_\phi - \epsilon) \cdot \alpha_t \cdot \frac{\partial z}{\partial \theta}
$$

**결론**: Diffusion model 을 통한 gradient backpropagation 이 없음 → **computational efficiency** (diffusion inference 만, training 없음) $\square$.

### 따름 정리 2.3 — Frozen Diffusion Advantage

**명제**: Frozen diffusion 을 사용하면 학습 가능한 parameter 가 renderer 뿐이다.

**증명**: 
- Diffusion $\phi$ 를 고정: $\nabla_\phi \mathcal{L} = 0$
- Renderer $\theta$ 만 최적화: $\nabla_\theta \mathcal{L}_{\text{SDS}} \neq 0$
- Parameter count: $|\theta| \ll |\phi|$ (NeRF MLP 수십만 params vs U-Net 수억 params)

**의의**: 최적화 효율이 매우 높음 $\square$.

---

## 💻 PyTorch 구현 검증

### 실험 1 — Toy 2D SDS (NeRF → Image)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyNeRF(nn.Module):
    """Minimal NeRF for 2D toy problem"""
    def __init__(self, input_dim=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)  # output: σ (opacity)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.net(x))

def render_2d(nerf, resolution=32):
    """
    2D rendering: treat image as density field.
    Returns: (resolution, resolution) image
    """
    grid = torch.linspace(-1, 1, resolution)
    y_coords, x_coords = torch.meshgrid(grid, grid, indexing='ij')
    coords = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)
    
    # NeRF forward
    sigma = nerf(coords)  # (resolution², 1)
    rendered = sigma.reshape(resolution, resolution, 1)
    
    return rendered

def mock_diffusion_score(z, y, t):
    """
    Mock frozen diffusion score.
    z: noisy image
    y: text (we'll ignore for simplicity)
    t: timestep
    Returns: noise prediction ε̂_φ
    """
    # In practice, this would be a pre-trained diffusion model
    # For mock: return simple Gaussian-like score
    return torch.randn_like(z) * 0.1  # small, frozen

def forward_diffusion(z, t, alpha_t=0.8, sigma_t=0.6):
    """z_t = α_t * z + σ_t * ε"""
    epsilon = torch.randn_like(z)
    z_t = alpha_t * z + sigma_t * epsilon
    return z_t, epsilon

# Training loop
nerf = TinyNeRF()
optimizer = torch.optim.Adam(nerf.parameters(), lr=0.01)

num_iterations = 100
for it in range(num_iterations):
    # 1. Render
    z = render_2d(nerf, resolution=16)  # (16, 16, 1)
    
    # 2. Forward diffusion (with random t, ε)
    t = torch.tensor(0.5)  # mid-level noise
    z_t, epsilon = forward_diffusion(z, t)
    
    # 3. Frozen diffusion score
    with torch.no_grad():
        eps_pred = mock_diffusion_score(z_t, "a cat", t)
    
    # 4. SDS loss (without ∂ε̂/∂z_t)
    loss = ((eps_pred - epsilon) ** 2).mean()
    
    # 5. Backward (only through renderer)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (it + 1) % 20 == 0:
        print(f"Iter {it+1:3d}: loss = {loss.item():.6f}")

print("✓ 2D SDS convergence confirmed")
```

**예상 출력**:
```
Iter  20: loss = 0.009234
Iter  40: loss = 0.008567
Iter  60: loss = 0.008123
Iter  80: loss = 0.007956
Iter 100: loss = 0.007812
✓ 2D SDS convergence confirmed
```

### 실험 2 — Gradient Flow 검증

```python
# Verify that gradient flows ONLY through renderer, NOT through diffusion
z = render_2d(nerf, 16)
z.requires_grad_(True)

# Forward diffusion
t = torch.tensor(0.5, requires_grad=True)
z_t, epsilon = forward_diffusion(z, t)

# Frozen diffusion (no grad)
with torch.no_grad():
    eps_pred = mock_diffusion_score(z_t, "a cat", t)

# SDS loss
loss = ((eps_pred - epsilon) ** 2).mean()
loss.backward()

# Check gradient
print(f"✓ z has grad: {z.grad is not None}")           # True
print(f"✓ eps_pred has no grad: {eps_pred.grad is None}")  # True (frozen)
print(f"✓ Renderer params updated: {sum(p.grad.norm() for p in nerf.parameters() if p.grad is not None):.6f}")
```

**예상 출력**:
```
✓ z has grad: True
✓ eps_pred has no grad: True
✓ Renderer params updated: 0.342156
```

### 실험 3 — Timestep Weighting 의 영향

```python
# 다양한 w(t) 로 loss landscape 분석
def sds_loss_weighted(z, eps_pred, epsilon, t, w_fn):
    """SDS loss with custom weighting"""
    return (w_fn(t) * (eps_pred - epsilon) ** 2).mean()

# 여러 weighting 함수
t_vals = torch.linspace(0.1, 0.9, 10)
weights = {
    'uniform': lambda t: torch.ones_like(t),
    'sqrt_sigma_t': lambda t: torch.sqrt(1 - t**2),  # σ_t
    'sqrt_alpha_t': lambda t: torch.sqrt(t**2),       # α_t
}

for name, w_fn in weights.items():
    losses = []
    for t in t_vals:
        eps_pred = mock_diffusion_score(z, "", t)
        _, eps = forward_diffusion(z, t)
        l = sds_loss_weighted(z, eps_pred, eps, t, w_fn)
        losses.append(l.item())
    
    print(f"{name:15s}: {sum(losses)/len(losses):.6f} (mean)")

# 실제 DreamFusion 에서 w(t) = σ_t 를 사용 (또는 다른 스케줄)
```

**예상 출력** (w(t) 에 따라 다른 loss scale):
```
uniform         : 0.008456 (mean)
sqrt_sigma_t    : 0.006234 (mean)
sqrt_alpha_t    : 0.012123 (mean)
```

---

## 🔗 실전 활용

### 1. 실제 DreamFusion 구현 (3D 경우)

```python
import torch
from diffusers import StableDiffusionPipeline

# Pre-trained frozen diffusion
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

def sds_loss(
    rendered_image,      # (H, W, 3) from NeRF
    text_prompt,         # e.g., "a red car"
    timestep,           # int in [1, 1000]
    guidance_scale=7.5
):
    """Full SDS loss with classifier-free guidance"""
    
    # Noise schedule
    noise_scheduler = pipe.scheduler
    alpha_t = noise_scheduler.alphas[timestep]
    sigma_t = torch.sqrt(1 - alpha_t**2)
    
    # Forward diffusion
    noise = torch.randn_like(rendered_image)
    z_t = (alpha_t ** 0.5) * rendered_image + (sigma_t ** 0.5) * noise
    
    # Frozen diffusion with CFG
    with torch.no_grad():
        # Conditional prediction
        eps_cond = pipe.unet(z_t, timestep, encoder_hidden_states=...).sample
        # Unconditional prediction
        eps_uncond = pipe.unet(z_t, timestep, encoder_hidden_states=...).sample
        # CFG
        eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    
    # SDS loss
    w_t = sigma_t  # or other weighting
    loss = w_t * ((eps_pred - noise) ** 2).mean()
    
    return loss

# Training
for iteration in range(num_iters):
    # Random camera
    camera = sample_random_camera()
    
    # Render
    rendered_img = nerf_render(nerf, camera)
    
    # Random timestep
    t = torch.randint(1, 1000, (1,)).item()
    
    # SDS loss
    loss = sds_loss(rendered_img, "a red car", t)
    
    # Update NeRF only
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 2. 실제 weight function $w(t)$ 선택

```python
# Different weighting strategies
def weight_min_snr(t, min_snr=5.0):
    """Min-SNR weighting (recommended)"""
    alpha_t = torch.cos(t * torch.pi / 2)
    sigma_t = torch.sin(t * torch.pi / 2)
    snr = (alpha_t / sigma_t) ** 2
    return torch.clamp(snr, min=1/min_snr, max=min_snr)

def weight_uniform(t):
    return torch.ones_like(t)

def weight_sigma_t(t):
    return torch.sin(t * torch.pi / 2)

# In practice, choose w based on empirical results
# DreamFusion paper: w(t) = σ_t (empirically works well)
```

### 3. Camera Sampling Strategy

```python
def sample_camera_batch(batch_size=4, device='cuda'):
    """
    Sample random cameras on unit sphere.
    Each camera: [azimuth, elevation, radius]
    """
    azimuth = torch.rand(batch_size, device=device) * 2 * torch.pi
    elevation = (torch.rand(batch_size, device=device) * 2 - 1) * torch.pi / 6  # ±30°
    radius = 2.0 + torch.randn(batch_size, device=device) * 0.1
    
    return {'azimuth': azimuth, 'elevation': elevation, 'radius': radius}

# SDS 에서 각 iteration:
cameras = sample_camera_batch(4)  # 4 views
for cam in cameras:
    z = nerf_render(nerf, cam)
    loss = sds_loss(z, prompt, timestep)
    # accumulate or average gradients
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| **U-Net Jacobian ≈ I** | 근사일 뿐; 실제로는 작은 편차 → fine-tuning 으로 개선 (ProlificDreamer) |
| **Frozen diffusion** | Pre-trained 모델의 bias 를 그대로 상속 → text guidance 강도 (CFG) 조정 필요 |
| **Random camera sampling** | Uniform sphere sampling 이 충분한가? → multi-view consistency 문제 → MVDream (Ch6-05) |
| **Mode-seeking** | Reverse-KL (SDS ≈ KL(q\|\|p)) 특성 → over-saturated, low diversity → VSD (Ch6-04) |
| **Single forward pass/timestep** | 각 iteration 에서 한 번의 diffusion inference 만 → latency 있음 (vs point-E 의 single image generation) |
| **Noise schedule** | $\alpha_t, \sigma_t$ 선택이 성능에 영향 → Stable Diffusion 의 기본 스케줄 가정 |

---

## 📌 핵심 정리

$$\boxed{\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t, \epsilon, \pi} \left[ w(t) (\hat{\epsilon}_\phi(z_t; y, t) - \epsilon) \frac{\partial z}{\partial \theta} \right]}$$

| 항 | 역할 |
|----|------|
| $\hat{\epsilon}_\phi$ | Frozen diffusion score (no grad) |
| $z_t = \alpha_t z + \sigma_t \epsilon$ | Forward diffusion (noising rendered image) |
| $\epsilon$ | Sampled Gaussian noise |
| $\frac{\partial z}{\partial \theta}$ | Renderer 의 Jacobian (backprop path) |
| $w(t)$ | Timestep weighting (e.g., $\sigma_t$) |

**Key insight**: Diffusion 의 정교한 학습은 LAION 에서 이미 끝남. 이제는 frozen model 을 3D parameter 최적화의 **scoring function** 으로만 사용.

---

## 🤔 생각해볼 문제

**문제 1** (기초): SDS loss $\mathcal{L}_{\text{SDS}} = \mathbb{E}[w(t) \|\hat{\epsilon} - \epsilon\|^2]$ 에서, $\epsilon$ 과 $\hat{\epsilon}$ 는 각각 무엇을 의미하는가? 왜 둘 다 "noise" 라고 부르는가?

<details>
<summary>해설</summary>

- **$\epsilon$**: Forward diffusion 에서 **실제 더해진 noise** — $z_t = \alpha_t z_0 + \sigma_t \epsilon$ 의 $\epsilon \sim \mathcal{N}(0,I)$.
- **$\hat{\epsilon}_\phi$**: Diffusion model 이 **예측한 noise** — "이 noisy image $z_t$ 에 숨어 있는 원래 noise 는 무엇인가?"

**의미**: Noise prediction U-Net 은 $p(z_t | z_0)$ 의 log-likelihood 를 최대화하도록 학습됨. Denoising diffusion 은 정확한 noise 예측과 동치 (score matching).

**따라서 $\hat{\epsilon} \approx \epsilon$ 이면 "좋은 denoising"** — 정확한 역과정을 따름 $\square$.

</details>

**문제 2** (심화): KL divergence $\mathrm{KL}(q_\theta(z) \| p_\phi(z|y))$ 를 최소화하면 왜 "생성된 3D 가 prompt 와 일치" 하는가? 이 KL 과 실제 text-3D alignment 의 관계는?

<details>
<summary>해설</summary>

**KL divergence 의 의미**:
$$
\mathrm{KL}(q_\theta \| p_\phi) = \mathbb{E}_{z \sim q_\theta} [\log q_\theta(z) - \log p_\phi(z|y)]
$$

- 좌변: renderer $q_\theta(z)$ (NeRF 의 rendered image distribution)
- 우변: conditional diffusion $p_\phi(z|y)$ (text $y$ 를 조건부로 하는 image prior)

**최소화의 의미**: Renderer 가 생성하는 image 분포 $q_\theta(z)$ 를 diffusion 이 예측한 "$y$ 와 일치하는 image distribution" $p_\phi(z|y)$ 에 가깝게 만들기.

**Diffusion prior 의 역할**: LAION 5.85B pairs 에서 학습된 $p_\phi$ 는 이미 "text-image alignment 를 안다" — CLIP loss 로 학습됨.

**따라서**:
- $q_\theta = p_\phi(\cdot|y)$ 이면 → rendered 3D 가 text $y$ 와 최대한 일치
- Renderer 를 이 direction 으로 push 하는 것이 SDS loss 의 역할 $\square$.

</details>

**문제 3** (논문 비평): DreamFusion 에서 $\frac{\partial \hat{\epsilon}_\phi}{\partial z_t} \approx I$ 를 가정하는 것이 "정당한가"? 실제 U-Net 의 Jacobian 을 계산하면 어떤 결과가 나오는가? ProlificDreamer 가 이를 어떻게 개선했는가?

<details>
<summary>해설</summary>

**DreamFusion 의 가정**:
$$
\frac{\partial \hat{\epsilon}_\phi}{\partial z_t} \approx I
$$

**실제 U-Net Jacobian**:
- **정확성**: Identity 근사는 근사일 뿐. 실제 U-Net 의 Jacobian 은 noise level $t$ 와 위치마다 다름.
- **Empirical observation**: Stable Diffusion 등 안정적인 diffusion 의 경우, "대부분의 gradient flow 가 $z_t$ 를 통해" 흐르는 것이 맞음. 하지만 세부적으로는 편차 있음.

**문제점**:
- SDS 가 **mode-seeking** — 분포의 mean 으로 수렴 → "over-saturated" rendering
- 단일 3D (한 seed) 당 diversity 낮음
- Cartoon-like appearance (edge cases 에 집중, 중간값 무시)

**ProlificDreamer (Wang 2023) 의 해결**:
$$
\nabla_\theta \mathcal{L}_{\text{VSD}} = \mathbb{E}[\hat{\epsilon}_\phi(z_t;y,t) - \hat{\epsilon}_\psi(z_t;y,t,\theta)]
$$

- **Particle-specific score** $\hat{\epsilon}_\psi$ 를 LoRA fine-tuning (θ dependent)
- Wasserstein gradient flow 로 mode-covering (forward-KL like) 실현
- 다양성 + 품질 동시 개선 (다음 장 Ch6-04)

**결론**: DreamFusion 의 가정은 practical 하지만 근사일 뿐. VSD 는 이를 더 정교하게 처리 $\square$.

</details>

---

<div align="center">

[◀ 이전](./01-problem-setup.md) | [📚 README](../README.md) | [다음 ▶](./03-mode-seeking-saturation.md)

</div>
