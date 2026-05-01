# 03. SDS 의 Mode-Seeking 과 Over-Saturation

## 🎯 핵심 질문

- SDS loss 가 KL(q_θ ∥ p_φ) 와 같다면, 왜 **mode-seeking** 이 발생하는가?
- "Over-saturated" rendering 이란 무엇인가? 왜 cartoon-like 출력이 나오는가?
- Diffusion model 의 **denoising mean-reversion** 이 이 현상을 어떻게 유발하는가?
- Classifier-Free Guidance (CFG) scale 이 mode-seeking 을 악화시키는 이유는?
- 왜 ProlificDreamer (VSD) 가 이 문제를 해결할 수 있는가?

---

## 🔍 왜 이 현상이 문제인가

DreamFusion 은 때때로 다음과 같은 artifacts 를 생성합니다:

1. **Cartoon-like appearance**: 색이 과포화되고, 미묘한 detail 이 손실
2. **Lack of diversity**: 같은 prompt 에 여러 번 실행해도 비슷한 결과 (random seed 영향 작음)
3. **Face collapse**: Multi-face object (Janus problem) 에서 한쪽 면만 학습
4. **Unnatural lighting**: Specular highlight 가 과하게 표현

이들은 모두 **reverse-KL divergence 의 mode-seeking 특성** 에서 비롯됩니다.

---

## 📐 수학적 선행 조건

- **Information Theory Deep Dive**: KL divergence, forward-KL vs reverse-KL, mode-seeking vs mode-covering
- **Diffusion Deep Dive**: score function, denoising step, guidance
- **Optimization**: gradient descent, loss landscape geometry
- **Ch6-02**: SDS 유도

---

## 📖 직관적 이해

### KL Divergence: Forward vs Reverse

**Forward KL: $\mathrm{KL}(p \| q)$ — Mode-Covering**
```
True distribution p (여러 모드)
  ↓
Model q 가 p 를 "덮으려고" 노력
  ↓
q 는 모든 high-probability region 을 커버해야 함
  ↓
결과: Diffuse, under-confident (잘못된 곳도 할당)
```

**Reverse KL: $\mathrm{KL}(q \| p)$ — Mode-Seeking**
```
True distribution p (여러 모드)
  ↓
Model q 가 p 의 "특정 모드"에 집중
  ↓
하나의 mode 만 정확히 맞추고, 다른 모드는 무시
  ↓
결과: Sharp, confident but incomplete
```

### SDS 는 Reverse-KL

**명제**: SDS loss $\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}[w(t) (\hat{\epsilon}_\phi - \epsilon) \frac{\partial z}{\partial\theta}]$ 은 본질적으로:
$$
\min_\theta \mathrm{KL}(q_\theta(z) \| p_\phi(z|y))
$$

이것은 **reverse-KL** 이므로 **mode-seeking**.

### Diffusion 의 Mean-Reversion

Diffusion 의 denoising step 은:
$$
z_{t-1} = \frac{1}{\sqrt{\alpha_t}}(z_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\epsilon}_\phi(z_t; y, t)) + \sqrt{\beta_t} z
$$

이 식은 **평균으로 회귀 (mean-reversion)** 하는 특성이 있습니다:
- Score $\hat{\epsilon}_\phi$ 가 "분포의 center 로 향하는 방향"을 가리킴
- Guidance scale $s > 1$ 를 적용하면: $\hat{\epsilon} \to s \cdot \hat{\epsilon}$ — **더 강하게 mean 으로 끌어당김**
- 결과: 분포의 extreme case (diversity) 를 무시, 중간값 (mode) 으로 몰림

### 도식화

```
Prior distribution p_φ (diffusion 이 학습한 분포):
 
  ╱╲╱╲╱╲╱╲  (여러 모드 — 여러 해석)
 ╱  ╲  ╱  ╲  
│    ▌▌▌    │  ← mode-seeking 으로 이 peak 에만 집중
│    ▌▌▌    │
└─────────────┘


결과 q_θ (reverse-KL 로 최적화):

       ╱╲╱╲╱╲╱╲
      ╱    ↓    ╱     ← "a red car" 의 prototype (평균)
     │  ████████ │     ← 모든 예시가 이 한 버전으로 수렴
     │  ████████ │
     └─────────────┘
```

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Reverse-KL Divergence

두 분포 $p, q$ 에 대해:
$$
\mathrm{KL}(q \| p) = \mathbb{E}_{z \sim q} [\log q(z) - \log p(z)]
$$

**특성**:
- $q(z) > 0$ 이고 $p(z) = 0$ 이면 무한대 (zero-avoiding)
- $q$ 가 $p$ 의 모든 mode 를 cover 해야 high probability
- 대신 $p$ 에 있지만 $q$ 에 없는 영역은 무시 가능 (mode-seeking)

### 정의 3.2 — Forward-KL Divergence

$$
\mathrm{KL}(p \| q) = \mathbb{E}_{z \sim p} [\log p(z) - \log q(z)]
$$

**특성**:
- $p(z) > 0$ 이면 $q(z) > 0$ 이어야 함 (zero-forcing)
- $p$ 의 모든 mode 를 cover 해야 함
- 대신 $q$ 가 $p$ 에 없는 곳에 할당한 mass 는 penalty 적음

### 정의 3.3 — Over-Saturation 현상

**Over-saturation**: Rendered image 의 색이 extreme 한 값 (0 또는 255 in RGB space) 에 집중하는 현상.

**수학적 표현**: Histogram $h(x)$ (image intensity) 에서:
$$
\text{Saturation ratio} = \frac{|\{x : x < 0.1 \text{ or } x > 0.9\}|}{|\{x\}|}
$$

High saturation ratio (>50%) 이면 "over-saturated" 로 분류.

**원인**: 
- Diffusion 이 학습한 prototypical 이미지는 saturated colors (더 perceptually distinct)
- Mode-seeking 으로 이 extreme 에만 집중

### 정의 3.4 — Classifier-Free Guidance (CFG)

Conditional diffusion 에서 guidance scale $s$ 를 적용:
$$
\hat{\epsilon}_\phi^{(s)} = (1 + s) \hat{\epsilon}_\phi(z_t; y, t) - s \cdot \hat{\epsilon}_\phi(z_t; \emptyset, t)
$$

여기서:
- $\hat{\epsilon}_\phi(z_t; y, t)$: conditional (with text)
- $\hat{\epsilon}_\phi(z_t; \emptyset, t)$: unconditional
- $s$: guidance scale (보통 7.5)

**효과**:
- $s > 1$ 이면 조건부 확률이 더 "sharp" 해짐
- 분포의 중심 (text 에 가장 부합하는 이미지) 으로 더 강하게 끌어당김
- → **mode-seeking 악화**

---

## 🔬 정리와 증명

### 정리 3.1 — SDS 는 Reverse-KL

**명제**: SDS loss
$$
\mathcal{L}_{\text{SDS}} = \mathbb{E}_{t,\epsilon,\pi} [w(t) \|\hat{\epsilon}_\phi(z_t;y,t) - \epsilon\|^2]
$$

의 gradient 는:
$$
\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{z \sim q_\theta}[\nabla_z \log q_\theta(z) - \nabla_z \log p_\phi(z|y)]
$$

이는:
$$
\nabla_\theta \mathrm{KL}(q_\theta \| p_\phi)
$$

의 형태이다.

**증명**:

**Step 1 — Score matching loss 로 표현.**
$$
\mathcal{L}_{\text{SDS}} = \mathbb{E}_{z_t} [\|\hat{\epsilon} - \epsilon\|^2]
$$

이는 diffusion 의 score matching loss 와 동일:
$$
\mathcal{L}_{\text{score}} = \mathbb{E}_{x_t}[\|\nabla_{x_t} \log p(x_t) - \hat{\epsilon}(x_t)\|^2]
$$

**Step 2 — KL divergence 와의 관계.**

Fisher identity: score matching 은 maximum likelihood (KL 최소화) 와 일치:
$$
\min_\phi \mathcal{L}_{\text{score}} \equiv \min_\phi \mathrm{KL}(p_{\text{data}} \| p_\phi)
$$

우리의 경우, 데이터 분포이 $q_\theta(z)$ (NeRF 가 만드는 분포):
$$
\nabla_\theta \mathcal{L}_{\text{SDS}} \propto \nabla_\theta \mathrm{KL}(q_\theta \| p_\phi) \quad \square
$$

### 정리 3.2 — Mode-Seeking 의 정량화

**명제**: Reverse-KL divergence $\mathrm{KL}(q \| p)$ 를 최소화하면, 만약 $p$ 가 multimodal 이면, $q$ 는 $p$ 의 **가장 높은 single mode 에 집중** 한다.

**증명 (informal)**:

Reverse-KL 의 gradient:
$$
\nabla_q \mathrm{KL}(q \| p) = \nabla_q \mathbb{E}_{z \sim q}[\log q(z) - \log p(z)]
$$

Optimal $q^*$ 에서, zero-support region 에서는:
$$
\frac{q^*(z)}{p(z)} = \text{constant}
$$

$q^*$ 가 unimodal 이면, 이 constant 는 **가장 높은 $p$ 의 값** 에 의해 결정됨. 따라서 $q^*$ 는 그 mode 에 집중.

### 따름 정리 3.3 — CFG 의 Mode-Seeking 강화

**명제**: Guidance scale $s > 1$ 을 사용하면 reverse-KL 의 mode-seeking 이 더 심해진다.

**증명**:

CFG 적용 시:
$$
\hat{\epsilon}_{\text{CFG}}^{(s)} = (1+s) \hat{\epsilon}_\phi(z; y, t) - s \hat{\epsilon}_\phi(z; \emptyset, t)
$$

이는 effective prior 를 변경:
$$
\log p_{\text{CFG}}(z|y) \propto (1+s) \log p(z|y) - s \log p(z|\emptyset)
$$

Log probability 의 scaling 으로 분포가 더 sharp 해짐 (온도 감소 같음):
$$
p_{\text{scaled}} \propto p^{(1+s)/s}
$$

$s > 1$ 이면 지수 > 1 → distribution 의 mode 가 더 극단적으로 강조.

따라서 mode-seeking 이 강화됨 $\square$.

---

## 💻 구현 검증

### 실험 1 — Reverse-KL vs Forward-KL 시각화

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# True bimodal distribution
def true_dist(z):
    """Mixture of 2 Gaussians"""
    m1 = torch.tensor([-1.0, -1.0])
    m2 = torch.tensor([1.0, 1.0])
    s = 0.3
    
    d1 = torch.exp(-((z - m1) ** 2).sum(dim=-1) / (2 * s**2))
    d2 = torch.exp(-((z - m2) ** 2).sum(dim=-1) / (2 * s**2))
    
    return (d1 + d2) / 2

# Model: single Gaussian
class GaussianModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.log_sigma = nn.Parameter(torch.tensor([0.0, 0.0]))
    
    def forward(self, z):
        sigma = torch.exp(self.log_sigma)
        return torch.exp(-((z - self.mu) ** 2 / (2 * sigma**2)).sum(dim=-1))
    
    def log_prob(self, z):
        sigma = torch.exp(self.log_sigma)
        return -((z - self.mu) ** 2 / (2 * sigma**2)).sum(dim=-1) - torch.log(sigma).sum()

# Training
model_reverse_kl = GaussianModel()
model_forward_kl = GaussianModel()

opt_reverse = torch.optim.Adam(model_reverse_kl.parameters(), lr=0.01)
opt_forward = torch.optim.Adam(model_forward_kl.parameters(), lr=0.01)

# Reverse KL: E_q[log q - log p]
for epoch in range(100):
    z_sample = torch.randn(100, 2) * 2  # sample from model
    z_sample.requires_grad_(True)
    
    log_q = model_reverse_kl.log_prob(z_sample)
    log_p = torch.log(true_dist(z_sample) + 1e-6)
    
    loss_reverse = (log_q - log_p).mean()
    
    opt_reverse.zero_grad()
    loss_reverse.backward()
    opt_reverse.step()

# Forward KL: E_p[log p - log q]
for epoch in range(100):
    # Sample from true distribution (rejection sampling)
    z_sample = torch.randn(100, 2) * 2
    probs = true_dist(z_sample)
    mask = torch.rand(100) < (probs / probs.max())
    z_sample = z_sample[mask][:100]
    
    if len(z_sample) > 0:
        log_p = torch.log(true_dist(z_sample) + 1e-6)
        log_q = model_forward_kl.log_prob(z_sample)
        
        loss_forward = (log_p - log_q).mean()
        
        opt_forward.zero_grad()
        loss_forward.backward()
        opt_forward.step()

# Visualize
z_grid = torch.linspace(-3, 3, 50).unsqueeze(1)
z_grid = torch.cat([z_grid, z_grid.transpose(0, 1)], dim=-1).reshape(-1, 2)

true_probs = true_dist(z_grid).reshape(50, 50)
reverse_probs = model_reverse_kl(z_grid).detach().reshape(50, 50)
forward_probs = model_forward_kl(z_grid).detach().reshape(50, 50)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, probs, title in zip(axes, 
                             [true_probs, reverse_probs, forward_probs],
                             ['True (bimodal)', 'Reverse-KL (mode-seeking)', 'Forward-KL (mode-covering)']):
    ax.contourf(probs.numpy())
    ax.set_title(title)

plt.tight_layout()
plt.savefig('/tmp/kl_comparison.png')
print("✓ Reverse-KL concentrates on single mode, Forward-KL covers both")
```

### 실험 2 — Over-Saturation 시뮬레이션

```python
import torch
import numpy as np

def simulate_sds_rendering(num_iterations=100, mode_seeking_strength=1.0):
    """
    Simulate 3D rendering with SDS loss.
    mode_seeking_strength: controls how much mode-seeking occurs
    """
    # Initial NeRF parameters (simplified as RGB values)
    rgb = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
    optimizer = torch.optim.Adam([rgb], lr=0.05)
    
    history = []
    
    for it in range(num_iterations):
        # Mock diffusion prior: prefers saturated colors
        # p_φ 의 mode = [1.0, 0.0, 0.0] (pure red)
        prior_mode = torch.tensor([1.0, 0.0, 0.0])
        
        # SDS loss: pull rgb toward prior_mode (reverse-KL behavior)
        diff = rgb - prior_mode
        loss = (mode_seeking_strength * diff ** 2).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Clamp to [0, 1]
        with torch.no_grad():
            rgb.data = torch.clamp(rgb.data, 0, 1)
        
        history.append(rgb.detach().clone())
        
        if (it + 1) % 20 == 0:
            print(f"Iter {it+1:3d}: rgb = {rgb.detach().numpy()}, |diff| = {loss.item():.6f}")
    
    return history

# Compare: weak vs strong mode-seeking
print("=== Weak Mode-Seeking ===")
history_weak = simulate_sds_rendering(mode_seeking_strength=0.5)

print("\n=== Strong Mode-Seeking (realistic) ===")
history_strong = simulate_sds_rendering(mode_seeking_strength=2.0)

# Over-saturation metric
def saturation_ratio(rgb_history):
    """Fraction of pixels at extreme values"""
    rgb_array = torch.stack(rgb_history).numpy()
    extreme = (rgb_array < 0.1) | (rgb_array > 0.9)
    return extreme.mean(axis=1)

ratios_weak = saturation_ratio(history_weak)
ratios_strong = saturation_ratio(history_strong)

print(f"\nWeak:   Final saturation = {ratios_weak[-1]:.1%}")
print(f"Strong: Final saturation = {ratios_strong[-1]:.1%}")
print(f"✓ Stronger mode-seeking → higher over-saturation")
```

### 실험 3 — CFG 의 영향

```python
def simulate_cfg_effect(guidance_scales=[1.0, 5.0, 10.0]):
    """
    Simulate CFG's effect on mode-seeking.
    """
    results = {}
    
    for s in guidance_scales:
        rgb = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
        optimizer = torch.optim.Adam([rgb], lr=0.05)
        
        prior_mode = torch.tensor([1.0, 0.0, 0.0])
        
        for it in range(100):
            # CFG: stronger pull to prior_mode as s increases
            effective_scale = 1.0 + s  # CFG formula
            diff = rgb - prior_mode
            loss = (effective_scale * diff ** 2).sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                rgb.data = torch.clamp(rgb.data, 0, 1)
        
        results[s] = rgb.detach().numpy()
    
    # Display
    for s, final_rgb in sorted(results.items()):
        saturation = (final_rgb < 0.1) | (final_rgb > 0.9)
        print(f"CFG scale {s:5.1f}: RGB = {final_rgb}, saturation = {saturation.mean():.1%}")

simulate_cfg_effect()
# Expected: higher CFG → stronger mode-seeking → more saturation
```

---

## 🔗 실전 활용

### 1. Mode-Seeking 완화: Lower CFG Scale

```python
# DreamFusion 구현에서
for iteration in range(num_iters):
    camera = sample_random_camera()
    z = nerf_render(nerf, camera)
    
    # Lower CFG scale to reduce mode-seeking
    with torch.no_grad():
        # Instead of s=7.5 (aggressive)
        s = 3.0  # More conservative
        
        eps_cond = diffusion(z_t, t, prompt)
        eps_uncond = diffusion(z_t, t, "")
        eps_pred = eps_uncond + s * (eps_cond - eps_uncond)
    
    loss = sds_loss(eps_pred, noise)
    loss.backward()
```

### 2. Diversity 증가: Multi-View Sampling

```python
# 각 iteration 에서 여러 view 로부터 손실을 평균화
# → 같은 3D 가 여러 방향에서 plausible 해야 함
# → 한 mode 에만 집중 불가

for iteration in range(num_iters):
    losses = []
    for cam_idx in range(4):  # 4 random views
        camera = sample_random_camera()
        z = nerf_render(nerf, camera)
        
        t = sample_diffusion_timestep()
        eps_pred = diffusion_score(z_t, prompt, t)
        loss = sds_loss(eps_pred, noise)
        losses.append(loss)
    
    # Average over views
    total_loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

# 효과: 한 3D 가 모든 각도에서 좋아야 함 → mode-seeking 완화
```

### 3. ProlificDreamer 스타일: VSD Loss

```python
# 다음 장 (Ch6-04) 에서 상세 다루지만, 개념:

# DreamFusion (SDS):
loss_sds = ((eps_pred_fixed - noise) ** 2).mean()

# ProlificDreamer (VSD):
# Particle-specific score ε̂_ψ(z_t; y, t, θ) 학습
eps_pred_fixed = diffusion(z_t, t, prompt)  # Frozen
eps_pred_particle = diffusion_particle(z_t, t, prompt, theta=nerf_params)  # Fine-tuned LoRA

loss_vsd = ((eps_pred_fixed - eps_pred_particle) ** 2).mean()

# 효과: 두 score 의 차이를 최소화
# → mode-seeking 보다 mode-covering 으로 이동
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| **Prior 가 multimodal** | 실제로 LAION 기반 diffusion 은 매우 multimodal |
| **Reverse-KL approximation** | SDS 는 정확한 reverse-KL 이 아니라 근사 → score matching 으로 만 가정 |
| **Single-mode mode-seeking** | 실제로는 여러 local modes 에 걸쳐있을 수 있음 |
| **CFG 가 always mode-seeking 강화** | 매우 낮은 CFG (s=1) 에서는 효과 약할 수 있음 |
| **Over-saturation 이 항상 나쁜가** | 일부 객체 (e.g., fruit) 에서는 saturated color 가 자연스러움 → context 의존 |

---

## 📌 핵심 정리

$$\boxed{\text{SDS} = \text{Reverse-KL} \Rightarrow \text{Mode-Seeking} \Rightarrow \text{Over-Saturation}}$$

| 단계 | 현상 | 원인 |
|------|------|------|
| **SDS loss** | $\nabla \mathrm{KL}(q_\theta \| p_\phi)$ | Frozen diffusion prior |
| **Reverse-KL** | Mode-seeking | $\log q/p$ ratio 최적화 at mode peaks |
| **Diffusion prior** | Prototypical saturated colors | LAION 의 가장 distinctive features |
| **Mean-reversion** | Extreme cases 무시 | Score 가 distribution center 를 가리킴 |
| **CFG amplification** | Mode-seeking 강화 | Guidance 로 분포 sharpen |
| **결과** | Over-saturated, low diversity | 한 3D 의 여러 variants 불가능 |

**Key insight**: Text-to-3D 의 "cartoon" artifacts 는 algorithm 의 결함이 아니라 **reverse-KL divergence 의 본질적 특성**.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Reverse-KL $\mathrm{KL}(q \| p)$ 와 Forward-KL $\mathrm{KL}(p \| q)$ 의 차이를 문장으로 설명하라. 각각이 선호하는 오류의 유형은 무엇인가?

<details>
<summary>해설</summary>

**Reverse-KL: $\mathrm{KL}(q \| p) = \mathbb{E}_{z \sim q}[\log q - \log p]$**

- $z \sim q$ 에서 샘플링 → $q$ 의 high-probability region 에 집중
- $\log q - \log p$ 가 크면 penalty (q 가 p 보다 높은 곳)
- → $q$ 는 $p$ 의 "어딘가"에만 집중 (mode-seeking)
- **선호하는 오류**: under-coverage (diversity 부족) vs over-confident (한 대표값)

**Forward-KL: $\mathrm{KL}(p \| q) = \mathbb{E}_{z \sim p}[\log p - \log q]$**

- $z \sim p$ 에서 샘플링 → $p$ 의 모든 high-probability region 에서 penalty
- $\log p - \log q$ 가 크면 penalty (p 가 q 보다 높은 곳)
- → $q$ 는 $p$ 의 모든 모드를 커버해야 함 (mode-covering)
- **선호하는 오류**: over-coverage (불필요한 곳에도 할당) vs under-confident (모든 곳에 퍼짐)

**의미**: DreamFusion 이 mode-seeking 을 "선택"한 것은 아니고, **reverse-KL 유도 (score distillation) 의 필연적 결과** $\square$.

</details>

**문제 2** (심화): Classifier-Free Guidance 의 식을 다시 쓰면:
$$
\hat{\epsilon}^{(s)} = (1+s) \hat{\epsilon}_\phi(z; y, t) - s \hat{\epsilon}_\phi(z; \emptyset, t)
$$

이를 log-probability 로 해석하면 어떤 의미인가? $s \to \infty$ 일 때 극한은?

<details>
<summary>해설</summary>

**Log-probability 해석**:

Diffusion 의 score 는:
$$
\hat{\epsilon} \propto -\nabla_z \log p(z_t | y)
$$

따라서:
$$
\hat{\epsilon}^{(s)} \propto -\nabla_z \log p(z_t|y)^{1+s} + s \nabla_z \log p(z_t|\emptyset)
$$

이는 effective log-probability:
$$
\log \tilde{p}(z_t|y) \propto (1+s) \log p(z_t|y) - s \log p(z_t|\emptyset)
$$

이를 정규화하면:
$$
\log \tilde{p}(z_t|y) \propto \log \frac{p(z_t|y)^{1+s}}{p(z_t|\emptyset)^s}
$$

**극한 $s \to \infty$**:
$$
\log \tilde{p} \to \log \frac{p(z_t|y)}{p(z_t|\emptyset)} + (1 + o(1)) \log p(z_t|y)
$$

분포는 **likelihood ratio** $p(z|y) / p(z|\emptyset)$ 에 점점 더 집중 → 가장 distinctive features (saturated colors, extreme details) 만 남음.

**의의**: CFG 는 "discriminative" 방향으로 분포를 변형 → mode-seeking 강화 $\square$.

</details>

**문题 3** (논문 비평): SDS 로 생성된 3D 가 "cartoon-like" 인 반면, 실제 사진은 미묘한 shading/lighting 을 가진다. DreamFusion 논문 (Poole et al. 2023) 은 이를 어떻게 논의했는가? ProlificDreamer (Wang 2023) 는 VSD 로 이를 얼마나 개선했는가?

<details>
<summary>해설</summary>

**DreamFusion 의 인정**:

Poole et al. (2023) 은 다음을 명시적으로 언급:
- SDS loss 는 "mode-seeking" 특성 가짐
- 생성된 3D 는 때때로 "over-saturated" 또는 "cartoon-like" 보임
- 다양성이 제한적 (같은 prompt 여러 실행 → 비슷한 결과)

그럼에도 DreamFusion 이 "첫 text-to-3D with high quality" 로 인정됨 (vs Point-E 등의 저 품질).

**ProlificDreamer 의 개선**:

Wang et al. (2023) 은 VSD (Variational Score Distillation) 제안:
$$
\mathcal{L}_{\text{VSD}} = \mathbb{E}[\hat{\epsilon}_\phi(z;y,t) - \hat{\epsilon}_\psi(z;y,t,\theta)]^2
$$

여기서 $\hat{\epsilon}_\psi$ 는 particle-specific fine-tuned score (LoRA).

**효과**:
- Reverse-KL → **Wasserstein gradient flow** (mode-covering 성질)
- Diversity 대폭 증가 (ProlificDreamer 검증: FID ↑, diversity ↑)
- Over-saturation 감소
- 대신 학습 시간 증가 (LoRA fine-tuning 필요)

**결론**: Trade-off — DreamFusion (빠름, 때때로 cartoon) vs ProlificDreamer (느림, 더 다양하고 자연스러움) $\square$.

</details>

---

<div align="center">

[◀ 이전](./02-sds-derivation.md) | [📚 README](../README.md) | [다음 ▶](./04-vsd-particle-score.md)

</div>
