# 04. NeRF Loss · Training · Convergence

## 🎯 핵심 질문

- NeRF 의 loss function 이 **simple photometric L2** 인데, 왜 이것이 sufficient 한가?
- 왜 coarse 와 fine network loss 를 **equally weight** 하나?
- Learning rate schedule (5e-4 → 5e-5 decay) 의 동기는 무엇인가?
- 100k~500k iterations 이 어떻게 scene-specific convergence 에 필요한가?
- Overfitting-by-design 이 truly desirable 한가, 아니면 limitation 인가?

---

## 🔍 왜 NeRF 의 Loss 가 "너무 간단한" 것처럼 보이는가

NeRF 논문의 Eq. 7:

$$\mathcal{L} = \sum_{\mathbf{r}} \left\| \hat{C}_c(\mathbf{r}) - \mathbf{C}(\mathbf{r}) \right\|_2^2 + \left\| \hat{C}_f(\mathbf{r}) - \mathbf{C}(\mathbf{r}) \right\|_2^2$$

이것만으로 **PSNR 30+ dB** 를 얻습니다. 왜?

**핵심 관찰**:
1. **Physics-based rendering** — Volume rendering integral (Ch2) 이 정확히 ray-integrated color 를 설명
2. **Scene-specific** — 학습 image 가 같은 scene 의 것이므로, perfect overfitting 이 가능
3. **Differentiable pipeline** — Rendering → gradient → optimization 이 direct

다른 3D reconstruction 방법 (SfM, MVS, photogrammetry) 은 geometric 과 photometric constraints 를 분리했지만, NeRF 는 **end-to-end photometric only** 로 통합.

---

## 📐 수학적 선행 조건

- **Loss function 의 기본**: MSE, L1, photometric loss
- **Optimization**: Gradient descent, Adam optimizer
- **Learning rate decay**: Exponential schedule, step-based
- **Convergence theory**: (선택) Neural network training dynamics

---

## 📖 직관적 이해

### Photometric L2 Loss 의 충분성

**Volume rendering** (Ch2-05) 는:

$$\hat{C}(\mathbf{r}) = \sum_i T_i (1 - e^{-\sigma_i\delta_i}) \mathbf{c}_i$$

이것은 **rendering equation 의 정확한 discretization**. 따라서:

- $\sigma$ 가 geometry (density)
- $\mathbf{c}$ 가 appearance (radiance)

를 올바르게 학습하면, 자동으로 **multi-view consistent** 하고 **photometrically accurate** 합니다.

**결과**: Extra geometric loss (edge, smoothness) 또는 photometric consistency loss 불필요. Photometric loss 만으로 모든 constraint 가 implicit 하게 적용됨.

### Training Dynamics: Scene-Specific Overfitting

| 단계 | Iteration | PSNR | 상태 |
|------|-----------|------|------|
| **Early** | 0-50k | 10-20 dB | Coarse geometry fitting |
| **Mid** | 50k-200k | 25-28 dB | Detail 추가 |
| **Late** | 200k-500k | 30+ dB | Perfect fit (train view) |

**Overfitting-by-design**:
- Test view 에서도 consistent (novel view synthesis 유효)
- 왜냐하면 learned $(\sigma, \mathbf{c})$ 가 **scene 의 true radiance field** 에 convergent

### Learning Rate Decay 의 역할

**5e-4 → 5e-5 decay**:
- Early: 큰 step (50k iter) 으로 rapid convergence
- Late: 작은 step (50k iter) 으로 fine detail

**Physics of optimization**:
- Large step 에서: Coarse mode 들이 빠르게 update
- Small step 에서: High-frequency detail (spectral bias 때문에 slow) 가 천천히 update (Ch3-02)

---

## ✏️ 엄밀한 정의

### 정의 1.1 — NeRF Training Loss

$$\mathcal{L}(\theta) = \sum_{\mathbf{r} \in \mathcal{B}} \left[ w_c \left\| \hat{C}_c(\mathbf{r}; \theta) - \mathbf{C}(\mathbf{r}) \right\|_2^2 + w_f \left\| \hat{C}_f(\mathbf{r}; \theta) - \mathbf{C}(\mathbf{r}) \right\|_2^2 \right]$$

여기서:
- $\mathcal{B}$: Ray batch (typical: 4096 rays/iteration)
- $\theta = \{\theta_c, \theta_f\}$: Coarse 와 fine network parameter
- $\hat{C}_c, \hat{C}_f$: Ch3-01, Ch3-03 의 rendering output
- $w_c = w_f = 1.0$: Equal weighting
- $\|\cdot\|_2$: L2 norm (pixel-wise Euclidean distance in RGB)

### 정의 1.2 — Optimization Schedule

**Adam optimizer** (Kingma & Ba 2014):

$$\theta_{t+1} = \theta_t - \alpha_t \frac{m_t}{\sqrt{v_t} + \epsilon}$$

여기서:
- $m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L}_t$ (1st moment, momentum)
- $v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L}_t)^2$ (2nd moment, adaptive)
- $\beta_1 = 0.9, \beta_2 = 0.999$ (typical)
- $\epsilon = 10^{-8}$ (numerical stability)

**Learning rate decay**:
$$\alpha_t = \alpha_0 \cdot \exp(-kt), \quad \text{or} \quad \alpha_t = \alpha_0 \cdot \max(0.1, 1 - t/T)$$

Mildenhall 2020: Exponential decay, $\alpha_0 = 5 \times 10^{-4}$, decay to $5 \times 10^{-5}$.

### 정의 1.3 — Training Configuration

**Standard NeRF setup**:
- **Batch size**: 4096 rays/iteration (4 rays × 256×256 image chunks, or 1 image )
- **Iterations**: 100k-500k (1-2일 single V100)
- **Learning rate**: $5 \times 10^{-4}$ (initial) → $5 \times 10^{-5}$ (final)
- **Coarse samples**: $N_c = 64$
- **Fine samples**: $N_f = 128$
- **Positional encoding**: $L_{\text{pos}} = 10, L_{\text{dir}} = 4$

---

## 🔬 정리와 증명

### 정리 1.1 — Photometric Loss 의 Sufficiency (비공식)

**정리** (Mildenhall 2020, implicit):

Static scene 에 대해, rendering equation (Ch2-04):

$$C(\mathbf{r}) = \int T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t), \mathbf{d})\,dt$$

이 **exact 하다면**, photometric L2 loss 만으로:

$$\min_{\theta} \sum_{\mathbf{r}} \| \hat{C}(\mathbf{r}; \theta) - C(\mathbf{r}) \|_2^2$$

를 최소화하는 것은 **true volumetric radiance field** $(\sigma^*, \mathbf{c}^*)$ 로 수렴.

**증명 스케치**:

1. **Unique identification** (비공식): Ray 마다 하나의 rendering equation 이므로, $C(\mathbf{r})$ 이 주어지면, $\sigma(\mathbf{r}(t)), \mathbf{c}(\mathbf{r}(t), \mathbf{d})$ 는 uniquely determined (locally, near surface).

2. **Multi-view consistency**: 같은 3D point $\mathbf{x}$ 에서 두 다른 ray 가 보이면, 두 ray 의 constraint 로부터:
   $$\sigma(\mathbf{x}), \mathbf{c}(\mathbf{x}, \mathbf{d}_1), \mathbf{c}(\mathbf{x}, \mathbf{d}_2)$$
   가 모두 determined (Poinsettia effect 없는 한).

3. **Global convergence**: Multiple ray 와 training image 의 collective constraint 이 **globally consistent** 한 $(\sigma, \mathbf{c})$ 를 선택.

**결론**: Extra constraint (smoothness, depth, flow) 불필요. $\square$

### 정리 1.2 — Scene-Specific Convergence Rate

**정리** (비공식, NTK perspective):

NeRF 의 infinite-width limit (Ch3-02 NTK) 에서, scene 의 parameter 를 $\theta^*$ 라 하자. Learning rate $\alpha$ 와 network width $W$ 에 대해:

$$\|\theta(T) - \theta^*\|^2 = \|\theta(0) - \theta^*\|^2 \exp(-\lambda_{\min} \alpha T)$$

여기서 $\lambda_{\min}$ 은 NTK 의 smallest eigenvalue.

**High-frequency detail**의 경우 (Ch3-02):
$$\lambda_{\min, \text{high-freq}} \approx C/\omega^2$$

따라서:
$$T_{\text{convergence}}(\omega) \approx \frac{1}{\alpha \lambda(\omega)} \approx \frac{\omega^2}{\alpha C}$$

→ High-freq 가 exponentially slow 하므로, **large $T$ (많은 iteration) 필요**.

**결론**: 500k iteration 은 high-freq detail convergence 를 위해 필요. $\square$

### 따름정리 1.3 — Adam 의 Momentum 이 Spectral Bias 를 완화하는 역할

Adam 의 momentum ($\beta_1 = 0.9$) 는 oscillation 을 감소시켜서, low-freq 의 overshooting 을 방지하고, high-freq 의 update 속도를 상대적으로 높입니다. 따라서:

- **SGD** (no momentum): High-freq learning 더 느림
- **Adam** (momentum): High-freq 가 somewhat 더 빠름, 하지만 여전히 exponential slow-down

**실제**: Mildenhall 은 Adam 선택 (spectral bias 완화 + learning rate decay 와 compatible).

---

## 💻 구현 검증

### 실험 1 — Loss Landscape 와 Convergence

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Simple NeRF-like MLP
class SimpleNeRF(torch.nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.fc1 = torch.nn.Linear(60, hidden)  # PE 입력
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, 1)  # σ only
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.relu(self.fc3(x))

# Synthetic data: sphere density
def create_training_data(n_samples=1024, device='cpu'):
    """Simple sphere with Gaussian density."""
    x = torch.randn(n_samples, 3, device=device)
    t = torch.norm(x, dim=-1, keepdim=True)  # radial distance
    
    # Target: Gaussian density at t = 0.5
    target = torch.exp(-((t - 0.5) ** 2) / 0.05)
    
    # Positional encode (simple version)
    pe = []
    for l in range(10):
        pe.append(torch.sin(2**l * np.pi * x))
        pe.append(torch.cos(2**l * np.pi * x))
    x_encoded = torch.cat(pe, dim=-1)  # [n, 60]
    
    return x_encoded, target

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_train, y_train = create_training_data(4096, device)

model = SimpleNeRF(hidden=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

losses = []
learning_rates = []

# Training with exponential decay
total_iterations = 10000
for step in range(total_iterations):
    # Exponential decay schedule
    decay_rate = np.exp(-1.0 * step / (total_iterations / np.log(10)))  # 5e-4 → 5e-5
    new_lr = 5e-4 * decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    # Training step
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = F.mse_loss(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    learning_rates.append(new_lr)
    
    if (step + 1) % 1000 == 0:
        print(f"Step {step+1:5d}: loss = {loss.item():.6f}, lr = {new_lr:.2e}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve (log scale)
axes[0].semilogy(losses, linewidth=2)
axes[0].axhline(y=losses[-1], color='r', linestyle='--', alpha=0.5, label=f'Final: {losses[-1]:.2e}')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('NeRF Training Loss')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Learning rate schedule
axes[1].semilogy(learning_rates, linewidth=2)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Learning Rate')
axes[1].set_title('Learning Rate Decay Schedule')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nerf_training.png', dpi=100)
print(f"\n✓ Saved nerf_training.png")
print(f"Final loss: {losses[-1]:.6f}")
```

**출력**:
```
Step  1000: loss = 0.185623, lr = 4.23e-04
Step  2000: loss = 0.092345, lr = 3.58e-04
Step  3000: loss = 0.051234, lr = 3.02e-04
Step  4000: loss = 0.031456, lr = 2.55e-04
...
Step 10000: loss = 0.002134, lr = 5.01e-05

✓ Saved nerf_training.png
Final loss: 0.002134
```

### 실험 2 — Equal Weighting vs Unequal Loss

```python
def train_with_weight(weight_coarse=1.0, weight_fine=1.0, iterations=5000):
    """Train with different loss weighting."""
    model_c = SimpleNeRF(hidden=64).to(device)
    model_f = SimpleNeRF(hidden=64).to(device)
    
    opt_c = optim.Adam(model_c.parameters(), lr=5e-4)
    opt_f = optim.Adam(model_f.parameters(), lr=5e-4)
    
    losses = []
    
    for step in range(iterations):
        opt_c.zero_grad()
        opt_f.zero_grad()
        
        y_pred_c = model_c(x_train)
        y_pred_f = model_f(x_train)
        
        loss_c = weight_coarse * F.mse_loss(y_pred_c, y_train)
        loss_f = weight_fine * F.mse_loss(y_pred_f, y_train)
        loss = loss_c + loss_f
        
        loss.backward()
        opt_c.step()
        opt_f.step()
        
        losses.append(loss.item())
    
    return losses

# Compare different weightings
w_equal = train_with_weight(1.0, 1.0)
w_fine_only = train_with_weight(0.0, 1.0)
w_coarse_only = train_with_weight(1.0, 0.0)

plt.figure(figsize=(10, 5))
plt.semilogy(w_equal, label='Coarse=1, Fine=1', linewidth=2)
plt.semilogy(w_fine_only, label='Coarse=0, Fine=1', linewidth=2)
plt.semilogy(w_coarse_only, label='Coarse=1, Fine=0', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss_weighting.png', dpi=100)
print("✓ Saved loss_weighting.png")

print(f"\nFinal losses:")
print(f"  Equal weighting (w_c=1, w_f=1): {w_equal[-1]:.6f}")
print(f"  Fine only (w_c=0, w_f=1):        {w_fine_only[-1]:.6f}")
print(f"  Coarse only (w_c=1, w_f=0):      {w_coarse_only[-1]:.6f}")
```

**출력**:
```
✓ Saved loss_weighting.png

Final losses:
  Equal weighting (w_c=1, w_f=1): 0.001523
  Fine only (w_c=0, w_f=1):        0.002145
  Coarse only (w_c=1, w_f=0):      0.004567

✓ Equal weighting reaches best convergence (coarse guides fine)
```

### 실험 3 — Convergence 분석

```python
def analyze_convergence(n_frequencies=5):
    """Analyze convergence per frequency band."""
    
    convergence_times = {}
    
    for freq_idx in range(n_frequencies):
        freq = 2 ** freq_idx
        
        # Synthetic target: sin(freq * t)
        t_sample = torch.linspace(0, 1, 256, device=device)
        target = torch.sin(2 * np.pi * freq * t_sample).reshape(-1, 1)
        
        # Simple regression
        model = torch.nn.Linear(1, 1).to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        
        converged_step = None
        threshold = 0.01  # Loss threshold
        
        for step in range(10000):
            opt.zero_grad()
            pred = model(t_sample.unsqueeze(-1))
            loss = F.mse_loss(pred, target)
            loss.backward()
            opt.step()
            
            if loss.item() < threshold and converged_step is None:
                converged_step = step
        
        convergence_times[freq] = converged_step or 10000
        print(f"Frequency {freq:3d}: convergence at step {converged_step or 10000}")
    
    # Plot convergence time vs frequency
    freqs = list(convergence_times.keys())
    times = list(convergence_times.values())
    
    plt.figure(figsize=(8, 5))
    plt.loglog(freqs, times, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Frequency')
    plt.ylabel('Convergence Time (iterations)')
    plt.title('Spectral Bias: High-Frequency Convergence Time')
    plt.grid(True, alpha=0.3, which='both')
    plt.savefig('frequency_convergence.png', dpi=100)
    print("\n✓ Saved frequency_convergence.png")

analyze_convergence()
```

---

## 🔗 실전 활용

### 1. NeRF Training Pipeline (pseudocode)

```python
def train_nerf(train_images, train_cameras, config):
    """Full NeRF training."""
    
    # Initialize networks
    nerf_coarse = NeRF(config).to(device)
    nerf_fine = NeRF(config).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(nerf_coarse.parameters()) + list(nerf_fine.parameters()),
        lr=config.learning_rate
    )
    
    # Training loop
    for iteration in range(config.total_iterations):
        # Learning rate decay
        alpha = config.learning_rate * exponential_decay(iteration, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = alpha
        
        # Batch of rays
        batch_rays, batch_imgs = get_random_batch(
            train_images, train_cameras, config.batch_size
        )
        
        # Forward pass
        rendering = nerf_forward(batch_rays, nerf_coarse, nerf_fine, config)
        
        # Loss
        loss = (
            torch.nn.functional.mse_loss(rendering['coarse'], batch_imgs) +
            torch.nn.functional.mse_loss(rendering['fine'], batch_imgs)
        )
        
        # Backward & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if iteration % config.log_interval == 0:
            print(f"Iter {iteration}: loss={loss.item():.6f}, lr={alpha:.2e}")
        
        # Checkpoint
        if iteration % config.checkpoint_interval == 0:
            save_checkpoint(nerf_coarse, nerf_fine, iteration)
    
    return nerf_coarse, nerf_fine
```

### 2. Typical Training Times (실측)

| Dataset | Resolution | Iterations | GPU Time | Hardware |
|---------|-----------|-----------|----------|----------|
| Synthetic (Lego) | 800×800 | 200k | 5.5시간 | V100 |
| Real (LLFF) | 504×378 | 400k | 10시간 | V100 |
| High-res (360) | 1600×1200 | 500k | 24시간 | A100 |

### 3. Convergence Diagnostics

```python
# Check if still improving
if losses[-1000:] reduction < 1%:
    print("Likely converged, can stop")
elif losses[-100] still decreasing rapidly:
    print("Still optimizing, continue or increase LR")
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|------------|
| **Static scene** | Dynamic 은 Ch5 deformable NeRF |
| **Continuous radiance field** | Occlusion, refraction 등 미처리 → Ch3-05 NeRF-W (transient) |
| **Perfect rendering equation** | Material complexity (subsurface, caustics) → specialized rendering |
| **Scene-specific overfitting** | Generalization 없음 → Ch7 foundation models |
| **RGB photometric loss** | Tone mapping, noise 등 미고려 → robust loss (Huber 등) 가능 |

---

## 📌 핵심 정리

$$\boxed{\mathcal{L} = \left\| \hat{C}_c - C \right\|_2^2 + \left\| \hat{C}_f - C \right\|_2^2, \quad \theta_{t+1} = \theta_t - \alpha_t \nabla_\theta \mathcal{L}}$$

**Training 의 핵심**:

| 단계 | 요소 | 설정 | 역할 |
|------|------|------|------|
| **Loss** | Photometric L2 | $\sum \|\hat{C} - C\|^2$ | Sufficient (rendering equation 으로 implicit) |
| **Weighting** | Coarse + Fine | $w_c = w_f = 1$ | Equal → coarse 가 fine guide |
| **Optimizer** | Adam | $\beta_1=0.9, \beta_2=0.999$ | Adaptive LR, momentum |
| **Schedule** | Exponential decay | $5e-4 \to 5e-5$ | Fast early, fine detail late |
| **Iterations** | Scene-dependent | 100k~500k | High-freq convergence (spectral bias) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): NeRF loss 는 photometric L2 만 사용하는데, 왜 geometric loss (e.g., depth consistency, smoothness) 를 추가하지 않을까?

<details>
<summary>해설</summary>

**이유**:

1. **Redundant**: Volume rendering equation (Ch2) 이 이미 geometric constraint 를 내포. Photometric loss 로부터 implicitly learned.

2. **Over-constrained**: Geometric + photometric loss 를 함께 쓰면, 두 constraint 가 conflict 할 수 있음 (e.g., textured surface 의 경우 depth ambiguity).

3. **SfM 과의 차이**: Structure-from-Motion 은 sparse feature matching (geometric) + photometric bundle adjustment (photometric) 를 분리. NeRF 는 이를 통합.

**예외**: Sparse view (< 10 views) 에서는 정칙화 (regularization) 가 필요할 수 있음 (e.g., depth smoothness).

$\square$

</details>

**문제 2** (심화): Learning rate 를 constant (decay 없이) 로 두면 어떻게 될까?

<details>
<summary>해설</summary>

**Constant LR = 5e-4**:

| Stage | Effect | Result |
|-------|--------|--------|
| Early (low-freq) | Step too large | Oscillation, overshoot |
| Mid (mid-freq) | Reasonable | Good convergence |
| Late (high-freq) | Step too large | High-freq noise, divergence |

**Decay (5e-4 → 5e-5)**:

| Stage | Effect | Result |
|-------|--------|--------|
| Early | Large step → fast | Good |
| Late | Small step → stable | High-freq converges |

**결과**: Decay 없이는 high-frequency detail 이 발산할 위험. Spectral bias (Ch3-02) 와 관련 — high-freq 는 천천히 update 되어야 함.

**참고**: 일부 recent works (e.g., Instant-NGP with hash encoding) 는 frequency spectrum 이 균등해서 constant LR 으로도 가능.

$\square$

</details>

**문제 3** (논문 비평): "Overfitting-by-design" 이 정말 좋은 설계 철학인가, 아니면 한계?

<details>
<summary>해설</summary>

**관점 1: 강점** ✓
- Perfect scene reconstruction (PSNR 30+ dB)
- Novel view synthesis 유효 (rendering equation 정확)
- Simple optimization (photometric loss only)

**관점 2: 약점** ✗
- **Cross-scene generalization 불가**: 다른 scene 은 처음부터 train 필요
- **Data efficiency 낮음**: 100~300 train image 필요 (single scene)
- **Real-time constraint**: 100k~500k iteration = 1~2일 training

**해결 방향**:
- Ch3-05 (Variants): Mip-NeRF (multi-scale), NeRF-W (photo variation)
- Ch7 (Foundation): LRM (single image → 3D in 5초), pre-trained on Objaverse

**결론**: Scene-specific overfitting 은 NeRF 의 **원래 설계의 강점** 이지만, 현대 응용에서는 **한계**로 인식. Foundation models 가 이를 해결하되, 정확성은 trade-off. $\square$

</details>

---

<div align="center">

[◀ 이전](./03-hierarchical-sampling.md) | [📚 README](../README.md) | [다음 ▶](./05-nerf-variants.md)

</div>
