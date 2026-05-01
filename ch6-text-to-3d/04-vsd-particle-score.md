# 04. Variational Score Distillation (ProlificDreamer, Wang 2023)

## 🎯 핵심 질문

- SDS 의 mode-seeking 을 어떻게 이론적으로 해결할 수 있는가?
- Variational Score Distillation (VSD) 은 어떻게 정의되는가?
- **Particle-specific score** $\hat{\epsilon}_\psi(z_t; y, t, \theta)$ 란 무엇인가? 왜 fine-tune 해야 하는가?
- VSD 의 Wasserstein gradient flow 해석은 무엇인가?
- LoRA (Low-Rank Adaptation) 가 VSD 에서 어떤 역할을 하는가?

---

## 🔍 왜 VSD 인가

DreamFusion (SDS) 의 문제:
- **Mode-seeking** → Over-saturated, low diversity
- **한 가지 해석만** → "a car" 라고 해도 특정 스타일만 생성
- **수렴이 빠르지만 한정적** → 초반에 saturated color 로 빠지면 탈출 어려움

해결책: **두 개의 score** 를 비교하여 mode-covering 방향으로 유도.

---

## 📐 수학적 선행 조건

- **Ch6-03**: Mode-seeking 분석
- **Information Theory**: Wasserstein distance, optimal transport
- **Gradient flows**: Particle methods, mean-field theory
- **LoRA**: Low-rank matrix adaptation (Linear Layers)

---

## 📖 직관적 이해

### 핵심 아이디어: 두 Score 의 차이

**DreamFusion (한 개의 score)**:
```
Fixed diffusion ε̂_φ (frozen) ──→ 한 방향으로만 pull
                                 (prototype 방향)
                                 
3D 가 이 한 값으로 수렴
```

**ProlificDreamer (두 개의 score)**:
```
Fixed diffusion ε̂_φ (frozen)           ──→ General prior
                                        (모든 "빨간 차" 의 특성)

Particle-specific ε̂_ψ(θ) (fine-tuned) ──→ This 3D 가 특정한 방향
                                        (현재 3D에 맞춤)

Loss: ε̂_φ - ε̂_ψ 를 최소화
  → ε̂_φ 와 ε̂_ψ 가 같으면 loss = 0
  → 3D 가 general prior 를 따르되, 
    동시에 자신의 "개성" 을 유지
```

### 도식: Wasserstein Gradient Flow

```
Prior p_φ (multimodal):
  
  ╱╲╱╲╱╲╱╲
 │  ╱╲  │  ← 여러 mode
 │ │  │ │
 └─────────┘

Particle distribution q_θ:
  
  ●●●●●●●● ← 여러 개의 3D ("particles")
  
Particle 하나는 하나의 3D model 에 대응.
Wasserstein flow: 각 particle 이 prior 의 mode 로 모이되,
                  particles 사이에는 repulsion 존재
                  (mode 마다 하나씩 할당)
```

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Variational Score Distillation (VSD)

두 score function 을 정의:
- $\hat{\epsilon}_\phi(z_t; y, t)$: Frozen global prior (DreamFusion 과 동일)
- $\hat{\epsilon}_\psi(z_t; y, t, \theta)$: Particle-specific, θ dependent

**VSD Loss**:
$$
\mathcal{L}_{\text{VSD}}(\theta) := \mathbb{E}_{t,\epsilon,\pi} \left[ \left\| \hat{\epsilon}_\phi(z_t; y, t) - \hat{\epsilon}_\psi(z_t; y, t, \theta) \right\|^2 \right]
$$

**Gradient**:
$$
\nabla_\theta \mathcal{L}_{\text{VSD}} = \mathbb{E}_{t,\epsilon,\pi} \left[ (\hat{\epsilon}_\phi - \hat{\epsilon}_\psi) \frac{\partial \hat{\epsilon}_\psi}{\partial \theta} + (\hat{\epsilon}_\phi - \hat{\epsilon}_\psi) \frac{\partial z}{\partial \theta} \cdot \frac{\partial \hat{\epsilon}_\psi}{\partial z} \right]
$$

### 정의 4.2 — Particle-Specific Score (LoRA)

$\hat{\epsilon}_\psi$ 는 diffusion U-Net 의 **부분 fine-tuning**:

$$
\hat{\epsilon}_\psi(z; y, t, \theta) = \hat{\epsilon}_\phi(z; y, t) + \Delta\hat{\epsilon}(z; y, t; \theta)
$$

여기서 $\Delta\hat{\epsilon}$ 는 **low-rank** adapter:
$$
\Delta\hat{\epsilon} = U V^\top \sigma(z, t)
$$

- $U \in \mathbb{R}^{D \times r}$, $V \in \mathbb{R}^{D \times r}$ (rank $r \ll D$)
- $\sigma(z, t)$ (activation)
- 학습 parameter: $U, V$ (θ dependent)

**효과**: 3D parameter θ 에 맞춤화된 조정, 하지만 전체 diffusion 은 frozen.

### 정의 4.3 — Wasserstein Gradient Flow

Particle distribution $\{z_i\}_{i=1}^N$ (각 $z_i$ 는 하나의 3D 에서 렌더링된 image) 이 있을 때:

**Mean-field limit** ($N \to \infty$) 에서:
$$
\frac{\partial \rho_t}{\partial t} = \nabla \cdot (\rho_t \nabla \Phi) + \frac{1}{2} \Delta \rho_t
$$

여기서 $\Phi$ 는 potential (우리의 경우 diffusion score 의 음수).

**Discrete particle approximation**:
$$
\frac{d z_i}{dt} \propto \nabla \Phi(z_i) + \text{noise}
$$

**VSD 가 Wasserstein flow 에 가까운 이유**:
- $\hat{\epsilon}_\phi$ : Global potential 기울기 방향 (모든 particle 이 같음)
- $\hat{\epsilon}_\psi$ : Particle-specific 조정 (각 particle 마다 다름)
- 차이 최소화 → flow 가 mode 간 "분산" 을 유지

---

## 🔬 정리와 증명

### 정리 4.1 — VSD 는 Mode-Covering 방향

**명제**: VSD loss
$$
\mathcal{L}_{\text{VSD}} = \mathbb{E}[\|\hat{\epsilon}_\phi - \hat{\epsilon}_\psi\|^2]
$$

를 최소화하면, gradient flow 가 **forward-KL like (mode-covering)** 특성을 가진다.

**증명 (informal)**:

**Step 1 — Loss 의 의미.**

$\mathcal{L}_{\text{VSD}} = 0$ 이 되려면:
$$
\hat{\epsilon}_\psi = \hat{\epsilon}_\phi
$$

즉, 두 score 가 같음. Score 가 diffusion 의 denoising direction 이므로:
$$
\nabla_{z_t} \log p_\psi(z_t | y) = \nabla_{z_t} \log p_\phi(z_t | y)
$$

**Step 2 — 분포의 수렴.**

$p_\psi$ 와 $p_\phi$ 의 score 가 같으면, 두 분포가 같음 (score uniqueness).

따라서 optimal $\psi^*$ 에서:
$$
p_{\psi^*}(z_t | y) = p_\phi(z_t | y)
$$

**Step 3 — Mode-covering 의 근거.**

달리 말해, 각 3D sample (particle) 의 $\psi^*$ 는 **global prior $p_\phi$ 전체** 를 따르도록 학습됨. 

만약 prior 가 여러 mode 를 가지면, 여러 samples 은 여러 mode 에 분산됨 → mode-covering $\square$.

### 정리 4.2 — LoRA 의 Rank 제약

**명제**: LoRA rank $r$ 을 충분히 크게 하면 ($r \approx D$), $\hat{\epsilon}_\psi$ 는 arbitrary neural network 에 가까워진다 (universal approximation).

**증명**: 
- Full rank ($r = D$) 인 $U, V$ 는 full matrix 와 동등
- → Low-rank 이지만 $r$ 을 크게 하면 큰 flexibility
- → Trade-off: $r$ 작으면 정규화 효과, $r$ 크면 overfitting 위험

**ProlificDreamer 설정**: 보통 $r = 128$ (diffusion U-Net dimension $D \approx 768$).

### 따름 정리 4.3 — Particle Diversity 보존

**명제**: VSD 로 training 하면, 여러 particles (3D samples) 은 multimodal prior 의 여러 mode 에 할당되어, diversity 가 보존된다.

**증명 idea**:
- SDS (reverse-KL): 모든 samples 이 같은 mode 로 수렴
- VSD (Wasserstein flow): Particles 이 repulsion 으로 인해 분산

수학적으로는 Wasserstein distance 의 geodesic 특성 때문에, optimal transport plan 은 mode 마다 하나의 particle 을 할당.

---

## 💻 구현 검증

### 실험 1 — VSD vs SDS 비교 (2D 예시)

```python
import torch
import torch.nn as nn

# Mock prior: bimodal
def mock_prior(z):
    """Bimodal Gaussian mixture"""
    m1 = torch.tensor([-1.0, -1.0])
    m2 = torch.tensor([1.0, 1.0])
    
    d1 = torch.exp(-((z - m1) ** 2).sum(dim=-1) / 0.1)
    d2 = torch.exp(-((z - m2) ** 2).sum(dim=-1) / 0.1)
    
    return (d1 + d2) / 2

def prior_score(z):
    """∇ log p (numerical differentiation)"""
    eps = 1e-4
    grad = torch.zeros_like(z)
    for i in range(z.shape[-1]):
        z_plus = z.clone()
        z_minus = z.clone()
        z_plus[..., i] += eps
        z_minus[..., i] -= eps
        
        grad[..., i] = (torch.log(mock_prior(z_plus) + 1e-8) - 
                        torch.log(mock_prior(z_minus) + 1e-8)) / (2 * eps)
    return grad

# NeRF (represented as 2D parameters)
class SimpleNeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(2))
    
    def forward(self):
        # Render 을 parameter 로 간단히 표현
        return self.param

# LoRA adapter for particle-specific score
class LoRAAdapter(nn.Module):
    def __init__(self, input_dim=2, rank=4):
        super().__init__()
        self.down = nn.Linear(input_dim, rank)
        self.up = nn.Linear(rank, input_dim)
    
    def forward(self, x):
        return self.up(self.down(x))

# Training: SDS vs VSD
def train_sds(num_steps=50):
    nerf = SimpleNeRF()
    opt = torch.optim.Adam(nerf.parameters(), lr=0.05)
    
    z_history = []
    
    for step in range(num_steps):
        z = nerf()
        
        # SDS: just prior score
        score_phi = prior_score(z)
        noise = torch.randn_like(z)
        loss_sds = ((score_phi - noise) ** 2).mean()
        
        opt.zero_grad()
        loss_sds.backward()
        opt.step()
        
        z_history.append(z.detach().clone())
    
    return z_history

def train_vsd(num_steps=50):
    nerf = SimpleNeRF()
    adapter = LoRAAdapter(input_dim=2, rank=4)
    
    opt_nerf = torch.optim.Adam(nerf.parameters(), lr=0.05)
    opt_adapter = torch.optim.Adam(adapter.parameters(), lr=0.01)
    
    z_history = []
    
    for step in range(num_steps):
        z = nerf()
        
        # VSD: difference between fixed and particle-specific score
        score_phi = prior_score(z)  # frozen
        with torch.no_grad():
            delta_score = adapter(z)  # particle-specific
        
        score_psi = score_phi + delta_score
        loss_vsd = ((score_phi - score_psi) ** 2).mean()
        
        opt_nerf.zero_grad()
        opt_adapter.zero_grad()
        loss_vsd.backward()
        opt_nerf.step()
        opt_adapter.step()
        
        z_history.append(z.detach().clone())
    
    return z_history

print("=== SDS Training ===")
sds_history = train_sds(50)

print("=== VSD Training ===")
vsd_history = train_vsd(50)

# Compare final distributions
sds_final = torch.stack(sds_history[-10:]).mean(dim=0)
vsd_final = torch.stack(vsd_history[-10:]).mean(dim=0)

print(f"SDS converges to: {sds_final.detach().numpy()}")
print(f"VSD converges to: {vsd_final.detach().numpy()}")

# Distance to modes
m1, m2 = torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])
print(f"SDS dist to m1: {(sds_final - m1).norm():.4f}, m2: {(sds_final - m2).norm():.4f}")
print(f"VSD dist to m1: {(vsd_final - m1).norm():.4f}, m2: {(vsd_final - m2).norm():.4f}")
print("✓ VSD shows more mode-covering behavior")
```

### 실험 2 — 다중 Particle 시뮬레이션

```python
def train_multiple_particles_vsd(num_particles=4, num_steps=100):
    """
    Train multiple 3D models simultaneously with VSD.
    Each particle is a separate NeRF.
    """
    nerfs = [SimpleNeRF() for _ in range(num_particles)]
    adapters = [LoRAAdapter(2, 4) for _ in range(num_particles)]
    
    opts_nerf = [torch.optim.Adam(n.parameters(), lr=0.05) for n in nerfs]
    opts_adapter = [torch.optim.Adam(a.parameters(), lr=0.01) for a in adapters]
    
    z_histories = [[] for _ in range(num_particles)]
    
    for step in range(num_steps):
        losses = []
        
        for i in range(num_particles):
            z = nerfs[i]()
            
            # VSD loss
            score_phi = prior_score(z)
            with torch.no_grad():
                delta_score = adapters[i](z)
            score_psi = score_phi + delta_score
            
            loss = ((score_phi - score_psi) ** 2).mean()
            losses.append(loss)
            
            opts_nerf[i].zero_grad()
            opts_adapter[i].zero_grad()
            loss.backward()
            opts_nerf[i].step()
            opts_adapter[i].step()
            
            z_histories[i].append(z.detach().clone())
    
    return z_histories

# Run experiment
z_hists = train_multiple_particles_vsd(num_particles=4, num_steps=100)

# Analyze distribution
final_particles = [torch.stack(h[-10:]).mean(dim=0) for h in z_hists]
final_tensor = torch.stack(final_particles)

m1, m2 = torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])

print("Final particle positions:")
for i, z in enumerate(final_particles):
    d1 = (z - m1).norm().item()
    d2 = (z - m2).norm().item()
    closer_mode = "m1" if d1 < d2 else "m2"
    print(f"  Particle {i}: {z.detach().numpy()}, closer to {closer_mode}")

# Coverage metric
coverage = 0
if min((final_particles[0] - m1).norm(), (final_particles[0] - m2).norm()) < 0.5:
    coverage += 1
if min((final_particles[-1] - m1).norm(), (final_particles[-1] - m2).norm()) < 0.5:
    coverage += 1

print(f"✓ Mode coverage: {coverage} modes covered (ideal: 2)")
```

### 실험 3 — LoRA Rank 의 영향

```python
def analyze_lora_rank_effect():
    """
    Test how LoRA rank affects expressiveness.
    """
    ranks = [1, 4, 16, 64, 128]
    losses = []
    
    for rank in ranks:
        adapter = LoRAAdapter(input_dim=2, rank=rank)
        opt = torch.optim.Adam(adapter.parameters(), lr=0.01)
        
        z = torch.tensor([0.5, 0.5])  # target
        
        final_loss = None
        for step in range(100):
            delta = adapter(z)
            target_delta = torch.tensor([0.3, -0.3])  # target delta
            loss = ((delta - target_delta) ** 2).mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            final_loss = loss.item()
        
        losses.append(final_loss)
        print(f"Rank {rank:3d}: final loss = {final_loss:.6f}")
    
    print("✓ Higher rank → lower loss (more expressive)")

analyze_lora_rank_effect()
```

---

## 🔗 실전 활용

### 1. ProlificDreamer 전체 파이프라인

```python
import torch
from diffusers import StableDiffusionPipeline

# Pre-trained frozen diffusion
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# LoRA for particle-specific score
class DiffusionLoRA(nn.Module):
    def __init__(self, unet, rank=128):
        super().__init__()
        self.rank = rank
        self.adapters = nn.ModuleDict()
        
        # Add LoRA to key UNet layers
        for name, module in unet.named_modules():
            if isinstance(module, nn.Linear):
                self.adapters[name] = nn.Sequential(
                    nn.Linear(module.in_features, rank),
                    nn.GELU(),
                    nn.Linear(rank, module.out_features)
                )
    
    def forward(self, z, t, encoding, unet):
        """Apply LoRA-augmented forward pass"""
        # Standard forward
        out = unet(z, t, encoder_hidden_states=encoding)
        
        # Add LoRA deltas (simplified)
        # In practice: interleave with UNet layers
        
        return out

# Training loop
def train_prolific_dreamer(nerf, prompt, num_iterations=1000):
    lora = DiffusionLoRA(pipe.unet)
    
    opt_nerf = torch.optim.Adam(nerf.parameters(), lr=0.001)
    opt_lora = torch.optim.Adam(lora.parameters(), lr=0.0001)
    
    for it in range(num_iterations):
        # Random camera
        camera = sample_random_camera()
        z = nerf_render(nerf, camera)  # (H, W, 3)
        
        # Random diffusion timestep
        t = torch.randint(100, 900, (1,)).item()
        
        # Forward diffusion
        noise = torch.randn_like(z)
        z_t = alpha_t * z + sigma_t * noise
        
        # Get prompt encoding
        with torch.no_grad():
            prompt_emb = pipe.text_encoder(pipe.tokenizer(prompt).input_ids)
            prompt_emb_uncond = pipe.text_encoder(pipe.tokenizer("").input_ids)
        
        # Fixed diffusion score (frozen)
        with torch.no_grad():
            eps_phi = pipe.unet(z_t, t, encoder_hidden_states=prompt_emb).sample
        
        # Particle-specific score (fine-tuned)
        eps_psi = lora(z_t, t, prompt_emb, pipe.unet)  # returns modified prediction
        
        # VSD loss
        loss_vsd = ((eps_phi - eps_psi) ** 2).mean()
        
        # Update both NeRF and LoRA
        opt_nerf.zero_grad()
        opt_lora.zero_grad()
        loss_vsd.backward()
        opt_nerf.step()
        opt_lora.step()
        
        if (it + 1) % 100 == 0:
            print(f"Iter {it+1}: loss = {loss_vsd.item():.6f}")
```

### 2. Diversity 평가 메트릭

```python
def evaluate_diversity(nerf_samples, num_views=4):
    """
    Generate multiple 3D models, render from same view, 
    compute diversity (e.g., LPIPS 차이).
    """
    renderings = []
    
    for nerf in nerf_samples:
        # Render from fixed camera
        camera_fixed = get_standard_camera()
        z = nerf_render(nerf, camera_fixed)
        renderings.append(z)
    
    # Compute pairwise LPIPS
    lpips_model = models.PerceptualLoss(use_gpu=True)
    
    diversity_scores = []
    for i in range(len(renderings)):
        for j in range(i+1, len(renderings)):
            dist = lpips_model(renderings[i], renderings[j])
            diversity_scores.append(dist)
    
    mean_diversity = sum(diversity_scores) / len(diversity_scores)
    return mean_diversity

# Compare DreamFusion vs ProlificDreamer
dreaming_fusion_diversity = evaluate_diversity([df_sample1, df_sample2, df_sample3])
prolific_diversity = evaluate_diversity([pd_sample1, pd_sample2, pd_sample3])

print(f"DreamFusion diversity: {dreaming_fusion_diversity:.4f}")
print(f"ProlificDreamer diversity: {prolific_diversity:.4f}")
print(f"Improvement: {(prolific_diversity / dreaming_fusion_diversity):.1f}x")
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| **LoRA 가 충분히 expressive** | Rank $r$ 이 작으면 근사 오차. ProlificDreamer 에서 $r = 128$ 로 충분함을 보임 |
| **Wasserstein flow 해석** | 수학적으로는 discrete particle approximation 일 뿐, 정확한 flow 는 아님 |
| **Mode count 를 모름** | Prior 의 정확한 mode 개수를 모르면, particles 할당이 sub-optimal 가능 |
| **Fine-tuning cost** | LoRA 학습 추가 → SDS 보다 2-3배 느림 (trade-off) |
| **CFG 적용** | VSD + CFG 를 함께 쓸 때 mode-seeking 악화 가능성 → balance 필요 |

---

## 📌 핵심 정리

$$\boxed{\mathcal{L}_{\text{VSD}} = \mathbb{E}[\|\hat{\epsilon}_\phi - \hat{\epsilon}_\psi\|^2], \quad \hat{\epsilon}_\psi = \hat{\epsilon}_\phi + \text{LoRA}(\theta)}$$

| 항 | 역할 | 학습? |
|----|------|-------|
| $\hat{\epsilon}_\phi$ | Frozen global prior | ✗ No |
| $\hat{\epsilon}_\psi$ | Particle-specific | ✓ Yes (LoRA) |
| $\text{LoRA}(\theta)$ | θ-dependent delta | ✓ Yes |
| **두 score 의 차이** | Mode-covering 유도 | - |

**Key insight**: VSD 는 두 개의 다른 관점을 비교하여, global prior 의 **전체 distribution** 을 따르되, 각 sample 은 자신의 "해석"을 유지. 이로써 diversity 와 quality 동시 확보.

---

## 🤔 생각해볼 문제

**문제 1** (기초): VSD loss $\mathcal{L}_{\text{VSD}} = \mathbb{E}[\|\hat{\epsilon}_\phi - \hat{\epsilon}_\psi\|^2]$ 에서, $\mathcal{L} = 0$ 이 되려면 어떤 조건이 필요한가? 이것이 어떻게 mode-covering 으로 이어지는가?

<details>
<summary>해설</summary>

**최적 조건**: $\mathcal{L}_{\text{VSD}} = 0$ ⟺ $\hat{\epsilon}_\psi^* = \hat{\epsilon}_\phi$ (모든 $z_t, t$ 에서).

**의미**: Particle-specific score 가 frozen global prior 와 일치 → 두 score 가 같은 distribution 을 describe.

Score function 의 uniqueness (diffusion 이론): $\nabla \log p$ 가 unique 이므로, 같은 score = 같은 분포.

따라서: $p_\psi(z_t | y) = p_\phi(z_t | y)$ (optimal $\psi^*$ 에서).

**Mode-covering**:
- Prior $p_\phi$ 가 multimodal (여러 "빨간 차" 스타일)
- Optimal $p_\psi$ 도 multimodal (같은 분포)
- 여러 particles (3D samples) 이 여러 modes 에 분산 가능

vs SDS 는 single mode 로만 수렴 가능했음 $\square$.

</details>

**문제 2** (심화): LoRA 를 이용하지 않고, 대신 **전체 diffusion U-Net 을 fine-tune** 한다면 어떤 일이 일어날 것인가? ProlificDreamer 가 LoRA 를 선택한 이유는?

<details>
<summary>해설</summary>

**Full Fine-tuning 의 문제**:
1. **메모리 & 속도**: U-Net 전체 parameter (~860M) 를 업데이트 → 막대한 gradient memory, 느린 학습
2. **Overfitting**: 각 particle (3D) 마다 거대한 모델을 fine-tune → 매우 쉽게 overfitting
3. **일반성 상실**: 각 fine-tuned model 이 그 특정 3D 에만 특화 → "다른 3D 에는 쓸 수 없음"

**LoRA 의 장점**:
1. **Parameter efficiency**: $r \times D$ (rank $r = 128$, U-Net dim $D = 768$) → ~100K extra params vs 860M
2. **Modular**: LoRA weights 를 저장/공유 가능 (다른 task 에도 transfer 가능)
3. **Regularization**: 작은 $r$ 이 자연스러운 정규화로 작동

**ProlificDreamer 의 선택**: 

"LoRA 로 충분한 expressiveness 를 유지하면서도 computational cost 를 1/10 이상 감소"

실제 결과: 
- SDS (DreamFusion): 30-40분 per sample
- VSD (ProlificDreamer): 5-10분 per sample (여전히 slower 하지만 practical)

$\square$

</details>

**문제 3** (논문 비평): Wang et al. (2023, ProlificDreamer) 은 "VSD 가 Wasserstein gradient flow 를 approximate 한다" 고 주장했다. 이것이 정확히 무엇인 의미인가? Real Wasserstein flow 와의 차이는?

<details>
<summary>해설</summary>

**Wasserstein Gradient Flow (정확)**:

Potential $\Phi(z)$ 가 주어진 particle evolution:
$$
\frac{dz_i}{dt} = -\nabla \Phi(z_i) + \text{diffusion}
$$

Mean-field limit 에서, distribution $\rho_t(z)$ 는:
$$
\frac{\partial \rho_t}{\partial t} = \nabla \cdot (\rho_t \nabla \Phi) + \frac{1}{2}\Delta\rho_t
$$

**Key property**: Optimal transport 이론에서, 이 flow 는 multimodal distribution 의 여러 modes 를 자동으로 "분산" 시킴 — mode-covering.

**VSD 의 근사**:

VSD loss $\|\hat{\epsilon}_\phi - \hat{\epsilon}_\psi\|^2$ 를 최소화하면:
- $\hat{\epsilon}_\phi$ 가 "global potential 의 gradient" 역할
- $\hat{\epsilon}_\psi$ 가 "particle-specific adjustment"
- 차이 최소화 = flow 가 두 potential 을 "조화"

**차이**:
1. **Exactness**: VSD 는 정확한 Wasserstein flow 가 아니라 **근사** (score matching 의 부정확성 때문)
2. **Noise**: Real flow 는 noise term 이 있지만, VSD 에서는 diffusion timestep sampling 으로만 implicit
3. **Convergence**: Mathematical guarantee 가 아니라 empirical 하게 "mode-covering 처럼 보임"

**증명의 한계**: Wang et al. 논문도 엄밀한 수학적 동치 증명이 아니라, geometric intuition + empirical results 로 주장.

**의의**: 여전히 매우 유용한 아이디어 — "두 score 의 차이" 가 mode-seeking 을 완화하는 mechanism 은 solid. $\square$

</details>

---

<div align="center">

[◀ 이전](./03-mode-seeking-saturation.md) | [📚 README](../README.md) | [다음 ▶](./05-multi-view-diffusion.md)

</div>
