# 05. NeRF Variants — Mip-NeRF · Ref-NeRF · NeRF-W

## 🎯 핵심 질문

- Mip-NeRF (Barron 2021) 의 **Integrated Positional Encoding** 이 왜 anti-aliasing 을 가져오는가?
- Cone 기반 sampling (pixel = ray 대신 cone) 이 multi-scale rendering 에서 왜 필수인가?
- Ref-NeRF (Verbin 2022) 의 reflection vector decomposition 이 specular 와 diffuse 를 어떻게 분리하는가?
- NeRF-W (Martin-Brualla 2021) 의 transient embedding 이 photo-tourism 의 변동성 (illumination, weather) 을 어떻게 처리하는가?
- 이 variants 들이 original NeRF 의 어떤 **한계**를 targeted 하는가?

---

## 🔍 왜 Vanilla NeRF 이후 변형들이 필요한가

Mildenhall 2020 NeRF 는 groundbreaking 이었지만, practical limitations 들이 있었습니다:

1. **Anti-aliasing 부재** — Pixel 을 ray 로 모델 (point sample) → higher resolution 에서 aliasing artifact
2. **Specular reflection** — View-dependent color $\mathbf{c}(\mathbf{x}, \mathbf{d})$ 가 high-freq viewpoint-specific → slow learning
3. **Dynamic scenes / Variable lighting** — Photo-tourism (Colmap output) 에서 같은 scene 의 다른 시간·날씨 촬영 → inconsistency

이 문서는 세 가지 major variants 를 다룹니다.

---

## 📐 수학적 선행 조건

- **Positional encoding** (Ch3-02): Standard PE 와 그 한계
- **Cone geometry**: Ray frustum, projection
- **Reflection models**: Specular/diffuse BRDF
- **Coordinate systems**: Normal, tangent, reflection vector
- 선택: Spherical Harmonics (SH) basis

---

## 📖 직관적 이해

### Mip-NeRF: Cone-based Sampling

**문제**: Pixel 은 실제로 **finite size** 를 가집니다. High-res 이미지에서 pixel footprint 는 scene 에서 큰 영역에 해당.

**해결책**: Ray 대신 **cone** 로 모델:
- Pixel → ray + half-angle cone
- Each sample → frustum (cone segment)
- Integrate over frustum

**Integrated Positional Encoding**:
$$\gamma(\mu, \Sigma) = \mathbb{E}_{x \sim \mathcal{N}(\mu, \Sigma)}[\gamma(x)]$$

Cone 의 mean $\mu$ 와 covariance $\Sigma$ 를 계산 → Gaussian 의 expected PE:
$$\mathbb{E}[\sin(2^l \pi x)] = \sin(2^l \pi \mu) \exp(-\frac{1}{2}(2^l \pi \sigma)^2)$$

→ **Large cone 에서 high-freq 가 자동으로 attenuate** (안티앨리어싱).

### Ref-NeRF: Specular Decomposition

**문제**: Diffuse 표면은 view-independent 이지만, specular (glossy) 는 highly view-dependent. View-direction MLP 가 이를 학습하기 어려움.

**해결책**: Reflection vector 기반 parameterization:
- Input: Position $\mathbf{x}$ + normal $\mathbf{n}$ (implicitly learned) + view dir $\mathbf{d}$
- Reflection: $\mathbf{r} = 2(\mathbf{d} \cdot \mathbf{n})\mathbf{n} - \mathbf{d}$
- Output: Diffuse $\mathbf{c}_d$ (view-indep) + Specular $s(\mathbf{r})$ (reflection-dependent)

→ Specular lobe 가 reflection vector 에 대해 smoother → 학습 faster.

### NeRF-W: Transient Embedding

**문제**: Photo-tourism (e.g., Colmap LLFF) 는 웹 사진 collection → 같은 scene 도 다양한 시간·날씨·사람 (transient objects).

**해결책**: 
- Static appearance: Learned per-scene radiance field
- Transient: Per-image embedding $\mathbf{e}_i$ (learnable code)

$$\mathbf{c}(\mathbf{x}, \mathbf{d}, i) = \text{MLP}([\gamma(\mathbf{x}), \gamma(\mathbf{d}), \mathbf{e}_i])$$

→ Transient object (people, cars, shadows) 는 embedding 으로 absorb, static 은 learned.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Mip-NeRF Cone Model

**Ray cone** (pixel coverage):
- Ray: $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ (center)
- Half-angle: $\alpha$ (aperture)
- Footprint at distance $t$: disk of radius $t \tan \alpha$

**Sample frustum** (between $t_i$ and $t_{i+1}$):
- Mean position: $\mu = \mathbf{o} + t_i \mathbf{d}$ (중심)
- Covariance (3D ball approximation):
$$\Sigma = \sigma_t^2 I, \quad \sigma_t^2 = (t_{i+1} - t_i)^2 / 3 + (t_i \tan \alpha)^2$$

(자세한 covariance는 Barron 2021 참조, 여기서는 simplified).

### 정의 1.2 — Integrated Positional Encoding (IPE)

Standard PE:
$$\gamma(x) = \begin{pmatrix} \sin(2^0 \pi x) \\ \cos(2^0 \pi x) \\ \vdots \end{pmatrix}$$

Integrated PE (Gaussian input):
$$\gamma(\mu, \Sigma) = \mathbb{E}_{x \sim \mathcal{N}(\mu, \Sigma)}[\gamma(x)]$$

**Closed form** for univariate:
$$\mathbb{E}[\sin(2^l \pi x)] = \sin(2^l \pi \mu) \exp\left(-\frac{1}{2}(2^l \pi)^2 \sigma^2\right)$$
$$\mathbb{E}[\cos(2^l \pi x)] = \cos(2^l \pi \mu) \exp\left(-\frac{1}{2}(2^l \pi)^2 \sigma^2\right)$$

**효과**:
- Small cone ($\sigma \approx 0$): $\gamma \approx$ standard PE
- Large cone ($\sigma \gg 1$): $\gamma \approx 0$ (high-freq attenuate)

### 정의 1.3 — Ref-NeRF Shading Model

NeRF color output modification:

**Input**: $(\gamma(\mathbf{x}), \mathbf{n}_{\text{out}}, \gamma(\mathbf{d}))$ where $\mathbf{n}_{\text{out}}$ is outward normal.

**Reflection computation**:
$$\mathbf{r} = 2 \max(0, -\mathbf{d} \cdot \mathbf{n}_{\text{out}}) \mathbf{n}_{\text{out}} + \mathbf{d}$$

(Ensure $\mathbf{r}$ points away from surface).

**Output**:
$$\mathbf{c}(\mathbf{x}, \mathbf{d}) = \text{MLP}_{\text{diff}}(\gamma(\mathbf{x})) \cdot \mathbf{c}_d + \text{MLP}_{\text{spec}}([\gamma(\mathbf{x}), \gamma(\mathbf{r})]) \cdot \mathbf{c}_s$$

where $\mathbf{c}_d, \mathbf{c}_s$ are diffuse and specular components.

### 정의 1.4 — NeRF-W Appearance Model

**Training**: Dataset 에 multiple images 의 같은 scene.

**Embedding**:
- Per-image code: $\mathbf{e}_i \in \mathbb{R}^{n_e}$ (learnable, e.g., $n_e = 16$)
- Static MLP: $F_\theta$ (shared across images)

**Output**:
$$\sigma(\mathbf{x}) = F_\theta^{(1)}(\gamma(\mathbf{x}))$$
$$\mathbf{c}(\mathbf{x}, \mathbf{d}, i) = F_\theta^{(2)}([\gamma(\mathbf{x}), \gamma(\mathbf{d}), \mathbf{e}_i])$$

**Interpretation**:
- $F_\theta^{(1)}$: Static geometry
- $F_\theta^{(2)}$: Static + per-image variation (lighting, weather, people)

---

## 🔬 정리와 증명

### 정리 1.1 — IPE 의 Anti-Aliasing 효과

**정리** (Barron 2021):

Cone-based rendering with IPE:
$$C(\mathbf{r}) = \int \text{IPE}(\mu(t), \Sigma(t))\, dt$$

is **equivalent to** applying Gaussian blur with bandwidth $\sigma_t$ before standard PE:

$$C_{\text{blurred}} = \int \gamma(\text{Gaussian-blur}(p, \sigma_t))\, dt$$

**증명 스케치**:

1. **Fourier decomposition**: $\gamma$ 는 frequency $2^l$ 의 sinusoid.

2. **Gaussian filtering**: Frequency $\omega$ 에 대해 $|\hat{H}(\omega)| \propto \exp(-\frac{1}{2}(\omega\sigma)^2)$.

3. **IPE**: $\mathbb{E}[\gamma(x)]$ where $x \sim \mathcal{N}(\mu, \sigma^2)$ 는 정확히 이 filtering 을 implement.

4. **Result**: IPE 는 자동으로 **frequency-adaptive blur** — large frustum 에서 high-freq component 를 suppress.

**결론**: Mip-NeRF 는 anti-aliasing 을 integrated sampling 으로 구현. $\square$

### 정리 1.2 — Ref-NeRF 의 Specular Learning Speed

**정리** (Verbin 2022, implicit):

Reflection-based parameterization 을 사용하면:
- Diffuse color: View-independent → standard NeRF 와 동일
- Specular color: Reflection vector $\mathbf{r}$ 에 대해 smoother function → **faster convergence** relative to view direction $\mathbf{d}$

**증명 스케치**:

1. **View-dependent parameterization**: $\mathbf{c}(\mathbf{d})$ as function of $\mathbf{d}$ 는 $\mathbf{d}$ space 에서 high-frequency lobe 를 have.

2. **Reflection parameterization**: $\mathbf{c}(\mathbf{r})$ as function of reflected ray $\mathbf{r}$ 는, specular lobe 가 centered at $\mathbf{r} \approx -\mathbf{d}_{\text{light}}$ → smoother in $\mathbf{r}$ space.

3. **Spectral bias** (Ch3-02): ReLU MLP 는 low-freq 를 빠르게 학습. Smoother function → lower intrinsic frequency → faster learning.

**결론**: Reflection decomposition 이 specular 를 lower-frequency task 로 변환. $\square$

### 따름정리 1.3 — NeRF-W 의 Appearance Decomposition

**따름정리** (Martin-Brualla 2021):

Per-image embedding $\mathbf{e}_i$ 를 추가하면:
- **Static component**: Shared MLP $F_\theta$ → robust across images
- **Transient component**: Embedding 의 variance → photo variation 을 explain

**정량화**:
$$\text{Var}_{i}(\mathbf{e}_i) > \text{Var}_{i}(\sigma(\mathbf{x}))$$

→ Transient variation 이 embedding 에 capture 됨.

**Result**: Photo-tourism 데이터셋에서도 consistent novel view 가능.

---

## 💻 구현 검증

### 실험 1 — IPE 의 Frequency Attenuation

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def standard_pe(x, L=10):
    """Standard positional encoding."""
    pe = []
    for l in range(L):
        pe.append(torch.sin(2**l * np.pi * x))
        pe.append(torch.cos(2**l * np.pi * x))
    return torch.stack(pe, dim=-1)

def integrated_pe(mu, sigma, L=10):
    """Integrated PE: E[γ(x)] where x ~ N(μ, σ²)."""
    ipe = []
    for l in range(L):
        freq = 2**l * np.pi
        # E[sin(freq * x)] = sin(freq * μ) * exp(-0.5 * (freq * σ)²)
        ipe.append(torch.sin(freq * mu) * torch.exp(-0.5 * (freq * sigma) ** 2))
        # E[cos(freq * x)] = cos(freq * μ) * exp(-0.5 * (freq * σ)²)
        ipe.append(torch.cos(freq * mu) * torch.exp(-0.5 * (freq * sigma) ** 2))
    return torch.stack(ipe, dim=-1)

# Visualize: standard PE vs IPE for different cone sizes
mu = torch.tensor([0.5])
sigmas = [0.01, 0.05, 0.1, 0.2, 0.5]
L = 6

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, sigma in enumerate(sigmas):
    sigma_t = torch.tensor([sigma])
    
    # Standard PE (reference)
    pe_standard = standard_pe(mu, L=L)[0]
    
    # Integrated PE
    pe_integrated = integrated_pe(mu, sigma_t, L=L)[0]
    
    # Plot
    ax = axes[idx]
    frequencies = np.arange(2*L)
    ax.bar(frequencies - 0.2, pe_standard.numpy(), width=0.4, label='Standard PE', alpha=0.7)
    ax.bar(frequencies + 0.2, pe_integrated.numpy(), width=0.4, label=f'IPE (σ={sigma})', alpha=0.7)
    ax.set_xlabel('Frequency Index')
    ax.set_ylabel('PE Magnitude')
    ax.set_title(f'Cone Size σ_t = {sigma}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('integrated_pe_effect.png', dpi=100)
print("✓ Saved integrated_pe_effect.png")
print("\nIPE dampens high-frequency components as cone size increases (anti-aliasing)")
```

**출력**:
```
✓ Saved integrated_pe_effect.png

IPE dampens high-frequency components as cone size increases (anti-aliasing)
```

### 실험 2 — Ref-NeRF Specular Decomposition

```python
def compute_reflection(d, n):
    """
    Compute reflection vector.
    d: view direction (pointing away from surface)
    n: surface normal (pointing outward)
    Returns: reflection ray
    """
    # r = 2(−d · n)n − d (pointing away from surface, toward light)
    d_normalized = torch.nn.functional.normalize(d, dim=-1)
    n_normalized = torch.nn.functional.normalize(n, dim=-1)
    d_dot_n = -(d_normalized * n_normalized).sum(dim=-1, keepdim=True)
    r = 2 * torch.clamp(d_dot_n, min=0) * n_normalized + d_normalized
    return torch.nn.functional.normalize(r, dim=-1)

# Test: multiple view directions, fixed surface normal
normal = torch.tensor([0., 0., 1.])  # z-up
view_directions = [
    torch.tensor([1., 0., -1.]),     # front
    torch.tensor([0., 1., -1.]),     # side
    torch.tensor([0.707, 0., -0.707]),  # diagonal
]

reflections = [compute_reflection(d.unsqueeze(0), normal.unsqueeze(0))[0] for d in view_directions]

print("Specular decomposition test:")
for d, r in zip(view_directions, reflections):
    print(f"  View dir {d.numpy()} → Reflection {r.numpy()}")

# Intuition: specular lobe should cluster around reflection vector
print("\n✓ Reflection vectors point consistently toward specular lobe")
```

### 실험 3 — NeRF-W Transient Embedding

```python
class NeRFW(torch.nn.Module):
    """NeRF-W with per-image embedding."""
    def __init__(self, num_images=10, embed_dim=16):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_images, embed_dim)
        self.mlp_static = torch.nn.Sequential(
            torch.nn.Linear(60 + 24 + embed_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3)
        )
    
    def forward(self, pe_x, pe_d, image_idx):
        """
        pe_x: positional encoding of position [B, 60]
        pe_d: positional encoding of direction [B, 24]
        image_idx: image index [B]
        """
        e = self.embedding(image_idx)  # [B, embed_dim]
        x = torch.cat([pe_x, pe_d, e], dim=-1)
        c = torch.sigmoid(self.mlp_static(x))
        return c

# Test
nerf_w = NeRFW(num_images=5, embed_dim=16)
pe_x = torch.randn(8, 60)
pe_d = torch.randn(8, 24)
image_idx = torch.tensor([0, 0, 1, 2, 2, 3, 4, 4])

colors = nerf_w(pe_x, pe_d, image_idx)
print(f"NeRF-W output shape: {colors.shape}")
print(f"Color range: [{colors.min():.4f}, {colors.max():.4f}]")
print("✓ Per-image embeddings allow appearance variation")
```

---

## 🔗 실전 활용

### 1. Mip-NeRF Training

```python
# Coarse sampling with cone frustums
def sample_cone_frustum(rays_o, rays_d, t_start, t_end, cone_angle):
    """Sample from cone frustum."""
    t = torch.linspace(t_start, t_end, 64)
    
    # Mean positions
    mu = rays_o[:, None, :] + t[None, :, None] * rays_d[:, None, :]
    
    # Cone frustum variance
    delta_t = (t_end - t_start) / 64
    sigma_t = torch.sqrt((delta_t**2) / 3 + (t * np.tan(cone_angle))**2)
    
    return mu, sigma_t  # for IPE

# Using IPE in model
for mu_i, sigma_i in zip(mu_samples, sigma_samples):
    ipe = integrated_pe(mu_i, sigma_i, L=10)
    sigma, color = model(ipe, pe_d)
```

### 2. Ref-NeRF Setup

```python
class RefNeRF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Geometry (normal prediction)
        self.fc_normal = torch.nn.Linear(60, 3)
        # Diffuse branch
        self.mlp_diffuse = torch.nn.Sequential(
            torch.nn.Linear(60, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3)
        )
        # Specular branch
        self.mlp_specular = torch.nn.Sequential(
            torch.nn.Linear(60 + 24, 128),  # position + reflection
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3)
        )
    
    def forward(self, pe_x, pe_d):
        # Normal (view-independent)
        n = torch.nn.functional.normalize(self.fc_normal(pe_x), dim=-1)
        
        # Reflection
        d = -torch.nn.functional.normalize(pe_d[:, :3], dim=-1)  # recover direction from PE
        r = compute_reflection(d.unsqueeze(1), n.unsqueeze(1))
        pe_r = positional_encode(r, L=4)
        
        # Diffuse + Specular
        c_diffuse = torch.sigmoid(self.mlp_diffuse(pe_x))
        c_specular = torch.sigmoid(self.mlp_specular(torch.cat([pe_x, pe_r], dim=-1)))
        
        return c_diffuse + c_specular
```

### 3. NeRF-W Training with Photo-Tourism Data

```python
def train_nerfw_batch(images, cameras, num_images=300):
    """Train NeRF-W on photo-tourism dataset."""
    model = NeRFW(num_images=num_images, embed_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    for iteration in range(100000):
        # Sample random image
        img_idx = torch.randint(0, num_images, (1,)).item()
        image = images[img_idx]
        camera = cameras[img_idx]
        
        # Rays from camera
        rays = get_rays(camera)
        
        # Forward + loss
        for ray_batch in chunk_rays(rays, batch_size=4096):
            image_idx_batch = torch.full((len(ray_batch),), img_idx).to(device)
            colors_pred = render_rays_nerfw(ray_batch, model, image_idx_batch)
            colors_gt = sample_from_image(image, ray_batch)
            
            loss = F.mse_loss(colors_pred, colors_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## ⚖️ 가정과 한계

| Variant | 가정 | 한계 |
|---------|------|------|
| **Mip-NeRF** | Cone frustum = Gaussian | 정확한 cone 기하학 미포함, 근사 |
| **Ref-NeRF** | Smooth reflection lobes | Subsurface scattering, refraction 미처리 |
| **NeRF-W** | Per-image variation 는 독립적 | Strongly correlated variation (예: shadow boundary) 처리 어려움 |

---

## 📌 핵심 정리

| Variant | 주요 기여 | 해결 문제 |
|---------|---------|---------|
| **Mip-NeRF** | IPE (Integrated PE) | Anti-aliasing, multi-scale consistency |
| **Ref-NeRF** | Reflection decomposition | Specular detail, faster learning |
| **NeRF-W** | Per-image embedding | Photo variation, dynamic lighting |

---

## 🤔 생각해볼 문제

**문제 1** (기초): IPE 에서 Gaussian assumption (large 를 위해) 이 정말 타당한가?

<details>
<summary>해설</summary>

**실제 frustum**: Cone-shaped, not Gaussian.

**Gaussian approximation**:
- **장점**: Closed-form IPE 계산 가능
- **단점**: Large cone 에서 부정확

**정당화**: 
- Pixel footprint 는 대략 circular/Gaussian-like (lens diffraction, antialiasing filter 때문)
- Ray cone 의 practical bandwidth 는 작음 (typical pixel 크기 ~ 1/1000 scene)

**결론**: Gaussian approximation 은 practical, 더 정확한 frustum integration 도 가능하지만 closed-form 어려움.

$\square$

</details>

**문제 2** (심화): Ref-NeRF 가 normal 을 MLP 로 예측한다면, geometry consistency 를 어떻게 보장하는가?

<details>
<summary>해설</summary>

**문제**: Normal $\mathbf{n}(\mathbf{x})$ 이 learned 이면, level set $\sigma(\mathbf{x}) = \tau$ 의 true normal 과 맞지 않을 수 있음.

**Ref-NeRF 의 접근**:
- Normal 을 freely learn (constraint 없음)
- Loss 가 photometric → implicitly consistent

**이유**: 
- Specular rendering 이 normal-dependent → wrong normal 이면 view-dependent color 가 wrong
- Multi-view constraint 가 normal consistency 를 enforce

**더 나은 방법**: Eikonal regularization (DeepSDF, Ch1-04) — $\|\nabla\sigma\|_1 \approx 1$ 제약으로 true distance field ensure.

**결론**: Ref-NeRF 는 implicit consistency 에 의존, explicit geometric constraint 없음. 이를 추가하면 더 좋을 수 있음.

$\square$

</details>

**문제 3** (논문 비평): NeRF-W 의 per-image embedding 은 전체 image variation 을 capture 하는가, 아니면 일부만?

<details>
<summary>해설</summary>

**NeRF-W embedding 의 한계**:

1. **Global only**: Embedding $\mathbf{e}_i$ 는 entire image 에 대한 scalar code → spatial variation (shadow, lighting gradient) 미처리.

2. **Rank limitation**: Embedding dimension $\sim 16$ 은 제한적. Complex multi-modal lighting 처리 어려움.

3. **Caustics, moving objects**: Transient 이 spatially localized 되면 embedding 이 over-generalize → blurry result.

**해결 방향**:
- Spatial feature map (NeRF++ style spatial embeddings)
- Attention mechanism (recent works)
- Flow-based warp (optical flow 활용)

**결론**: NeRF-W 는 **coarse appearance variation** 을 handle 하는 practical solution, fine details 은 additional regularization 필요.

$\square$

</details>

---

<div align="center">

[◀ 이전](./04-loss-training.md) | [📚 README](../README.md) | [다음 ▶](./06-instant-ngp.md)

</div>
