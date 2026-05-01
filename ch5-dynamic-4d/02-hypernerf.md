# 02. HyperNeRF — Topology Change 와 Ambient Slicing

## 🎯 핵심 질문

- Nerfies (canonical + deformation) 는 **fixed topology** 를 가정하는데, 왜 입 열림/닫힘 같은 topology change 를 표현할 수 없는가?
- Canonical space 를 $\mathbb{R}^3$ 에서 $\mathbb{R}^{3+H}$ (ambient space) 로 확장하면 어떻게 topology change 가 가능해지는가?
- Time-conditional slicing surface $\mathbf{w}(t) \in \mathbb{R}^H$ 는 정확히 무엇이고, $H = 2$ 차원이 왜 충분한가?
- HyperNeRF (Park 2021) 의 이론적 근거와 실무적 구현은 무엇인가?

---

## 🔍 왜 Topology Change 가 필요한가

Nerfies 의 한계:

$$\mathbf{x}^{\text{can}} = \mathbf{x} + D_\psi(\mathbf{x}, t) \quad \text{(fixed topology)}$$

이 formulation 은 **homeomorphism** (일대일 연속 대응) 을 가정합니다. 즉:
- 각 $t$ 에서 canonical space 의 점들이 observed space 로 **일대일 대응**
- Topology 가 변할 수 없음 (연결성, 구멍 개수 등)

하지만 실제 장면에서:
1. **입 열림/닫힘**: 입 안쪽이 visible/hidden 으로 전환
2. **손가락 교차**: 두 surface 가 만남 (self-intersection)
3. **Cloth folding**: 천이 겹침

이런 경우 **single canonical space** 만으로는 불가능 → $\mathbb{R}^{3+H}$ ambient space 로 확장.

---

## 📐 수학적 선행 조건

- **Differential geometry**: manifold, submanifold, codimension
- **Level set method**: implicit surface as $w(\mathbf{x}) = c$
- **01. Deformable NeRF**: canonical space, deformation field
- Topology: fundamental group, Euler characteristic

---

## 📖 직관적 이해

### Ambient Space 의 직관

3D space $\mathbb{R}^3$ 에서는 closed surface (입) 를 나타낼 수 없습니다:
- 표면 내부와 외부의 연결성이 일정
- 시간에 따라 열림/닫힘 은 **topological singularity**

하지만 **4D space $\mathbb{R}^4$** (3D + 1 dimension) 에서는:
- Canonical surface 를 slice 할 수 있음
- 시간에 따라 다른 depth 에서 다른 부분을 보임

**구체 예**: 2D plane 에서 line 을 시간에 따라 slice 하기

```
        w  (hidden dimension)
        ↑
        │   ●  (t=0)  w(t=0)
        │  ╱
        │ ╱
        │●────────────► t (time)
        │ ╲
        │  ●  (t=1)  w(t=1)
        └─────────────→ x
```

시간에 따라 **다른 w 값에서 슬라이스** 하면, 2D object 가 시간에 따라 "모양" 이 변함. 이를 3D 에 확대.

### H = 2 차원이 충분한 이유

Ambient space 를 $\mathbb{R}^{3+H}$ 로 정의할 때, "입이 열림/닫힘" 같은 single topology change 를 표현하려면:
- Canonical 3D surface: 2D manifold (input)
- Hidden dimension H: 1개 추가 → 가능한 topology: 2개 (open/closed)
- H = 2: 더 복잡한 변화 허용 (multiple parts, self-intersection, etc.)

수학적으로: **codimension** = ambient dim - manifold dim = $(3+H) - 3 = H$. Generic position 에서 codimension 1 manifold 들의 intersection 은 잘 정의됨.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Ambient Space 와 Canonical Manifold

Ambient space: $\mathcal{A} = \mathbb{R}^{3+H}$ where $H \geq 1$

Canonical 4D manifold: $\mathcal{M}^{\text{can}} \subset \mathcal{A}$, 3D surface (codimension 1)

Coordinates: $(\mathbf{x}^{\text{can}}, w^{\text{can}}) \in \mathbb{R}^3 \times \mathbb{R}^H$

### 정의 2.2 — Slicing Surface

시간 $t$ 에 따른 **hyperplane** (slicing surface):

$$\mathcal{S}(t) = \{\,(\mathbf{x}, \mathbf{w}) \in \mathcal{A} : w_1 = w_1(t), \ldots, w_H = w_H(t)\,\}$$

또는 implicit form:

$$\mathcal{S}(t) = \{\,(\mathbf{x}, \mathbf{w}) : \mathbf{w} = \mathbf{w}(t)\,\}$$

where $\mathbf{w}(t): \mathbb{R} \to \mathbb{R}^H$ 는 time-dependent vector.

### 정의 2.3 — Observed Geometry

Observed 3D surface at time $t$:

$$\mathcal{M}(t) = \mathcal{M}^{\text{can}} \cap \mathcal{S}(t)$$

즉, canonical 4D manifold 를 time-dependent slice 로 자른 것.

### 정의 2.4 — Ambient NeRF

기본 NeRF 를 ambient space 로 확장:

$$F_\theta(\mathbf{x}^{\text{amb}}, \mathbf{w}^{\text{amb}}, \mathbf{d}) \to (\sigma, \mathbf{c})$$

- Input: ambient position $(\mathbf{x}^{\text{amb}}, \mathbf{w}^{\text{amb}}) \in \mathbb{R}^{3+H}$
- Output: density $\sigma$, color $\mathbf{c}$

### 정의 2.5 — Forward Query (Rendering)

Given ray in 3D $\mathbf{o} + s\mathbf{d}$, time $t$:

$$C(t) = \int_0^{\infty} T(s) \cdot \sigma(\mathbf{o} + s\mathbf{d}, \mathbf{w}(t), \mathbf{d}) \cdot \mathbf{c}(\ldots) \, ds$$

즉, **ambient $w$ 좌표는 고정** ($= \mathbf{w}(t)$), 3D ray 는 $(x, y, z)$ 방향으로 traverse.

---

## 🔬 정리와 증명

### 정리 2.1 — Ambient Slicing 으로 Topology Change 표현 가능

$H \geq 1$ 차원의 ambient space 와 time-dependent slicing $\mathbf{w}(t)$ 가 주어질 때:

$$\mathcal{M}(t) = \{\,(\mathbf{x}, \mathbf{w}(t)) : (\mathbf{x}, \mathbf{w}(t)) \in \mathcal{M}^{\text{can}}\,\}$$

는 일반적인 경우 서로 다른 topology 를 가질 수 있다.

**증명 스케치**:

Canonical manifold $\mathcal{M}^{\text{can}}$ 를 생각해보면:
- Regular case: $\mathbf{w}$ 축 방향으로 횡단적으로 ("transversally") 다양함
- Slicing 할 때마다 다른 cross-section 을 얻음
- 예: 2D 에서 S-shape curve 를 수직선으로 slice 하면 intersection 개수가 변함

Ambient slicing 는 이를 일반화:
$$\text{Euler characteristic} \quad \chi(\mathcal{M}(t)) = \int_{\mathcal{M}^{\text{can}}} \delta(\mathbf{w} - \mathbf{w}(t))\, d\mathcal{M}$$

$\mathbf{w}(t)$ 가 연속이지만 transversal intersection 의 수가 변할 수 있음 → topology change 가능. $\square$

### 정리 2.2 — H = 2 의 충분성 (Park 2021 empirical finding)

실제 비디오 장면 (사람의 입 열림/닫힘 등) 에서, $H = 2$ 는 대부분의 topology change 를 capture 하기에 충분하다.

**이론적 근거**:
- 각 frame 에서 topology change 는 typically **1개 또는 2개의 event** (입이 열림, 닫힘)
- $H = 2$ 의 slice surface 는 2-parameter family → 충분한 flexibility

**경험적**: $H = 4$ 이상은 marginal improvement, computational overhead 만 증가.

### 따름정리 2.3 — Deformation Field vs Ambient Slicing

Deformation field $D_\psi$ (Nerfies) 를 ambient space 에서 해석하면:

$$D_\psi(\mathbf{x}, t) = \mathbf{x} - \mathbf{x}^{\text{amb}} \big|_{\mathbf{w}=\text{const.}(t)}$$

즉, **특정 $w$ path** 를 따라가는 특수한 경우. HyperNeRF 는 더 일반적 (arbitrary $\mathbf{w}(t)$).

---

## 💻 PyTorch 구현 검증

### 실험 1 — Ambient NeRF 의 기본 구조

```python
import torch
import torch.nn as nn

class AmbientNeRF(nn.Module):
    """Ambient NeRF: F(x, w, d) -> (σ, c)."""
    def __init__(self, dim_pos=3, dim_ambient=2, dim_dir=3, hidden=256):
        super().__init__()
        # Input encoding: x (3) + w (H) + d (3)
        self.mlp_feat = nn.Sequential(
            nn.Linear(dim_pos * 6 + dim_ambient * 4 + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.sigma = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Softplus()
        )
        self.mlp_rgb = nn.Sequential(
            nn.Linear(hidden + dim_dir * 6, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 3),
            nn.Sigmoid(),
        )
    
    def forward(self, x, w, d):
        """
        x: (B, 3) spatial coordinate in 3D
        w: (B, H) ambient coordinate
        d: (B, 3) direction
        """
        # Positional encoding for x
        x_enc = torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1)
        
        # Encoding for w (usually smaller, periodic)
        w_enc = torch.cat([w, torch.sin(2*torch.pi*w), 
                          torch.cos(2*torch.pi*w), 
                          torch.sin(4*torch.pi*w)], dim=-1)
        
        # Direction encoding
        d_enc = torch.cat([d, torch.sin(d), torch.cos(d)], dim=-1)
        
        feat = self.mlp_feat(torch.cat([x_enc, w_enc, d_enc], dim=-1))
        sigma = self.sigma(feat)
        rgb = self.mlp_rgb(torch.cat([feat, d_enc], dim=-1))
        return sigma, rgb

class SlicingFunction(nn.Module):
    """Learned time-dependent slicing: t -> w(t)."""
    def __init__(self, dim_ambient=2, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim_ambient),
        )
    
    def forward(self, t):
        """
        t: (B, 1) time coordinate
        Returns: w (B, H) ambient coordinate
        """
        # Periodic encoding of time
        t_enc = torch.sin(2 * torch.pi * t)
        return self.mlp(t_enc)

class HyperNeRF(nn.Module):
    """Full HyperNeRF: Ambient NeRF + Slicing."""
    def __init__(self, dim_ambient=2):
        super().__init__()
        self.nerf_ambient = AmbientNeRF(dim_ambient=dim_ambient)
        self.slicing = SlicingFunction(dim_ambient=dim_ambient)
    
    def query(self, x, t, d):
        """
        x: (B, 3) observed 3D position
        t: (B, 1) time
        d: (B, 3) ray direction
        """
        w = self.slicing(t)  # (B, H)
        sigma, rgb = self.nerf_ambient(x, w, d)
        return sigma, rgb, w
```

### 실험 2 — Rendering with Ambient Slicing

```python
def render_rays_hypernerf(model, ray_origins, ray_dirs, ts, n_samples=64):
    """
    Forward volume rendering with ambient slicing.
    """
    near, far = 0.1, 10.0
    device = ray_origins.device
    
    z_samples = torch.linspace(near, far, n_samples, device=device)
    z_samples = z_samples.unsqueeze(0) + \
                torch.randn(ray_origins.shape[0], 1, device=device) * 0.01
    
    points = ray_origins.unsqueeze(1) + ray_dirs.unsqueeze(1) * z_samples.unsqueeze(-1)
    # (H*W, n_samples, 3)
    
    B, N, _ = points.shape
    points_flat = points.reshape(-1, 3)
    ts_flat = ts.repeat(1, N).reshape(-1, 1)
    dirs_flat = ray_dirs.repeat(1, N).reshape(-1, 3)
    
    # Query ambient NeRF
    sigma, rgb, w_query = model.query(points_flat, ts_flat, dirs_flat)
    sigma = sigma.reshape(B, N)
    rgb = rgb.reshape(B, N, 3)
    
    # Volume rendering
    delta_z = torch.diff(z_samples, dim=-1, prepend=torch.zeros(B, 1, device=device))
    alpha = 1.0 - torch.exp(-sigma * delta_z)
    
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    T = torch.cat([torch.ones(B, 1, device=device), T[:, :-1]], dim=-1)
    
    weights = T * alpha
    rgb_final = (weights.unsqueeze(-1) * rgb).sum(dim=1)
    
    return rgb_final, weights, w_query

# Training
model = HyperNeRF(dim_ambient=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    for batch in data_loader:
        rays_o, rays_d, ts, gt_rgb = batch
        
        pred_rgb, weights, w = render_rays_hypernerf(model, rays_o, rays_d, ts)
        
        loss_photo = ((pred_rgb - gt_rgb) ** 2).mean()
        
        # Optional: encourage smooth w(t)
        loss_smooth_w = 0.001 * (w[:, 1:] - w[:, :-1]).abs().mean()
        
        loss = loss_photo + loss_smooth_w
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 실험 3 — Topology Change Visualization

```python
def visualize_topology_change(model, times):
    """
    시간에 따른 w(t) 의 변화 시각화.
    """
    ts = torch.from_numpy(times).float().unsqueeze(-1)
    ws = model.slicing(ts).detach().cpu().numpy()
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # w(t) trajectory
    axes[0].plot(ws[:, 0], ws[:, 1], 'b-o', alpha=0.6)
    axes[0].set_xlabel('w_1')
    axes[0].set_ylabel('w_2')
    axes[0].set_title('Slicing trajectory in ambient space')
    axes[0].grid(True)
    
    # w_1, w_2 over time
    axes[1].plot(times, ws[:, 0], 'r-', label='w_1(t)', linewidth=2)
    axes[1].plot(times, ws[:, 1], 'b-', label='w_2(t)', linewidth=2)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('w value')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('topology_change.png', dpi=100)
    print("Saved to topology_change.png")

# Test
times = torch.linspace(0, 1, 50)
model = HyperNeRF(dim_ambient=2)
visualize_topology_change(model, times)
```

---

## 🔗 실전 활용

### 1. 입 열림/닫힘 (Mouth Opening)

입 안쪽 surface:
- Ambient space: 구 표면 (입의 outer) + hidden dimension (입 안 깊이)
- $w(t) = [\cos(2\pi t), \sin(2\pi t)]$ — circular trajectory
- $t = 0$: 입 닫힘, $t = 0.5$: 입 열림
- $\mathcal{M}(t)$ 의 topology: closed curve → open curve 로 변화

### 2. 손가락 교차 (Hand Occlusion)

두 손가락:
- Ambient dim: 2 개 필요
- $w(t)$ 가 적절히 이동하면, 서로 교차하는 configuration 표현 가능
- 실제로는 Nerfies style deformation + occlusion inpainting 으로도 근사 가능

### 3. 포인트 클라우드 추출

주어진 시간 $t$ 에 대해, density threshold 이상인 점들:

```python
def extract_point_cloud(model, t, density_threshold=0.5, grid_size=32):
    """Ambient NeRF 에서 point cloud 추출."""
    x_grid = torch.linspace(-1, 1, grid_size, device=device)
    X, Y, Z = torch.meshgrid(x_grid, x_grid, x_grid, indexing='ij')
    points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    
    t_batch = torch.full((points.shape[0], 1), t.item(), device=device)
    d_dummy = torch.zeros_like(points)  # direction not used for density
    
    sigma, _, _ = model.query(points, t_batch, d_dummy)
    
    valid = (sigma.squeeze() > density_threshold).cpu().numpy()
    pcd = points[valid].cpu().numpy()
    
    return pcd
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Ambient dim H = 2 고정 | Topology change 1~2개만 (복잡한 경우 H 증가) |
| Canonical manifold 단일 | Multiple part object 는 어려움 (segment-wise application) |
| Slicing transversal | Tangential intersection 불안정 (regularization 필요) |
| Smooth w(t) | Abrupt topology (진짜 occlusion) 불가 (inpainting 필요) |
| Multi-view consistency | Monocular 약함 (camera+deformation jointly optimize) |

---

## 📌 핵심 정리

$$\boxed{\mathcal{M}(t) = \mathcal{M}^{\text{can}} \cap \{\mathbf{w} = \mathbf{w}(t)\}}$$

| 개념 | 정의 | 이점 |
|------|------|------|
| Ambient space | $\mathbb{R}^{3+H}$ | topology change 가능 |
| Slicing surface | $\mathbf{w}(t)$ hyperplane | time-dependent geometry |
| Observed surface | intersection | 다양한 topology 형태 |
| H = 2 | $\mathbb{R}^5$ ambient | 실제 장면 대부분 sufficient |

**핵심**: Canonical 4D manifold + time-dependent slicing = topology-changing 장면의 효율적 표현.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Ambient dimension $H = 1$ 이면 왜 일반적인 topology change 를 표현할 수 없는가? 예를 들어 입이 열림/닫힘은 $H = 1$ 로 충분한가?

<details>
<summary>해설</summary>

$H = 1$ 의 경우:
- Slicing surface: $w = w(t)$ (1D 슬라이스)
- Canonical 3D manifold 를 수평선으로 자르는 것
- 교점의 연결성: 대부분 변하지 않음 (generic case 에서 transversal slice 의 연결 성분 수는 stable)

**입 열림/닫힘**:
- Topologically: closed → open 변화
- 이를 표현하려면 **동시에 여러 부분이 연결/분리** 되어야 함
- $H = 1$ では 불가능 (slice 가 1D 이므로 하나의 연속 곡선만 가능)

$H = 2$ 로 증가하면:
- Slicing surface: hyperplane (2D codimension 1)
- 더 복잡한 교점 geometry 가능
- 입이 열릴 때 음성 영역에서 음성 표면으로 변화 가능

따라서 **H = 1 은 부족, H ≥ 2 필요**. $\square$

</details>

**문제 2** (심화): Slicing function $\mathbf{w}(t)$ 를 학습할 때, time periodicity 를 강제하면 어떤 문제가 생기는가? 예를 들어 같은 동작을 반복하는 경우 (talking head 의 반복 입 움직임).

<details>
<summary>해설</summary>

Periodic slicing:
- 예: $\mathbf{w}(t) = [\cos(2\pi \omega t), \sin(2\pi \omega t)]$ with learned frequency $\omega$
- 장점: 반복 동작에서 자연스러운 cycle, extrapolation 가능
- 단점: **하나의 주파수** 로 제한 → 다양한 속도의 동작 불가

**해결책**:
- Learnable Fourier basis: $\mathbf{w}(t) = \sum_k a_k \sin(k\omega t) + b_k \cos(k\omega t)$
- 또는 quasi-periodic: periodicity constraint 제거, instead smoothness loss 추가
- Data 가 충분하면 **parametric-free** slicing function (fully learned MLP)

비주기 동작 (예: 입 다양한 표정) 은 periodic assumption 이 harmful 할 수 있음. $\square$

</details>

**문제 3** (논문 비평): HyperNeRF 는 ambient space 에서 **균일한 codimension** 을 가정한다. 실제 장면에서 어떤 부분은 simple topology (팔) 이고 어떤 부분은 complex topology (입) 일 수 있다. 이를 어떻게 처리할까?

<details>
<summary>해설</summary>

현재 HyperNeRF 제약:
- 전체 scene 이 같은 ambient space 사용
- H = 2 는 타협 (모든 부분에 충분한가?)

**가능한 해결책**:

1. **Part-wise ambient dimensions**:
   - Face region: $H = 2$ (복잡한 입 변화)
   - Arm region: $H = 0$ (simple rigid motion, Nerfies sufficient)
   - Learned segmentation 으로 region 분할

2. **Hierarchical ambient**:
   - Global: $H = 1$ (전체 posture)
   - Local: $H = 1$ per part (부분별 deformation)
   - Concatenate 해서 효율적 표현

3. **Adaptive codimension** (미래 연구):
   - Network 가 각 point 에서 필요한 "ambient dimension" 예측
   - 자동으로 complex region 에만 high dim 할당

**Park 2021 (HyperNeRF)**: 고정 H 사용, empirical 하게 충분함을 보임. 더 정교한 방법은 open problem. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-deformable-nerf.md) | [📚 README](../README.md) | [다음 ▶](./03-4d-gaussian-splatting.md)

</div>
