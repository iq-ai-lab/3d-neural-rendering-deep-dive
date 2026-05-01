# 01. Dynamic NeRF — Canonical Space 와 Deformation Field

## 🎯 핵심 질문

- 정적 NeRF $F_\theta(\mathbf{x}, \mathbf{d})$ 를 시간 의존 장면으로 확장할 때, 왜 naive $F_\theta(\mathbf{x}, t, \mathbf{d})$ 보다 **canonical space** + **deformation field** 가 우수한가?
- Deformation field $D_\psi(\mathbf{x}, t) \to \Delta\mathbf{x}$ 는 어떻게 정의되고, 왜 **SE(3) group** 을 사용하는가?
- Nerfies (Park 2021) 의 elastic regularizer 는 무엇이고, 왜 unrealistic deformation 을 방지하는가?
- D-NeRF (Pumarola 2021) 의 변형은 무엇인가? 그리고 실전에서 어떤 차이가 발생하는가?

---

## 🔍 왜 Dynamic NeRF 가 필요한가

4D 장면 (3D space + time) 을 표현하는 가장 직관적인 방법은 NeRF 를 시간으로 확장하는 것입니다:

$$F_\theta(\mathbf{x}, t, \mathbf{d}) \to (\sigma, \mathbf{c})$$

하지만 이 **naive approach** 는 다음의 문제를 갖습니다:

1. **Temporal redundancy**: 시간 변화가 작으면 (slow motion), 같은 위치의 neural feature 가 highly correlated → 네트워크가 각 time step 마다 별도의 feature 를 학습해야 함 (비효율)
2. **Poor generalization**: 학습 시간 범위 밖에서 extrapolation 이 매우 약함
3. **High memory & computation**: time dimension 추가로 network capacity 증가

**해법**: Canonical space 와 deformation field 의 분리:
- **Canonical NeRF** $F_\theta(\mathbf{x}^{\text{can}}, \mathbf{d})$ — 변하지 않는 장면의 "참" geometry/appearance
- **Deformation field** $D_\psi(\mathbf{x}, t)$ — 시간마다 공간이 어떻게 변형되는지

Query 할 때: $\mathbf{x}^{\text{can}} = \mathbf{x} + D_\psi(\mathbf{x}, t)$

---

## 📐 수학적 선행 조건

- **NeRF & Radiance Field**: Volume rendering integral, positional encoding, MLP
- **Rigid body dynamics**: SE(3) group, rotation matrix $\mathbf{R}$, translation $\mathbf{t}$
- **Regularization**: Elastic energy, smoothness loss
- 선택: Lie group theory (SE(3) 의 manifold structure)

---

## 📖 직관적 이해

### Canonical Space 의 역할

정적 NeRF 는 모든 물체의 상태를 기억합니다. 하지만 사람이 팔을 드는 동작에서:
- 각 frame 마다 팔의 위치가 다름
- 그러나 **팔의 기하학적 구조 (canonical geometry)** 는 변하지 않음
- 단지 **어디에 있는지 (위치, 자세)** 만 변함

이를 분리하면:
1. **Canonical NeRF**: 팔의 "본질적" 모양 (색, 세세한 texture)
2. **Deformation field**: 각 frame 에서 팔의 점들이 어디로 움직였는지

이렇게 하면 **canonical feature 를 재사용** 할 수 있어 학습이 효율적.

### Deformation Field 의 형태

가장 단순한 형태는 **per-point displacement**:

$$\mathbf{x}^{\text{can}} = \mathbf{x} + D_\psi(\mathbf{x}, t)$$

여기서 $D_\psi: \mathbb{R}^3 \times \mathbb{R} \to \mathbb{R}^3$ 는 작은 MLP.

더 정교한 형태는 **SE(3) deformation** (Nerfies):

$$\mathbf{x}^{\text{can}} = \mathbf{R}(t) \mathbf{x} + \mathbf{t}(t) + D_\psi^{\text{elastic}}(\mathbf{x}, t)$$

- $\mathbf{R}(t) \in SO(3)$: rotation (직교 행렬)
- $\mathbf{t}(t) \in \mathbb{R}^3$: translation
- $D_\psi^{\text{elastic}}$: non-rigid deformation

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Canonical NeRF

$$F_\theta(\mathbf{x}^{\text{can}}, \mathbf{d}) = (\sigma^{\text{can}}(\mathbf{x}^{\text{can}}), \mathbf{c}^{\text{can}}(\mathbf{x}^{\text{can}}, \mathbf{d}))$$

- Input: canonical coordinate $\mathbf{x}^{\text{can}} \in \mathbb{R}^3$, direction $\mathbf{d} \in \mathbb{S}^2$
- Output: density $\sigma \geq 0$, RGB color $\mathbf{c} \in [0, 1]^3$
- **기본 가정**: canonical space 에서는 appearance/geometry 가 constant

### 정의 1.2 — Deformation Field

두 형태로 정의 가능:

**형태 1 (D-NeRF)**: 순수 displacement
$$\mathbf{x}^{\text{can}} = \mathbf{x} + D_\psi(\mathbf{x}, t)$$
where $D_\psi: \mathbb{R}^3 \times \mathbb{R} \to \mathbb{R}^3$

**형태 2 (Nerfies)**: SE(3) + elastic
$$\mathbf{x}^{\text{can}} = \mathbf{R}(t)[\mathbf{x} + D_\psi^{\text{elastic}}(\mathbf{x}, t)] + \mathbf{t}(t)$$

### 정의 1.3 — Forward Rendering

주어진 ray $\mathbf{o} + s\mathbf{d}$ ($s \geq 0$), 시간 $t$:

$$C(t) = \int_0^{\infty} T(s) \cdot \sigma^{\text{can}}(\mathbf{x}^{\text{can}}(s, t)) \cdot \mathbf{c}^{\text{can}}(\mathbf{x}^{\text{can}}(s, t), \mathbf{d})\, ds$$

where:
- $\mathbf{x}^{\text{can}}(s, t) = (\mathbf{o} + s\mathbf{d}) + D_\psi(\mathbf{o} + s\mathbf{d}, t)$
- $T(s) = \exp\left(-\int_0^s \sigma^{\text{can}}(\mathbf{x}^{\text{can}}(u, t)) du\right)$

### 정의 1.4 — Elastic Regularizer (Nerfies)

Deformation 의 "합리성" 을 강제하기 위해:

$$\mathcal{L}_{\text{elastic}} = \int_{\Omega} \left\|\nabla^2 D_\psi(\mathbf{x}, t)\right\|_F^2 d\mathbf{x}\, dt$$

where $\|\cdot\|_F$ 는 Frobenius norm. 이것은:
- **과도한 굽힘/늘어남** 을 방지 (너무 국소적인 변형)
- **이상적인 deformation** (smooth, 생리학적으로 가능한)을 유도

---

## 🔬 정리와 증명

### 정리 1.1 — Deformation Field 의 Jacobian 과 Volume Preservation

Deformation field $\mathbf{x}^{\text{can}} = \mathbf{x} + D_\psi(\mathbf{x}, t)$ 에 대해, Jacobian:

$$\mathbf{J}(\mathbf{x}, t) = \frac{\partial \mathbf{x}^{\text{can}}}{\partial \mathbf{x}} = \mathbf{I} + \nabla D_\psi(\mathbf{x}, t)$$

만약 $\|\nabla D_\psi\|_F$ 가 작으면 (elastic regularizer 로 유도),

$$\det(\mathbf{J}) \approx 1 + \text{tr}(\nabla D_\psi) + O(\|\nabla D_\psi\|^2)$$

**의미**: small displacement 에서 volume 이 거의 보존됨 → **생리학적으로 타당한 변형**.

### 정리 1.2 — SE(3) Deformation 의 Rigidity

SE(3) rigid deformation (Nerfies):

$$\mathbf{x}^{\text{can}} = \mathbf{R}(t) \mathbf{x} + \mathbf{t}(t)$$

에 대해, 임의의 두 점 $\mathbf{x}_1, \mathbf{x}_2$ 간 거리는 보존:

$$\|\mathbf{x}^{\text{can}}_1 - \mathbf{x}^{\text{can}}_2\| = \|\mathbf{R}(t)(\mathbf{x}_1 - \mathbf{x}_2)\| = \|\mathbf{x}_1 - \mathbf{x}_2\|$$

($\mathbf{R}$ orthogonal 이므로).

**따름정리**: SE(3) 만으로는 non-rigid change (재질 늘어남 등) 표현 불가 → elastic $D_\psi^{\text{elastic}}$ 추가 필요.

### 정리 1.3 — Deformation Field 의 추정 가능성 (Identifiability)

정적 scene 에서 NeRF 는 inherent ambiguity 를 갖습니다 (뒤에 작은 강도 vs 앞에 큰 강도). Dynamic scene 에서도:

$$F_\theta(\mathbf{x}^{\text{can}}, \mathbf{d}), \quad D_\psi(\mathbf{x}, t)$$

는 jointly non-unique. 그러나 **다중 view + temporal consistency** 가 주어지면:

- **Canonical geometry**: photometric loss 로 $F_\theta$ 결정
- **Deformation**: temporal consistency + multi-view epipolar constraint 로 $D_\psi$ 결정

---

## 💻 PyTorch 구현 검증

### 실험 1 — Deformation Field MLP 구성

```python
import torch
import torch.nn as nn

class CanonicalNeRF(nn.Module):
    """정적 NeRF."""
    def __init__(self, dim_pos=3, dim_dir=3, hidden=256):
        super().__init__()
        self.mlp_feat = nn.Sequential(
            nn.Linear(dim_pos * 6 + 3, hidden),  # positional encoding x 3
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.sigma = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
        self.mlp_rgb = nn.Sequential(
            nn.Linear(hidden + dim_dir * 6, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 3),
            nn.Sigmoid(),
        )
    
    def forward(self, x_can, d):
        """
        x_can: (B, 3) canonical position
        d:     (B, 3) direction
        """
        # Positional encoding
        x_enc = torch.cat([x_can, torch.sin(x_can), torch.cos(x_can)], dim=-1)
        d_enc = torch.cat([d, torch.sin(d), torch.cos(d)], dim=-1)
        
        feat = self.mlp_feat(x_enc)
        sigma = self.sigma(feat)
        rgb = self.mlp_rgb(torch.cat([feat, d_enc], dim=-1))
        return sigma, rgb

class DeformationField(nn.Module):
    """Deformation field: (x, t) -> Δx."""
    def __init__(self, dim_pos=3, dim_time=1, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_pos * 6 + dim_time, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim_pos),
        )
    
    def forward(self, x, t):
        """
        x:  (B, 3) observed position
        t:  (B, 1) time
        Returns: Δx in (B, 3)
        """
        # Encode position & time
        x_enc = torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1)
        t_enc = torch.sin(2 * torch.pi * t)  # periodic time encoding
        
        inp = torch.cat([x_enc, t_enc], dim=-1)
        delta_x = self.mlp(inp)
        return delta_x

class DynamicNeRF(nn.Module):
    """Canonical NeRF + Deformation Field."""
    def __init__(self):
        super().__init__()
        self.nerf_canonical = CanonicalNeRF()
        self.deform = DeformationField()
    
    def query(self, x, t, d):
        """
        x: (B, 3) observed 3D position
        t: (B, 1) time
        d: (B, 3) ray direction
        """
        # Deformation: x -> x_can
        delta_x = self.deform(x, t)
        x_can = x + delta_x  # forward mapping
        
        # Canonical NeRF query
        sigma, rgb = self.nerf_canonical(x_can, d)
        return sigma, rgb, x_can, delta_x
```

### 실험 2 — Forward Rendering + Loss

```python
def render_rays(model, ray_origins, ray_dirs, ts, n_samples=64):
    """
    ray_origins: (H*W, 3)
    ray_dirs:    (H*W, 3) normalized
    ts:          (H*W, 1) time for each ray
    """
    near, far = 0.1, 10.0
    device = ray_origins.device
    
    # Sample points along rays
    z_samples = torch.linspace(near, far, n_samples, device=device)  # (n_samples,)
    z_samples = z_samples.unsqueeze(0) + \
                torch.randn(ray_origins.shape[0], 1, device=device) * 0.01
    # (H*W, n_samples)
    
    points = ray_origins.unsqueeze(1) + ray_dirs.unsqueeze(1) * z_samples.unsqueeze(-1)
    # (H*W, n_samples, 3)
    
    # Flatten for batch processing
    B, N, _ = points.shape
    points_flat = points.reshape(-1, 3)                      # (B*N, 3)
    ts_flat = ts.repeat(1, N).reshape(-1, 1)                 # (B*N, 1)
    dirs_flat = ray_dirs.repeat(1, N).reshape(-1, 3)         # (B*N, 3)
    
    sigma, rgb, _, _ = model.query(points_flat, ts_flat, dirs_flat)
    sigma = sigma.reshape(B, N)  # (H*W, n_samples)
    rgb = rgb.reshape(B, N, 3)   # (H*W, n_samples, 3)
    
    # Volume rendering
    delta_z = torch.diff(z_samples, dim=-1, prepend=torch.zeros(B, 1, device=device))
    alpha = 1.0 - torch.exp(-sigma * delta_z)  # (B, N)
    
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    T = torch.cat([torch.ones(B, 1, device=device), T[:, :-1]], dim=-1)
    
    weights = T * alpha  # (B, N)
    
    rgb_final = (weights.unsqueeze(-1) * rgb).sum(dim=1)  # (B, 3)
    
    return rgb_final, weights

# Training loop (simplified)
model = DynamicNeRF()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    for batch in data_loader:
        rays_o, rays_d, ts, gt_rgb = batch  # gt_rgb from video
        
        pred_rgb, weights = render_rays(model, rays_o, rays_d, ts)
        
        # Photometric loss
        loss_photo = ((pred_rgb - gt_rgb) ** 2).mean()
        
        # Elastic regularizer (lazy approximation)
        elastic_loss = 0.01 * sum(
            p.abs().mean() 
            for name, p in model.deform.named_parameters()
            if 'bias' not in name
        )
        
        loss = loss_photo + elastic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 실험 3 — Deformation Jacobian 시각화

```python
def analyze_jacobian(model, x, t):
    """Deformation field 의 Jacobian 계산."""
    x = x.clone().detach().requires_grad_(True)
    
    delta_x = model.deform(x, t)
    
    # Compute Jacobian: d(delta_x) / dx
    J = torch.zeros(3, 3, device=x.device)
    for i in range(3):
        grad = torch.autograd.grad(
            delta_x[0, i], x, 
            create_graph=True, retain_graph=True
        )[0]
        J[i] = grad[0]
    
    # Analysis
    I = torch.eye(3, device=x.device)
    F = I + J  # deformation gradient
    det_F = torch.det(F)
    tr_F = torch.trace(F)
    
    print(f"Deformation gradient F:\n{F}")
    print(f"det(F) = {det_F:.4f} (ideally ≈ 1.0 for volume preservation)")
    print(f"tr(F) = {tr_F:.4f}")
    
    return J, det_F

# Test
x_test = torch.randn(1, 3)
t_test = torch.tensor([[0.5]])
J, det_F = analyze_jacobian(model, x_test, t_test)
```

---

## 🔗 실전 활용

### 1. Multi-View Video 재구성

다양한 카메라 각도에서 같은 시간의 영상:
- 각 view 의 photometric loss 합산
- Deformation field 는 shared (physics prior)
- 결과: consistency 한 4D geometry

### 2. Temporal Consistency Loss

인접 frame 간의 deformation smoothness:

$$\mathcal{L}_{\text{temporal}} = \int_{\Omega} \left\|\frac{\partial D_\psi}{\partial t}\right\|^2 d\Omega$$

급격한 움직임 방지, 자연스러운 motion.

### 3. 다른 정규화 항들

| 항 | 목적 | 가중치 |
|----|------|--------|
| Photometric | RGB 일치 | 1.0 |
| Elastic | smooth deformation | 0.01 |
| Temporal | smooth motion | 0.001 |
| Sparsity | 불필요한 deformation 삭제 | 0.0001 |

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Canonical space 고정 | Topology change 처리 불가 (HyperNeRF 에서 해결) |
| Small deformation | 큰 움직임 시 문제 (여러 segment 로 분리, 또는 multi-resolution) |
| Smooth motion | 급격한 occlusion 불가 (warping + inpainting) |
| Multi-view 가정 | Monocular 약함 (camera estimation 필요) |
| Continuous time | discrete frame 만 학습 (interpolation 약함) |

---

## 📌 핵심 정리

$$\boxed{\mathbf{x}^{\text{can}} = \mathbf{x} + D_\psi(\mathbf{x}, t)}$$

| 개념 | 역할 | 이점 |
|------|------|------|
| Canonical NeRF | 불변 geometry/appearance | 재사용, 빠른 수렴 |
| Deformation field | 시간 종속 변형 | 효율적, interpretable |
| Elastic regularizer | deformation smoothness | 합리적 motion |
| SE(3) component | rigid 움직임 | global transformation capture |

**핵심**: Canonical + deformation 분리 = dynamic scene 을 **저차원 representation** 으로 효율적 표현.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Deformation field MLP 를 sigmoid 또는 tanh 로 제한하지 않은 이유는? 만약 unbounded deformation 을 허용하면 어떤 문제가 발생하는가?

<details>
<summary>해설</summary>

Unbounded deformation 은:
1. **Mode collapse** — optimizer 가 매우 큰 $\Delta\mathbf{x}$ 를 할당하여 occlusion 을 "해결"
2. **Blurry canonical** — canonical NeRF 가 희미해져 photometric loss 감소
3. **Poor temporal consistency** — frame 마다 다른 변형으로 motion 이 노이즈스러움

**해결**: elastic regularizer $\|\nabla^2 D_\psi\|^2$ 가 큰 변형을 직접 페널티. unbounded 허용, but cost-controlled. $\square$

</details>

**문제 2** (심화): Nerfies 에서 SE(3) + elastic 을 사용하는 대신, 순수 displacement $\mathbf{x}^{\text{can}} = \mathbf{x} + D_\psi$ 만 사용하면 어떤 차이가 발생하는가? 언제 SE(3) 가 필수인가?

<details>
<summary>해설</summary>

순수 displacement 에서:
- Global rotation/translation 도 $D_\psi$ 가 학습해야 함 → 더 큰 네트워크 필요
- Elastic regularizer 가 global motion 을 제약 → 큰 rigid 움직임 시 conflict
- $D_\psi$ 의 degrees of freedom 낭비

SE(3) 명시적 사용:
- Global 6-DOF motion (3 rotation + 3 translation) 을 parametrically 분리
- $D_\psi^{\text{elastic}}$ 는 **non-rigid perturbation** 만 학습
- 더 효율적 + regularizer conflict 없음

**언제 필수**: 
- 팔/다리 움직임: rigid segment + elastic joint deformation 필요
- 카메라 moving 장면: global camera motion 을 SE(3) 로 쉽게 인수분해

**D-NeRF vs Nerfies 트레이드오프**:
- D-NeRF: 단순, 하지만 큰 motion 시 불안정
- Nerfies: 복잡, 하지만 더 정확한 motion capture

$\square$

</details>

**문제 3** (논문 비평): Canonical space 의 정의가 ambiguous 하다. 예를 들어 사람 영상에서 canonical time $t_0$ 를 어떻게 선택하는가? 다른 $t_0$ 를 선택하면 deformation field 가 어떻게 달라지는가?

<details>
<summary>해설</summary>

Canonical time 선택:
- 보통 **첫 frame** ($t_0 = 0$) 또는 **mean pose** (모든 frame 의 평균 skeleton) 선택
- Mean pose 가 더 robust (outlier frame 에 덜 민감)

다른 $t_0$ 선택 시:
- Deformation field 는 변함 (당연)
- 하지만 **predicted geometry/appearance** 는 동일 — only reparametrization
- 수학적으로 같은 scene, 다른 coordinate system

**해결책**: 
- Canonical space 를 **learned mean pose** 로 정의
- Training 중 모든 frame deformation 의 "average" 가 0 이 되도록 constraint 추가

$$\mathbb{E}_t[D_\psi(\mathbf{x}, t)] \approx 0$$

이렇게 하면 canonical space 가 유일하게 정의됨. $\square$

</details>

---

<div align="center">

[◀ 이전](../ch4-gaussian-splatting/06-3dgs-training-comparison.md) | [📚 README](../README.md) | [다음 ▶](./02-hypernerf.md)

</div>
