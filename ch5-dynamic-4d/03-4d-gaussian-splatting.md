# 03. 4D Gaussian Splatting — Polynomial Trajectory 와 Temporal Decomposition

## 🎯 핵심 질문

- 3DGS (정적) 를 4D (동적) 로 확장할 때, 왜 각 Gaussian 에 **polynomial trajectory** 를 주는가?
- Yang 2024 의 per-Gaussian trajectory $\mu(t) = \mu_0 + \sum_k a_k t^k$ 와 Wu 2024 의 **HexPlane decomposition** 는 무엇이 다른가?
- 두 방법의 trade-off 는? (long horizon stability, memory efficiency, expressivity)
- 실전에서 어떤 상황에 어떤 방법을 선택해야 하는가?

---

## 🔍 왜 4D Gaussian Splatting 인가

NeRF-based 4D (Ch1-02) 의 단점:
1. **Slow rendering**: volume integral + time sampling → 초당 1-2 frame
2. **Poor generalization**: 학습하지 않은 time 에서 blurry (temporal extrapolation weak)
3. **High memory**: MLP + time encoding 이 커짐

**Gaussian Splatting 의 장점** (Ch4-06):
- Explicit representation (Gaussian mean, covariance, color)
- Fast rasterization (real-time 30+ fps)
- Per-Gaussian 학습 → local control

**4D 확장**: 각 Gaussian 에 **time parameter** 추가
- Mean position: time-dependent
- Covariance: constant (또는 time-dependent 버전)
- Color/opacity: constant (또는 appearance table)

---

## 📐 수학적 선행 조건

- **Ch4-01 ~ 06**: 3DGS basics (Gaussian primitives, splatting, optimization)
- **Ch1**: Deformation field 직관
- **Polynomial basis**: 저차 interpolation
- **Matrix decomposition**: Tucker/HexPlane 구조
- Calculus: 시간에 대한 미분 (trajectory velocity)

---

## 📖 직관적 이해

### Approach 1: Per-Gaussian Polynomial Trajectory (Yang 2024)

각 Gaussian 의 center 를 시간의 polynomial 로 표현:

$$\boldsymbol{\mu}_i(t) = \boldsymbol{\mu}_{i,0} + \sum_{k=1}^{K} \boldsymbol{a}_{i,k} t^k$$

- $K = 1$: linear (constant velocity)
- $K = 2$: quadratic (constant acceleration)
- $K = 3, 4$: higher order (complex motion)

**장점**:
- 명시적, 해석 가능 (velocity = 1st derivative)
- Per-Gaussian 독립 → 매우 유연
- Long horizon 에서도 extrapolate 가능

**단점**:
- High degree polynomial: ringing artifact 위험
- 각 Gaussian 마다 coefficient 저장 → memory 증가
- 매우 복잡한 motion 은 여전히 힘듦

### Approach 2: HexPlane Decomposition (Wu 2024 — 4D-GS)

4D space (3D spatial × 1D time) 를 **6개의 2D plane** 으로 분해:

$$\text{4D = 3 spatial dimensions + 1 temporal dimension}$$

각 axis pair 에 대해 **2D plane matrix**:
- $P_{xy}$: x-y plane (z, t 축 고정)
- $P_{xz}$: x-z plane
- $P_{xt}$: x-time plane
- $P_{yz}$, $P_{yt}$, $P_{zt}$

어떤 point $(x, y, z, t)$ 의 feature 는:

$$F(x, y, z, t) = \sum_{\text{6 planes}} w_i \cdot \text{bilinear\_interpolate}(\text{plane}_i, \text{coords})$$

**장점**:
- Memory efficient (6 × 2D vs 1 × 4D full tensor)
- 균형잡힌 spatial-temporal 표현
- Fast lookup (bilinear interpolation)

**단점**:
- 고정된 decomposition structure → 일부 motion 은 과소 표현
- Long extrapolation 약함
- 낮은 rank assumption

---

## ✏️ 엄밀한 정의

### 정의 3.1 — 3DGS Gaussian Primitive

$$G_i(\mathbf{x}) = \alpha_i \cdot \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) \cdot \mathbf{c}_i$$

- $\boldsymbol{\mu}_i \in \mathbb{R}^3$: mean position
- $\boldsymbol{\Sigma}_i = \mathbf{R}_i \mathbf{L}_i^2 \mathbf{R}_i^T$: covariance (rank-3)
- $\mathbf{c}_i \in [0,1]^3$: RGB color
- $\alpha_i \in [0,1]$: opacity

### 정의 3.2 — Polynomial Trajectory (Yang 2024)

$$\boldsymbol{\mu}_i(t) = \boldsymbol{\mu}_{i,0} + \sum_{k=1}^{K} \boldsymbol{a}_{i,k} t^k \quad \text{(polynomial of degree } K \text{)}$$

Trajectory 의 velocity (1st derivative):

$$\mathbf{v}_i(t) = \frac{d\boldsymbol{\mu}_i}{dt} = \sum_{k=1}^{K} k \cdot \boldsymbol{a}_{i,k} t^{k-1}$$

Learnable parameters: $\{\boldsymbol{\mu}_{i,0}, \boldsymbol{a}_{i,1}, \ldots, \boldsymbol{a}_{i,K}\}$ per Gaussian.

### 정의 3.3 — 4D Gaussian at Time $t$

Polynomial trajectory 를 적용한 시간종속 Gaussian:

$$G_i(\mathbf{x}, t) = \alpha_i \cdot \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_i(t), \boldsymbol{\Sigma}_i) \cdot \mathbf{c}_i(t)$$

where:
- Mean is time-dependent (trajectory)
- Covariance typically constant (또는 anisotropic time-dependent, rare)
- Color 추가 module (SH-based or learned appearance)

### 정의 3.4 — HexPlane Feature Decomposition

4D space 를 6개 2D plane 으로 분해:

$$\mathbf{f}_{\text{hex}}(\mathbf{x}, t) = \sum_{i=1}^{6} w_i \cdot \phi_i(P_i; \text{coords}_i)$$

where:
- $P_i$: $i$-th plane matrix (stored as image-like tensor)
- $\text{coords}_i$: 2D coordinates on plane $i$ (bilinear interpolation)
- $w_i$: plane weight

**6 planes**:
1. XY-plane (z, t constant)
2. XZ-plane (y, t constant)
3. XT-plane (y, z constant)
4. YZ-plane (x, t constant)
5. YT-plane (x, z constant)
6. ZT-plane (x, y constant)

---

## 🔬 정리와 증명

### 정리 3.1 — Polynomial Degree 와 Approximation Error

Smooth trajectory $\boldsymbol{\mu}(t) \in C^{K+1}[0, T]$ 에 대해, degree-$K$ polynomial interpolant $\hat{\boldsymbol{\mu}}_K(t)$ 의 error:

$$\max_{t \in [0, T]} \|\boldsymbol{\mu}(t) - \hat{\boldsymbol{\mu}}_K(t)\| \lesssim C_K \cdot T^{K+1} \cdot \|\boldsymbol{\mu}^{(K+1)}\|_{\infty}$$

**의미**:
- Degree 높을수록 (K 크면) approximation 정확
- But variance 증가 → overfitting 위험
- Typical 사용: $K = 2$ (quadratic) 로 충분

### 정리 3.2 — HexPlane Rank Lower Bound

HexPlane decomposition 의 expressiveness 를 분석하면, rank of underlying 4D tensor:

$$\text{rank}(\text{4D tensor}) \leq \text{width}(\text{2D planes})$$

예: 각 2D plane 이 $64 \times 64$ 이면, 최대 rank-64.

**따름정리**: 매우 high-rank motion (e.g., independent rotation of all 3 axes) 은 underrepresented 될 수 있음.

### 정리 3.3 — Memory Complexity 비교

| 방법 | Parameters | Memory per Gaussian |
|------|-----------|-------------------|
| Polynomial (K=2) | $3 + 3K = 9$ per $\boldsymbol{\mu}$ | $O(1)$ (추가 9 scalars) |
| HexPlane | 6 × (width × height) global | $O(\text{width}^2)$ total |

N Gaussians, resolution R 일 때:
- Polynomial: $\approx 9N$ parameters
- HexPlane: $\approx 6 R^2$ parameters (N 무관, global model)

**결론**: HexPlane 은 $N$ 이 매우 크지 않으면 memory efficient; polynomial 은 dense Gaussian cloud 에서 competitive.

---

## 💻 PyTorch 구현 검증

### 실험 1 — Polynomial Trajectory MLP 및 Gaussian Splatting

```python
import torch
import torch.nn as nn
import math

class PolynomialTrajectory(nn.Module):
    """Per-Gaussian polynomial trajectory."""
    def __init__(self, num_gaussians=10000, degree=2, dim=3):
        super().__init__()
        self.degree = degree
        # Coefficients: mu_0 + a_1*t + a_2*t^2 + ...
        self.mu_0 = nn.Parameter(torch.randn(num_gaussians, dim) * 0.1)
        self.a = nn.ParameterList([
            nn.Parameter(torch.randn(num_gaussians, dim) * 0.01)
            for _ in range(degree)
        ])
    
    def forward(self, t):
        """
        t: (B,) time values in [0, 1]
        Returns: (B, num_gaussians, 3) positions
        """
        B = t.shape[0]
        mu = self.mu_0.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)
        
        t_power = t.unsqueeze(-1)  # (B, 1)
        for k, a_k in enumerate(self.a, start=1):
            mu = mu + a_k.unsqueeze(0) * t_power ** k
        
        return mu
    
    def velocity(self, t):
        """Return velocity (d mu / dt)."""
        B = t.shape[0]
        v = torch.zeros(B, self.a[0].shape[0], 3, device=t.device)
        
        t_power = torch.ones(B, 1, device=t.device)
        for k, a_k in enumerate(self.a, start=1):
            v = v + k * a_k.unsqueeze(0) * t_power
            t_power = t_power * t.unsqueeze(-1)
        
        return v

class GaussianSplattingTrajectory(nn.Module):
    """4D GS with polynomial trajectory."""
    def __init__(self, num_gaussians=10000):
        super().__init__()
        
        # Trajectory
        self.trajectory = PolynomialTrajectory(num_gaussians, degree=2)
        
        # Static properties
        self.color = nn.Parameter(torch.ones(num_gaussians, 3) * 0.5)  # RGB
        self.opacity = nn.Parameter(torch.ones(num_gaussians, 1))      # alpha
        
        # Covariance (static, as in 3DGS)
        self.log_scale = nn.Parameter(torch.zeros(num_gaussians, 3))
        self.rotation = nn.Parameter(torch.zeros(num_gaussians, 4))    # quaternion
        
        self.num_gaussians = num_gaussians
    
    def get_covariance(self):
        """Convert log_scale + quaternion to covariance matrix."""
        # Simplified: diagonal covariance
        scale = torch.exp(self.log_scale)  # (N, 3)
        # In practice, use quaternion to full 3x3 matrix
        return scale
    
    def forward(self, t):
        """
        t: (B,) time
        Returns: positions (B, N, 3), colors (N, 3), opacities (N, 1), scales (N, 3)
        """
        positions = self.trajectory(t)
        colors = torch.sigmoid(self.color)
        opacities = torch.sigmoid(self.opacity)
        scales = torch.exp(self.log_scale)
        
        return positions, colors, opacities, scales

# Test
model = GaussianSplattingTrajectory(num_gaussians=5000)
t = torch.linspace(0, 1, 10)
positions, colors, opacities, scales = model(t)
print(f"Positions shape: {positions.shape}")  # (10, 5000, 3)
print(f"Colors shape: {colors.shape}")        # (5000, 3)
```

### 실험 2 — HexPlane Decomposition

```python
class HexPlaneFeature(nn.Module):
    """4D feature decomposition via 6 planes."""
    def __init__(self, plane_width=128, feature_dim=8):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 6 planes: (xy, xz, xt, yz, yt, zt)
        # Each plane: (width, width, feature_dim)
        self.planes = nn.ModuleList([
            nn.Parameter(torch.randn(plane_width, plane_width, feature_dim) * 0.01)
            for _ in range(6)
        ])
        
        # Plane weights
        self.plane_weights = nn.Parameter(torch.ones(6) / 6)
        
        self.plane_width = plane_width
    
    def normalize_coords(self, coords, dim_pair):
        """Normalize 3D spatial/temporal coords to 2D plane coords."""
        # coords: (B, 4) with (x, y, z, t) in [-1, 1]
        # Return 2D coords in [0, width-1]
        idx_a, idx_b = dim_pair
        
        normalized = (coords[:, [idx_a, idx_b]] + 1) / 2 * (self.plane_width - 1)
        return normalized
    
    def bilinear_interp(self, plane, coords_2d):
        """Bilinear interpolation on 2D plane."""
        # plane: (W, W, F)
        # coords_2d: (B, 2) normalized to [0, W-1]
        
        B = coords_2d.shape[0]
        x, y = coords_2d[:, 0], coords_2d[:, 1]
        
        x0, y0 = torch.floor(x).long(), torch.floor(y).long()
        x1, y1 = x0 + 1, y0 + 1
        
        x0, y0 = torch.clamp(x0, 0, self.plane_width - 1), torch.clamp(y0, 0, self.plane_width - 1)
        x1, y1 = torch.clamp(x1, 0, self.plane_width - 1), torch.clamp(y1, 0, self.plane_width - 1)
        
        wx, wy = x - x0.float(), y - y0.float()
        
        # Bilinear blending
        v00 = plane[x0, y0]  # (B, F)
        v10 = plane[x1, y0]
        v01 = plane[x0, y1]
        v11 = plane[x1, y1]
        
        v = (1 - wx).unsqueeze(-1) * (1 - wy).unsqueeze(-1) * v00 + \
            wx.unsqueeze(-1) * (1 - wy).unsqueeze(-1) * v10 + \
            (1 - wx).unsqueeze(-1) * wy.unsqueeze(-1) * v01 + \
            wx.unsqueeze(-1) * wy.unsqueeze(-1) * v11
        
        return v
    
    def forward(self, coords):
        """
        coords: (B, 4) with (x, y, z, t) in [-1, 1]
        Returns: (B, F) feature vector
        """
        plane_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # xy, xz, xt, yz, yt, zt
        
        features = []
        for plane_idx, (dim_a, dim_b) in enumerate(plane_pairs):
            plane = self.planes[plane_idx]
            coords_2d = self.normalize_coords(coords, (dim_a, dim_b))
            feat = self.bilinear_interp(plane, coords_2d)
            features.append(feat)
        
        # Stack and weight
        features = torch.stack(features, dim=0)  # (6, B, F)
        weights = torch.softmax(self.plane_weights, dim=0).unsqueeze(-1).unsqueeze(-1)  # (6, 1, 1)
        
        feature_out = (features * weights).sum(dim=0)  # (B, F)
        return feature_out

# Test HexPlane
model_hex = HexPlaneFeature(plane_width=64, feature_dim=8)
coords = torch.randn(100, 4)  # 100 points, (x, y, z, t)
features = model_hex(coords)
print(f"HexPlane features shape: {features.shape}")  # (100, 8)
```

### 실험 3 — Trajectory vs HexPlane 비교

```python
def compare_trajectory_vs_hexplane(num_frames=30):
    """Polynomial trajectory 와 HexPlane 의 temporal smoothness 비교."""
    
    # Setup
    traj_model = GaussianSplattingTrajectory(num_gaussians=100)
    hex_model = HexPlaneFeature(plane_width=64, feature_dim=3)
    
    times = torch.linspace(0, 1, num_frames)
    
    # Polynomial trajectory
    positions_traj = []
    for t in times:
        pos, _, _, _ = traj_model(t.unsqueeze(0))
        positions_traj.append(pos[0, :10, :].detach())  # first 10 Gaussians
    positions_traj = torch.stack(positions_traj)  # (frames, 10, 3)
    
    # HexPlane features
    coords_hex = torch.stack([
        torch.linspace(-1, 1, num_frames),
        torch.zeros(num_frames),
        torch.zeros(num_frames),
        torch.linspace(0, 1, num_frames),  # time dimension
    ], dim=-1)  # (frames, 4)
    
    features_hex = []
    for coord in coords_hex:
        feat = hex_model(coord.unsqueeze(0))
        features_hex.append(feat)
    features_hex = torch.stack(features_hex)  # (frames, 3)
    
    # Smoothness metric: L2 norm of frame-to-frame differences
    smooth_traj = torch.norm(positions_traj[1:] - positions_traj[:-1], dim=-1).mean().item()
    smooth_hex = torch.norm(features_hex[1:] - features_hex[:-1], dim=-1).mean().item()
    
    print(f"Trajectory smoothness (L2 delta): {smooth_traj:.6f}")
    print(f"HexPlane smoothness (L2 delta):   {smooth_hex:.6f}")
    
    return smooth_traj, smooth_hex

smooth_t, smooth_h = compare_trajectory_vs_hexplane(num_frames=30)
```

---

## 🔗 실전 활용

### 1. Long-Horizon Video (30+ frames)

**Polynomial trajectory 추천**:
- Explicit velocity 정보
- Extrapolation 가능
- Per-Gaussian 계수로 미세 조정 용이

```python
# Initialize from optical flow
optical_flow = compute_optical_flow(video)  # (T-1, H, W, 2)

# Fit polynomial to flow (coarse to fine)
for gaussian_idx in range(num_gaussians):
    motion_vector = optical_flow[gaussian_idx]  # simplified
    # Polynomial least squares fit
    coeffs = np.polyfit(times[:-1], motion_vector, deg=2)
    model.trajectory.a[0][gaussian_idx] = torch.from_numpy(coeffs[0])
    model.trajectory.a[1][gaussian_idx] = torch.from_numpy(coeffs[1])
```

### 2. Memory-Constrained Setting

**HexPlane 추천**:
- Global 모델 (per-Gaussian 아님)
- 큰 Gaussian cloud 에서 효율적
- Inference fast (GPU bandwidth)

```python
# Deploy on edge device
hex_model = HexPlaneFeature(plane_width=32, feature_dim=4)  # minimal
# Render at 720p, 30fps with quantization
```

### 3. Hybrid: Polynomial + HexPlane

```python
class Hybrid4DGS(nn.Module):
    """Combines both approaches."""
    def __init__(self, num_gaussians=5000):
        super().__init__()
        self.traj = PolynomialTrajectory(num_gaussians, degree=1)  # linear only
        self.hex_color = HexPlaneFeature(plane_width=64, feature_dim=3)
    
    def forward(self, t):
        # Positions from polynomial (physics-aware)
        positions = self.traj(t)
        
        # Colors from HexPlane (flexible appearance)
        coords = torch.stack([
            torch.ones_like(t) * 0,
            torch.ones_like(t) * 0,
            torch.ones_like(t) * 0,
            t
        ], dim=-1)
        colors_feat = self.hex_color(coords)  # (B, 3)
        
        return positions, colors_feat
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Polynomial degree K 고정 | Complex nonlinear motion 불가 (K 증가, but overfitting) |
| Gaussian covariance 정적 | Deformable objects 약함 (time-dependent $\Sigma$ 추가) |
| HexPlane rank 제한 | High-rank motion underrepresented (higher resolution 또는 decomposition 변경) |
| Per-Gaussian 독립성 | Self-intersection 감지 불가 (post-processing 필요) |
| Continuous time interpolation | Discrete frame-only 학습 (smoother appearance loss) |

---

## 📌 핵심 정리

### Polynomial Trajectory (Yang 2024)

$$\boxed{\boldsymbol{\mu}_i(t) = \boldsymbol{\mu}_{i,0} + \sum_{k=1}^{K} \boldsymbol{a}_{i,k} t^k}$$

- **장점**: 명시적, 해석 가능, extrapolation, per-Gaussian control
- **단점**: memory per Gaussian, high-degree ringing

### HexPlane Decomposition (Wu 2024)

$$\boxed{F(x,y,z,t) = \sum_{6 \text{ planes}} w_i \cdot \text{bilinear}(P_i, \text{coords})}$$

- **장점**: memory efficient (global), fast lookup
- **단점**: fixed rank, extrapolation weak

---

## 🤔 생각해볼 문제

**문제 1** (기초): Polynomial degree $K = 2$ (quadratic) 를 사용할 때, velocity 는 linear 이다. 이게 Gaussian 의 acceleration (2nd derivative of position) 을 represent 할 수 있다는 뜻인가?

<details>
<summary>해설</summary>

네, 정확히 맞습니다:
- $\boldsymbol{\mu}(t) = \boldsymbol{\mu}_0 + \boldsymbol{a}_1 t + \boldsymbol{a}_2 t^2$
- $\mathbf{v}(t) = \boldsymbol{a}_1 + 2\boldsymbol{a}_2 t$ (velocity, 1st deriv)
- $\mathbf{a}(t) = 2\boldsymbol{a}_2$ (acceleration, 2nd deriv, **constant**)

즉, $K = 2$ 는 **constant acceleration** 만 캡처합니다.

실제로:
- 자유 낙하: constant acceleration ✓
- 탄성 충돌: piecewise constant acceleration → K=2 로 각 segment 별 approximation
- 회전 운동: centripetal acceleration (non-constant) → $K \geq 3$ 필요

따라서 $K = 2$ 는 **짧은 motion** 에 적합, longer sequences 는 $K \geq 3$ 권장. $\square$

</details>

**문제 2** (심화): HexPlane 의 6개 plane 은 모두 필요한가? 예를 들어 **pure translation** (모든 축에 uniform 하게) 이면, 몇 개 plane 만으로 충분할까?

<details>
<summary>해설</summary>

Pure translation 분석:
- Position: $\mathbf{x}(t) = \mathbf{x}_0 + \mathbf{v} t$ (constant velocity)
- 4D decomposition: translation 은 실제로 **rank-1** 구조

필요한 plane:
- XT, YT, ZT plane 만으로 충분 (time 축과 각 spatial 축)
- XY, XZ, YZ plane 은 불필요

즉, **3/6 plane 사용 가능**.

하지만 실제:
- Rotation, deformation, occlusion 등 혼재
- 모든 6 plane 사용 → more robust
- Learned weights: 자동으로 불필요한 plane 을 down-weight 가능

**최적화**: Adaptive plane selection — training 중 각 plane 의 gradient magnitude 모니터링, 작은 것 pruning. $\square$

</details>

**문제 3** (논문 비평): Polynomial trajectory 에서 degree K 를 validation set 에서 선택하면, overfitting 을 어떻게 방지할까? Regularization 전략은?

<details>
<summary>해설</summary>

Polynomial overfitting 위험:
- 높은 K: training data 정확히 fit, but oscillations (Runge phenomenon)
- 특히 endpoint 에서 extreme values

**해결책**:

1. **Degree 제한**: 실무에서 K ≤ 3 (cubic) 권장
   - K=1: constant velocity
   - K=2: linear accel (대부분의 natural motion)
   - K=3: higher-order effects (rare)

2. **L2 regularization on coefficients**:
   $$\mathcal{L}_{\text{coeff}} = \sum_i \sum_{k=1}^{K} \|\boldsymbol{a}_{i,k}\|^2$$
   → 큰 coefficient penalty

3. **Smoothness loss on velocity**:
   $$\mathcal{L}_{\text{smooth}} = \int_0^T \|\mathbf{v}'(t)\|^2 dt = \int_0^T \|2\boldsymbol{a}_2 + 6\boldsymbol{a}_3 t + \ldots\|^2$$
   → jerk (3rd deriv) 제약

4. **Cross-validation**: 
   - Train on frames 0-24
   - Test extrapolation on frames 25-29
   - K=2 보통 best (K=3 overfits)

**결론**: K=2 + smoothness loss 가 practical balance. Yang 2024 paper 도 K=2 사용. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-hypernerf.md) | [📚 README](../README.md) | [다음 ▶](./04-video-reconstruction.md)

</div>
