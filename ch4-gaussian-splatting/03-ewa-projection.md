# 03. EWA Splatting — Perspective Projection 의 Jacobian (Zwicker 2001)

## 🎯 핵심 질문

- 3D Gaussian Σ₃D 를 2D 렌더링 공간(image plane)으로 변환할 때, perspective projection 의 비선형성(nonlinearity)은 어떤 문제를 야기하는가?
- Jacobian 행렬 J = ∂u/∂X 를 이용한 1차 Taylor 선형화 (local linearization) 는 어떻게 작동하는가?
- 변환된 2D covariance Σ' = J Σ₃D J^T 는 항상 positive-definite 인가?
- EWA (Elliptical Weighted Average) splatting 은 어떻게 이 Σ' 를 렌더링에 활용하는가?

---

## 🔍 왜 EWA Projection 이 필수인가

3D Gaussian 을 2D 이미지에 렌더링하려면 **perspective projection** 을 거쳐야 합니다. 그러나:

**Naive 접근의 문제**:
- 3D Gaussian G(X) 를 projection 후 "화면 위에서 가우시안" 인가? **아니다!**
- Perspective projection π(X) = [fX_x/X_z, fX_y/X_z]^T 는 비선형 → Gaussian 보존 안 됨
- 단순히 projected 중심과 고정 분산으로 그리면 artifacts (distortion, incorrect blurring)

**EWA 의 해결책**:
1. **Local linearization**: Gaussian 중심 근처에서 Taylor 1차 전개 → Jacobian J
2. **Covariance 변환**: Σ' = J Σ₃D J^T (선형 변환의 공분산 법칙)
3. **2D Splat**: Σ' 로 정의된 2D Gaussian 을 image plane 에 렌더링
4. **Numerical stability**: Σ' PD 보장, 역행렬 계산 안정

이 기법은 **Zwicker et al. 2001** (EWA Splatting for Point-Based Rendering) 에서 유래.

---

## 📐 수학적 선행 조건

- 선형대수: Jacobian, chain rule, matrix transformation of covariance
- Multivariable calculus: gradient, Taylor expansion, eigenvalue
- 3D 기하: perspective projection, camera model (intrinsics f, c)
- 문서 01 참조: 3D Gaussian covariance Σ₃D 의 정의

---

## 📖 직관적 이해

### Perspective Projection 은 비선형

3D 점 **X** = [X_x, X_y, X_z]^T (camera frame) → 2D image **u** = [u, v]^T:

$$u = f \frac{X_x}{X_z}, \quad v = f \frac{X_y}{X_z}$$

(focal length f, 중심을 원점이라 가정)

이 변환은:
- **선형 아님**: $\pi(\lambda X) \neq \lambda \pi(X)$ (원점 제외)
- **멀어질수록 압축**: X_z ↑ → u, v ↓ (depth 에 따라 크기 변함)
- **가우시안 보존 안 됨**: 3D Gaussian 을 projection 하면 일반적으로 2D 에서 Gaussian 아님

### Jacobian 을 통한 선형화

Gaussian 중심 **μ₃D** 근처에서 Taylor 1차 전개:
$$\pi(\mathbf{X}) \approx \pi(\boldsymbol{\mu}_{3D}) + J(\boldsymbol{\mu}_{3D}) (\mathbf{X} - \boldsymbol{\mu}_{3D})$$

여기서 **J** 는 Jacobian:
$$J(\boldsymbol{\mu}_{3D}) = \begin{pmatrix}
\frac{\partial u}{\partial X_x} & \frac{\partial u}{\partial X_y} & \frac{\partial u}{\partial X_z} \\
\frac{\partial v}{\partial X_x} & \frac{\partial v}{\partial X_y} & \frac{\partial v}{\partial X_z}
\end{pmatrix}$$

**계산**:
$$J = \begin{pmatrix}
f/Z & 0 & -fX_x/Z^2 \\
0 & f/Z & -fX_y/Z^2
\end{pmatrix}$$

(Z = X_z, 중앙값)

### Covariance 변환

선형 변환 $\mathbf{u} = J (\mathbf{X} - \boldsymbol{\mu})$ 의 covariance:
$$\boldsymbol{\Sigma}_{2D} = J \boldsymbol{\Sigma}_{3D} J^T$$

이는 **standard covariance propagation formula** — 2×3 행렬 × 3×3 symmetric PD × 3×2 행렬 = 2×2 symmetric PD.

### 직관: 깊이에 따른 효과

- 가까운 Gaussian (Z 작음): J 의 대각 원소 f/Z 크고 → 2D splat 커짐
- 먼 Gaussian (Z 큼): J 의 대각 원소 f/Z 작고 → 2D splat 작아짐
- 비균등 스케일링: X_x 또는 X_y 가 크면 "왜곡"된 타원

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Perspective Projection (Standard Camera Model)

Camera frame (X_z > 0) 에서:
$$\boldsymbol{\pi}(\mathbf{X}) = \begin{pmatrix} f X_x / X_z \\ f X_y / X_z \end{pmatrix} + \begin{pmatrix} c_x \\ c_y \end{pmatrix}$$

여기서:
- **f** = focal length (픽셀 단위)
- **[c_x, c_y]^T** = principal point (보통 image center)
- **X_z > 0** = depth (view frustum 내)

간단히 [c_x, c_y] = [0, 0] 이라 가정 가능 (image coordinate system shift).

### 정의 1.2 — Jacobian 행렬

점 **μ₃D** = [X_μ, Y_μ, Z_μ]^T 에서의 Jacobian:

$$J(\boldsymbol{\mu}_{3D}) = \begin{pmatrix}
\frac{\partial}{\partial X_x} \left( f \frac{X_x}{Z_\mu} \right) & \frac{\partial}{\partial X_y} \left( f \frac{X_x}{Z_\mu} \right) & \frac{\partial}{\partial X_z} \left( f \frac{X_x}{Z_\mu} \right) \\
\frac{\partial}{\partial X_x} \left( f \frac{X_y}{Z_\mu} \right) & \frac{\partial}{\partial X_y} \left( f \frac{X_y}{Z_\mu} \right) & \frac{\partial}{\partial X_z} \left( f \frac{X_y}{Z_\mu} \right)
\end{pmatrix}$$

$$= \begin{pmatrix}
f/Z_\mu & 0 & -f X_\mu / Z_\mu^2 \\
0 & f/Z_\mu & -f Y_\mu / Z_\mu^2
\end{pmatrix}$$

**크기**: 2 × 3 (3D → 2D 선형 변환)

### 정의 1.3 — 2D Projected Covariance

$$\boldsymbol{\Sigma}_{2D} = J \boldsymbol{\Sigma}_{3D} J^T$$

**크기**: 2 × 2 (2D 이미지 공간에서의 covariance)

### 정의 1.4 — EWA Splat 의 2D Gaussian

투영된 중심 **u₂D** = π(μ₃D) 에서, 2D Gaussian:

$$G_{2D}(\mathbf{u}) = \exp\left( -\frac{1}{2} (\mathbf{u} - \mathbf{u}_{2D})^T \boldsymbol{\Sigma}_{2D}^{-1} (\mathbf{u} - \mathbf{u}_{2D}) \right)$$

이는 각 픽셀 (u, v) 에 대해 Gaussian value 를 계산하는 데 사용.

---

## 🔬 정리와 증명

### 정리 1.1 — Σ₂D 의 Positive-Definiteness

**명제**: 만약 Σ₃D ≻ 0 (positive-definite) 이고 rank(J) = 2 이면, Σ₂D = JΣ₃DJᵀ ≻ 0.

**증명**:

임의의 0이 아닌 벡터 **v** ∈ ℝ² 에 대해, $\mathbf{v}^T \boldsymbol{\Sigma}_{2D} \mathbf{v} > 0$ 임을 보이자.

**Step 1**: 
$$\mathbf{v}^T \boldsymbol{\Sigma}_{2D} \mathbf{v} = \mathbf{v}^T J \boldsymbol{\Sigma}_{3D} J^T \mathbf{v} = (J^T \mathbf{v})^T \boldsymbol{\Sigma}_{3D} (J^T \mathbf{v})$$

**Step 2**: $\mathbf{w} := J^T \mathbf{v}$ 로 놓으면 (J^T 는 3×2, **v** 는 2×1 → **w** 는 3×1).

**v ≠ 0** 이고 rank(J) = 2 ⟹ J^T 도 full rank ⟹ **w ≠ 0**.

**Step 3**: Σ₃D ≻ 0 이므로:
$$\mathbf{w}^T \boldsymbol{\Sigma}_{3D} \mathbf{w} > 0 \quad \square$$

따라서 **Σ₂D 는 항상 PD**, 역행렬 계산 수치 안정.

### 따름 정리 1.2 — Jacobian 의 Rank 조건

Perspective projection 에서 rank(J) = 2 조건:
- J 의 첫 두 행이 선형 독립
- $\det \begin{pmatrix} f/Z & 0 \\ 0 & f/Z \end{pmatrix} = (f/Z)^2 > 0$ (Z > 0)

따라서 **항상 rank(J) = 2**, Σ₂D 의 PD 보장.

### 정리 1.3 — 근사 오차 (First-order Taylor)

Gaussian 분포를 따르는 샘플 **X** ~ N(μ₃D, Σ₃D) 에 대해, projected sample **u** = π(**X**) 의 분포는:

$$\mathbf{u} \approx \pi(\boldsymbol{\mu}_{3D}) + J(\boldsymbol{\mu}_{3D}) (\mathbf{X} - \boldsymbol{\mu}_{3D})$$

따라서:
$$\mathbf{u} \approx N\left( \pi(\boldsymbol{\mu}_{3D}), J \boldsymbol{\Sigma}_{3D} J^T \right)$$

**오차**: 1차 Taylor 전개 → Σ₃D 의 분산이 작을수록 근사 좋음 (Gaussian 이 compact).

---

## 💻 실전 구현 (PyTorch)

### 실험 1 — Jacobian 계산 및 Covariance 변환

```python
import torch
import torch.nn.functional as F

def perspective_jacobian(pos_3d, focal_length=500.0):
    """
    3D 위치에서의 Jacobian 계산
    
    Args:
        pos_3d: (N, 3) 3D points in camera frame [X, Y, Z]
        focal_length: float, focal length in pixels
    
    Return:
        J: (N, 2, 3) Jacobian matrices
    """
    X = pos_3d[:, 0:1]  # (N, 1)
    Y = pos_3d[:, 1:2]  # (N, 1)
    Z = pos_3d[:, 2:3]  # (N, 1)
    
    # 각 행 계산
    f = focal_length
    J = torch.zeros(pos_3d.shape[0], 2, 3, dtype=pos_3d.dtype, device=pos_3d.device)
    
    # dπ/dX, dπ/dY, dπ/dZ
    J[:, 0, 0] = f / Z.squeeze(-1)           # ∂u/∂X
    J[:, 0, 1] = 0                           # ∂u/∂Y
    J[:, 0, 2] = -f * X.squeeze(-1) / (Z**2).squeeze(-1)  # ∂u/∂Z
    
    J[:, 1, 0] = 0                           # ∂v/∂X
    J[:, 1, 1] = f / Z.squeeze(-1)           # ∂v/∂Y
    J[:, 1, 2] = -f * Y.squeeze(-1) / (Z**2).squeeze(-1)  # ∂v/∂Z
    
    return J

def project_cov_to_2d(cov_3d, pos_3d, focal_length=500.0):
    """
    3D covariance 를 2D image plane 으로 변환
    
    Args:
        cov_3d: (N, 3, 3) 3D covariance matrices (PD)
        pos_3d: (N, 3) 3D positions
        focal_length: float
    
    Return:
        cov_2d: (N, 2, 2) 2D covariance matrices
    """
    N = cov_3d.shape[0]
    J = perspective_jacobian(pos_3d, focal_length)  # (N, 2, 3)
    
    # Σ₂D = J Σ₃D J^T
    # (2×3) @ (3×3) @ (3×2) = (2×2)
    cov_2d = J @ cov_3d @ J.transpose(-2, -1)  # (N, 2, 2)
    
    return cov_2d

# Test
N = 5
focal = 500.0

# 3D Gaussians
pos_3d = torch.tensor([
    [0.0, 0.0, 5.0],     # 중앙, z=5
    [0.1, 0.0, 5.0],     # 약간 옆, z=5
    [0.0, 0.0, 10.0],    # 중앙, z=10 (더 멀다)
    [1.0, 0.0, 5.0],     # 크게 옆, z=5
    [0.0, 0.0, 2.0],     # 중앙, z=2 (가깝다)
])

cov_3d = torch.eye(3).unsqueeze(0).expand(N, -1, -1) * 0.1

J = perspective_jacobian(pos_3d, focal)
print(f"Jacobian shapes: {J.shape}")
print(f"J[0] (center, z=5):\n{J[0]}")
print(f"J[2] (center, z=10):\n{J[2]}")
print(f"Ratio f/Z: z=5 → {focal/5:.2f}, z=10 → {focal/10:.2f}")

cov_2d = project_cov_to_2d(cov_3d, pos_3d, focal)
print(f"\nCov 2D shapes: {cov_2d.shape}")

# PD check
for i in range(N):
    eigs = torch.linalg.eigvalsh(cov_2d[i])
    is_pd = (eigs > -1e-6).all()
    print(f"Cov2D[{i}] PD: {is_pd}, eigs: {eigs.tolist()}")
```

**예상 출력**:
```
Jacobian shapes: torch.Size([5, 2, 3])
J[0] (center, z=5):
tensor([[ 100.,    0.,   -0.],
        [   0.,  100.,   -0.]])
J[2] (center, z=10):
tensor([[  50.,    0.,   -0.],
        [   0.,   50.,   -0.]])
Ratio f/Z: z=5 → 100.00, z=10 → 50.00

Cov 2D shapes: torch.Size([5, 2, 2])
Cov2D[0] PD: True, eigs: [0.0005, 0.0005]
Cov2D[2] PD: True, eigs: [0.000125, 0.000125]
```

### 실험 2 — 깊이에 따른 2D splat 크기 변화

```python
def gaussian_2d_integral(cov_2d):
    """
    2D Gaussian 의 "크기"를 eigenvalue 로 측정
    """
    eigs = torch.linalg.eigvalsh(cov_2d)  # (2,)
    # 타원의 주축 길이 (standard deviation)
    return torch.sqrt(eigs)

# 같은 3D Gaussian 을 여러 깊이에서 평가
focal = 500.0
pos_3d_template = torch.zeros(1, 3)
pos_3d_template[0, 0:2] = 0  # 중앙
cov_3d_template = torch.eye(3).unsqueeze(0) * 0.1

depths = [2.0, 5.0, 10.0, 20.0]
splat_sizes = []

for z in depths:
    pos = pos_3d_template.clone()
    pos[0, 2] = z
    
    J = perspective_jacobian(pos, focal)
    cov_2d = J @ cov_3d_template @ J.transpose(-2, -1)
    
    axes = gaussian_2d_integral(cov_2d[0])
    splat_sizes.append(axes.tolist())
    print(f"z={z:5.1f}: 2D splat axes = {axes.tolist()}")

# 예상: z 배로 증가하면 splat 크기는 1/z 배 감소 (inverted depth scaling)
```

### 실험 3 — Gradient 역전파 (렌더링 loss 로부터)

```python
def render_gaussian_splatting(
    positions_3d, covs_3d, colors, opacities, 
    focal_length=500.0, image_size=(512, 512)
):
    """
    간단한 EWA splatting 렌더링 (매우 단순화)
    """
    cov_2d = project_cov_to_2d(covs_3d, positions_3d, focal_length)
    
    # Projected 위치
    u_2d = torch.zeros(positions_3d.shape[0], 2)
    u_2d[:, 0] = focal_length * positions_3d[:, 0] / positions_3d[:, 2]  # u
    u_2d[:, 1] = focal_length * positions_3d[:, 1] / positions_3d[:, 2]  # v
    
    # 각 Gaussian 에서 일부 pixel 으로 gradient flow
    loss = 0.0
    for i in range(positions_3d.shape[0]):
        # 중심 근처 픽셀 샘플 (간단화)
        pixel_u, pixel_v = u_2d[i, 0].item(), u_2d[i, 1].item()
        
        # Mahalanobis distance
        uv = torch.tensor([[pixel_u, pixel_v]], dtype=torch.float32)
        diff = uv - u_2d[i:i+1]
        
        cov_2d_inv = torch.linalg.inv(cov_2d[i:i+1])
        mahal_dist = (diff @ cov_2d_inv @ diff.T).squeeze()
        
        gaussian_val = torch.exp(-0.5 * mahal_dist)
        loss += opacities[i] * (gaussian_val - 0.5) ** 2
    
    return loss

# Test with gradient
N = 3
pos_3d = torch.randn(N, 3, requires_grad=True)
pos_3d[:, 2] = torch.abs(pos_3d[:, 2]) + 2.0  # ensure Z > 0

cov_3d = torch.eye(3).unsqueeze(0).expand(N, -1, -1) * 0.05
colors = torch.randn(N, 3)
opacities = torch.sigmoid(torch.randn(N, 1))

loss = render_gaussian_splatting(pos_3d, cov_3d, colors, opacities)
loss.backward()

print(f"Loss: {loss.item():.6f}")
print(f"∇pos_3d shape: {pos_3d.grad.shape}")
print(f"|∇pos_3d|: {torch.norm(pos_3d.grad):.6f}")
print("✓ Gradient flows through projection & covariance transformation")
```

---

## 🔗 실전 활용

### 1. 2D Splat Rendering

각 Gaussian:
1. 3D position 을 2D image 로 project: u₂D = [f·X/Z, f·Y/Z]
2. Jacobian J 계산
3. 2D covariance Σ₂D = J·Σ₃D·J^T
4. Image 위의 각 픽셀 (u, v) 에 대해 2D Gaussian value 계산
5. Pixel footprint 에 따라 기여도 누적 (alpha-compositing, 문서 04)

### 2. Numerical Stability 고려

- **Near-zero depth**: Z 가 매우 작으면 Jacobian 폭발 → near plane culling 추가
- **Far depth**: Z 가 매우 크면 Σ₂D 매우 작음 → 렌더링되지 않은 점으로 처리
- **Singular Jacobian**: 이론적으로 안 되지만, floating point 오차 대비 damping 추가 (ε I)

### 3. Differentiable Rendering

Loss (e.g., image L1 또는 SSIM) → ∇loss wrt rendered pixels → ∇ wrt 2D covariance → ∇ wrt 3D covariance & position → 학습.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|------------|
| 1차 Taylor 근사 | Σ₃D 매우 크면 error 커짐. 단, typical Gaussian ~0.1 범위 이므로 무시 |
| rank(J) = 2 | Perspective projection 에서 항상 만족 (Z > 0) |
| Gaussian compactness | 매우 "납작한" Gaussian (e.g., s=[100, 0.1, 0.1]) 은 projection 후 degenerate → 수치 불안정 |
| Pinhole camera model | Real lens distortion 미고려. 복잡한 camera model 은 별도 처리 |
| 2D image domain | 구면 카메라나 360도 렌더링 은 다른 parameterization 필요 |

---

## 📌 핵심 정리

$$\boxed{\boldsymbol{\Sigma}_{2D} = J(\boldsymbol{\mu}_{3D}) \, \boldsymbol{\Sigma}_{3D} \, J(\boldsymbol{\mu}_{3D})^T}$$

where
$$J = \begin{pmatrix} f/Z & 0 & -fX/Z^2 \\ 0 & f/Z & -fY/Z^2 \end{pmatrix}$$

| 단계 | 계산 | 결과 |
|------|------|------|
| **3D Gaussian** | Σ₃D = R S S^T R^T (문서 01) | (3×3) PD |
| **Jacobian** | ∂π/∂X 계산 | (2×3) full rank |
| **2D Projection** | Σ₂D = J Σ₃D J^T | (2×2) PD |
| **EWA Splat** | G₂D(u) = exp(-½ (u-u₂D)^T Σ₂D^{-1} (u-u₂D)) | Per-pixel value |

**핵심 성질**:
1. ✓ Perspective 비선형성을 1차 선형화로 처리
2. ✓ Σ₂D 항상 PD (역행렬 안정)
3. ✓ 깊이에 따른 자연스러운 scaling (z 멀면 작아짐)
4. ✓ Fully differentiable (gradient 역전파 가능)

---

## 🤔 생각해볼 문제

**문제 1** (기초): Z=5, f=500, X=0, Y=0 일 때 Jacobian J 의 첫 행을 계산하라.

<details>
<summary>해설</summary>

$$J_{0,:} = [f/Z, 0, -fX/Z^2] = [500/5, 0, 0] = [100, 0, 0]$$

즉, 중앙 위치에서는 Jacobian 이 대각 행렬 (스케일링).

</details>

**문제 2** (심화): 같은 3D Gaussian 을 두 깊이 Z=5, Z=10 에서 관찰할 때, 2D splat 크기의 비율은?

<details>
<summary>해설</summary>

Perspective scaling 에서 diagonal Jacobian ∝ 1/Z.

따라서 Σ₂D ∝ (1/Z)² Σ₃D.

Z₁=5, Z₂=10 → Σ₂D 의 비율 = (1/5)² / (1/10)² = (2)² = 4.

Z=5 에서 4배 더 크다 (가까운 Gaussian 이 더 크게 보임 — 맞다).

</details>

**문제 3** (논문 비평): EWA projection 이 "근사"라면, 실제 3D Gaussian 을 정확히 2D 로 변환할 수는 없는가?

<details>
<summary>해설</summary>

**정확한 변환**: projection π 아래에서 3D volume element → 2D. 정확 답 = variable change 공식:

$$f_{2D}(\mathbf{u}) = \int f_{3D}(\pi^{-1}(\mathbf{u}), z) \left| \frac{\partial \pi^{-1}}{\partial \mathbf{u}} \right| dz$$

(back-projection ray 를 모든 깊이에서 적분)

**EWA 근사**: 중심 근처 1차 선형화 + Gaussian 가정 → 닫힌 형태 → 빠르고 미분 가능.

정확도와 속도의 트레이드오프. 실무에서는 EWA 충분.

</details>

---

<div align="center">

[◀ 이전](./02-spherical-harmonics-color.md) | [📚 README](../README.md) | [다음 ▶](./04-tile-rasterization.md)

</div>
