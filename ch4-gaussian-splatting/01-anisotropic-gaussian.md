# 01. 3D Gaussian 의 Anisotropic Parameterization (Kerbl 2023)

## 🎯 핵심 질문

- 3D 공간의 한 점 μ 주변에서 Gaussian 함수 G(x) 를 어떻게 정의하고, 왜 covariance 행렬 Σ 가 항상 positive-definite 이어야 하는가?
- Σ = R(q)SS^T R(q)^T 인수분해 (factorization) 의 의미는 무엇인가? Quaternion q 와 scaling 행렬 S 는 각각 무엇을 제어하는가?
- 이 parameterization 이 2D 렌더링을 위해 perspective projection 을 거칠 때 왜 중요한가?
- 실제 구현에서 gradient 가 역전파되는 경로는 어떤 구조인가?

---

## 🔍 왜 이 Parameterization 이 3DGS 의 핵심인가

3D Gaussian Splatting (Kerbl et al. 2023) 은 NeRF 와 달리 **명시적 기하학 표현** 으로 novel view 를 고속 렌더링합니다. 그 핵심은 각 Gaussian 을 다음 5가지로 매개변수화하는 것입니다:

1. **μ ∈ ℝ³** — 중심 위치
2. **q ∈ S³** — unit quaternion (Σ 의 회전)
3. **s ∈ ℝ³₊** — scaling vector (축 방향 분산)
4. **c ∈ ℝ⁴⁸** — Spherical Harmonics 계수 (색)
5. **α ∈ [0, 1]** — opacity

이 문서는 **1, 2, 3 번** (기하학) 의 수학을 다룹니다. Quaternion-scaled decomposition 은:
- Σ 를 **항상 positive-definite** 로 보장
- Rotation + anisotropy 를 **분리** (학습 안정성)
- Perspective projection 후 **Jacobian 계산이 명확** (문서 03)
- 미분가능한 렌더링 파이프라인 구성

---

## 📐 수학적 선행 조건

- 선형대수: 행렬 분해, 고유값, positive-definite (PD) 행렬
- 3D 변환: 회전 행렬, quaternion 기초 (unit norm, multiplication rule)
- 미적분: 행렬의 미분, chain rule, Jacobian
- (선택) Spectral theory — eigendecomposition 와 PD 조건

---

## 📖 직관적 이해

### Gaussian 함수와 Covariance

표준 3D Gaussian 함수:
$$G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

여기서:
- **μ** = 중심 (mean)
- **Σ** = covariance 행렬 (3×3, symmetric)

**Σ 는 왜 PD 여야 하는가?**
- Exponent 에서 $(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})$ 가 **항상 ≥ 0** 이어야 함 (거리 이니까)
- 이는 Σ^{-1} 이 PD ⟺ Σ 도 PD

### Quaternion + Scaling = 안전한 인수분해

일반적으로 3×3 행렬을 covariance 로 직접 사용하면:
- **문제**: 행렬의 6개 독립 원소를 최적화하면, 중간에 비-PD 상태 통과 가능 → Σ^{-1} 계산 불안정
- **해결**: 두 부분으로 나눔:

$$\boldsymbol{\Sigma} = R(\mathbf{q}) \mathbf{S} \mathbf{S}^T R(\mathbf{q})^T$$

여기서:
- **R(q)** = quaternion q 로부터 생성된 3×3 회전 행렬
- **S** = diagonal matrix, $\mathbf{S} = \text{diag}(s_1, s_2, s_3)$, $s_i > 0$

**왜 이렇게 하면 PD 인가?**

$$\boldsymbol{\Sigma} = R \cdot \mathbf{S}\mathbf{S}^T \cdot R^T = R \mathbf{S}\mathbf{S}^T R^T$$

$\mathbf{SS}^T$ 는 명확히 PD (대각선 원소 = $s_1^2, s_2^2, s_3^2 > 0$), R 은 orthogonal (행렬식=1) → congruence 변환도 PD 보존.

### 직관: 회전 + 타원체

- **R(q)**: 3D 타원체의 **방향**을 제어
- **S**: 타원체의 **주축 길이** (anisotropy)를 제어
  - 만약 s = (1, 1, 1) → 구 (isotropic)
  - s = (5, 1, 1) → x축 방향으로 늘어난 타원

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Unit Quaternion 과 회전 행렬

**Quaternion** $\mathbf{q} = (q_w, q_x, q_y, q_z) \in \mathbb{R}^4$, $\|\mathbf{q}\| = 1$ (unit norm).

회전 행렬 $R(\mathbf{q})$ 는:
$$R(\mathbf{q}) = \begin{pmatrix}
1 - 2(q_y^2 + q_z^2) & 2(q_xq_y - q_wq_z) & 2(q_xq_z + q_wq_y) \\
2(q_xq_y + q_wq_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_yq_z - q_wq_x) \\
2(q_xq_z - q_wq_y) & 2(q_yq_z + q_wq_x) & 1 - 2(q_x^2 + q_y^2)
\end{pmatrix}$$

**성질**:
- $R(\mathbf{q})^T R(\mathbf{q}) = I$ (orthogonal)
- $\det(R(\mathbf{q})) = 1$ (proper rotation)
- -q 와 q 는 같은 회전 (double cover) — SU(2) ↔ SO(3) map

### 정의 1.2 — Anisotropic Gaussian Parameterization

**5개 파라미터**:
- $\boldsymbol{\mu} \in \mathbb{R}^3$ — center
- $\mathbf{q} \in S^3 \subset \mathbb{R}^4$ — unit quaternion (with $\|\mathbf{q}\|=1$ constraint)
- $\mathbf{s} \in \mathbb{R}^3_{>0}$ — scaling vector, $s_i > 0$

**Covariance 행렬**:
$$\boldsymbol{\Sigma}(\mathbf{q}, \mathbf{s}) = R(\mathbf{q}) \begin{pmatrix} s_1^2 & 0 & 0 \\ 0 & s_2^2 & 0 \\ 0 & 0 & s_3^2 \end{pmatrix} R(\mathbf{q})^T$$

**Gaussian 함수**:
$$G(\mathbf{x}; \boldsymbol{\mu}, \mathbf{q}, \mathbf{s}) = \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

### 정의 1.3 — Covariance 의 역행렬

$$\boldsymbol{\Sigma}^{-1} = R(\mathbf{q}) \begin{pmatrix} s_1^{-2} & 0 & 0 \\ 0 & s_2^{-2} & 0 \\ 0 & 0 & s_3^{-2} \end{pmatrix} R(\mathbf{q})^T$$

(Σ^{-1} 계산 안정: 대각 역원만 → numerical robust)

---

## 🔬 정리와 증명

### 정리 1.1 — Σ = RSS^T R^T 인수분해로부터 PD 자동 보장

**명제**: 모든 quaternion q ∈ S³ 와 scaling vector s ∈ ℝ³₊ 에 대해, Σ = R(q)SS^T R(q)^T 는 **positive-definite**.

**증명**:

임의의 0이 아닌 벡터 **v** ∈ ℝ³에 대해, $\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} > 0$ 임을 보이자.

**Step 1** — 인수분해:
$$\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} = \mathbf{v}^T R(\mathbf{q}) \mathbf{S}\mathbf{S}^T R(\mathbf{q})^T \mathbf{v}$$

**Step 2** — Rotation 의 orthogonality 활용. $\mathbf{u} := R(\mathbf{q})^T \mathbf{v}$ 로 놓으면 ($R$ orthogonal 이므로 $\|\mathbf{u}\| = \|\mathbf{v}\|$):
$$\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} = \mathbf{u}^T \mathbf{S}\mathbf{S}^T \mathbf{u}$$

**Step 3** — $\mathbf{S}\mathbf{S}^T$ 의 PD 성:
$$\mathbf{S}\mathbf{S}^T = \text{diag}(s_1^2, s_2^2, s_3^2)$$

$s_i > 0$ 이므로:
$$\mathbf{u}^T \mathbf{S}\mathbf{S}^T \mathbf{u} = s_1^2 u_1^2 + s_2^2 u_2^2 + s_3^2 u_3^2 \geq 0$$

**v ≠ 0** ⟹ **u ≠ 0** ⟹ 합 **> 0**.

따라서 **Σ 는 항상 PD**. ∎

### 따름 정리 1.2 — Quaternion Normalization 은 필수

Quaternion q 를 정규화하지 않으면 R(q) 정의되지 않음. 실제 구현에서:
$$\mathbf{q} := \frac{\mathbf{q}_{\text{raw}}}{\|\mathbf{q}_{\text{raw}}\|}$$
각 iteration 후, 또는 loss에 정규화 항 추가.

### 정리 1.3 — Gradient Flow through Quaternion

Loss $L$ 에 대해 $\frac{\partial L}{\partial \mathbf{q}}$ 를 계산할 때, quaternion 제약 $\|\mathbf{q}\|=1$ 을 implicit 로 처리:

**Riemannian gradient** (tangent space 로 project):
$$\nabla_{\mathbf{q}} L = \frac{\partial L}{\partial \mathbf{q}} - \left(\mathbf{q}^T \frac{\partial L}{\partial \mathbf{q}}\right) \mathbf{q}$$

또는 **exponential map** via small rotation: $\mathbf{q} \leftarrow \mathbf{q} \exp(\mathbf{ω}^{\times})$ (where $\mathbf{ω} \in \mathbb{R}^3$ 작은 각속도).

---

## 💻 실전 구현 (PyTorch)

### 실험 1 — Gaussian Parameterization 과 Visualization

```python
import torch
import torch.nn.functional as F
import numpy as np

def quaternion_to_rotation_matrix(q):
    """
    q: (N, 4), unit quaternion [w, x, y, z]
    Return: (N, 3, 3) rotation matrix
    """
    N = q.shape[0]
    q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)
    
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    R = torch.zeros((N, 3, 3), dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)
    
    return R

def compute_covariance(q, s):
    """
    q: (N, 4) unit quaternions
    s: (N, 3) scaling vectors (positive)
    Return Σ: (N, 3, 3) covariance matrix
    """
    R = quaternion_to_rotation_matrix(q)
    S_diag = torch.diag_embed(s)
    Sigma = R @ S_diag @ S_diag.transpose(-2, -1) @ R.transpose(-2, -1)
    return Sigma

def gaussian_3d(x, mu, q, s):
    """
    x: (N, 3) points to evaluate
    mu: (1, 3) center
    q: (1, 4) quaternion
    s: (1, 3) scaling
    Return: (N,) Gaussian values
    """
    Sigma = compute_covariance(q, s)
    Sigma_inv = torch.linalg.inv(Sigma)
    
    diff = x - mu
    mahal = torch.sum(diff @ Sigma_inv[0] * diff, dim=1)
    return torch.exp(-0.5 * mahal)

num_gaussians = 2
mu = torch.randn(num_gaussians, 3)
q = F.normalize(torch.randn(num_gaussians, 4), dim=1)
s = torch.nn.functional.softplus(torch.randn(num_gaussians, 3)) + 0.1

Sigma = compute_covariance(q, s)
print(f"Σ shape: {Sigma.shape}")
print(f"Σ[0] eigenvalues: {torch.linalg.eigvalsh(Sigma[0])}")

for i in range(num_gaussians):
    eigs = torch.linalg.eigvalsh(Sigma[i])
    assert (eigs > 0).all(), f"Gaussian {i} not PD!"
print("✓ All covariance matrices are positive-definite")
```

### 실험 2 — Gradient 역전파 확인

```python
def test_gradient_flow():
    """Gaussian evaluation -> loss -> gradient"""
    num_gaussians = 3
    num_points = 100
    
    mu = torch.randn(num_gaussians, 3, requires_grad=True)
    q_raw = torch.randn(num_gaussians, 4, requires_grad=True)
    s_raw = torch.randn(num_gaussians, 3, requires_grad=True)
    
    q = F.normalize(q_raw, dim=1)
    s = F.softplus(s_raw) + 0.01
    
    points = torch.randn(num_points, 3)
    target = torch.ones(num_points)
    
    Sigma = compute_covariance(q, s)
    Sigma_inv = torch.linalg.inv(Sigma)
    
    loss = 0
    for i in range(num_gaussians):
        diff = points - mu[i:i+1]
        mahal = torch.sum(diff @ Sigma_inv[i] * diff, dim=1)
        G = torch.exp(-0.5 * mahal)
        loss += F.mse_loss(G, target)
    
    loss.backward()
    
    assert mu.grad is not None
    assert q_raw.grad is not None
    assert s_raw.grad is not None
    
    print(f"✓ Gradients computed successfully")
    print(f"  |∇μ| = {torch.norm(mu.grad):.6f}")
    print(f"  |∇q| = {torch.norm(q_raw.grad):.6f}")
    print(f"  |∇s| = {torch.norm(s_raw.grad):.6f}")

test_gradient_flow()
```

### 실험 3 — Scaling 에 따른 타원체 형태

```python
def visualize_anisotropy():
    """다양한 scaling 으로 Gaussian profile 시각화"""
    import matplotlib.pyplot as plt
    
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    
    cases = [
        ("Isotropic (1,1,1)", torch.tensor([[1.0, 1.0, 1.0]])),
        ("Elongated X (3,1,1)", torch.tensor([[3.0, 1.0, 1.0]])),
        ("Elongated Y (1,3,1)", torch.tensor([[1.0, 3.0, 1.0]])),
    ]
    
    grid = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(grid, grid)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (label, s) in enumerate(cases):
        points = torch.from_numpy(
            np.stack([X, Y, np.zeros_like(X)], axis=-1).reshape(-1, 3)
        ).float()
        
        G = gaussian_3d(points, torch.zeros(1, 3), q, s)
        G = G.reshape(X.shape).numpy()
        
        axes[idx].contourf(X, Y, G, levels=20, cmap='viridis')
        axes[idx].set_title(label)
        axes[idx].set_aspect('equal')
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('/tmp/gaussian_anisotropy.png', dpi=100)
    print("✓ Saved visualization")

visualize_anisotropy()
```

---

## 🔗 실전 활용

### 1. Initialization 전략

실제 3DGS 구현에서:
- **COLMAP point cloud**: 각 점 주변의 PCA 로 초기 covariance 추정 → q, s 로 분해
- **Random init**: q ~ uniform S³, s ~ log-normal 또는 작은 상수

### 2. Optimizer 선택

- **Adam** on (μ, q_raw, s_raw) with learning rate ~1e-3
- **Quaternion constraint**: 매 step 후 정규화 또는 exponential map 사용
- **Scaling constraint**: softplus(s_raw) + epsilon 로 s > 0 보장

### 3. 다음 단계: Projection 으로의 연결

문서 03 (EWA Projection) 에서:
- 3D Gaussian Σ 를 2D 렌더링 공간으로 변환
- Perspective projection 의 Jacobian 을 통해 Σ' = JΣJ^T 계산
- Σ' 의 역행렬이 필요 → PD 보장이 중요!

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|------------|
| s > 0 | softplus 또는 exp 로 강제. 0으로 collapse 되면 rank-deficient → 수치 불안정 |
| Quaternion unit norm | 매 iteration 정규화 필수. 안 하면 R(q) 정의 안 됨 |
| Σ inverse 계산 | 매우 작은 s_i → ill-conditioned. 정규화 또는 damping 추가 (Σ + εI) |
| Anisotropy 만으로 회전 표현 | 비균등 scaling 이 있으면 기울어진 타원 — quaternion 이 이를 정확히 표현 |
| 3×3 full covariance | 9개 원소 vs 8개 파라미터 (4 quaternion + 3 scaling + 1 PSD 제약) 더 compact |

---

## 📌 핵심 정리

$$\boxed{\boldsymbol{\Sigma} = R(\mathbf{q}) \begin{pmatrix} s_1^2 & 0 & 0 \\ 0 & s_2^2 & 0 \\ 0 & 0 & s_3^2 \end{pmatrix} R(\mathbf{q})^T}$$

| 양 | 정의 | 역할 |
|----|------|------|
| **μ** | ∈ ℝ³ | Gaussian center |
| **q** | ∈ S³, unit quaternion | Anisotropic direction (rotation) |
| **s** | ∈ ℝ³₊ | Scaling along principal axes |
| **Σ** | RSS^T R^T | Covariance (always PD) |
| **Σ^{-1}** | R S^{-2} R^T | Mahalanobis metric |

**핵심 성질**:
1. ✓ Σ 항상 PD (수치 안정)
2. ✓ 8개 파라미터로 3×3 대칭 행렬 표현
3. ✓ Gradient 안정 (rotation + scaling 분리)
4. ✓ Perspective projection 후 EWA 적용 가능 (문서 03)

---

## 🤔 생각해볼 문제

**문제 1** (기초): Quaternion q = (1, 0, 0, 0) 에 대응하는 회전 행렬 R(q) 를 손으로 계산하라. 이것이 항등 회전인가?

<details>
<summary>해설</summary>

q = (w=1, x=0, y=0, z=0) 를 quaternion 공식에 대입하면 R = I (항등 행렬). 네, identity rotation ✓.

</details>

**문제 2** (심화): Σ = RSS^T R^T 에서, 만약 S = I 이면 Σ = I. 이 경우 모든 방향이 동등한 분산을 가짐. Gaussian 의 형태는?

<details>
<summary>해설</summary>

S = I 이면: Local frame 과 world frame 모두 **구 (sphere)**, 모든 축에 분산 1. R(q) 는 회전해도 구는 불변이므로 아무 효과 없음 (isotropic). Anisotropy 가 생기는 순간 R 이 의미 있음.

</details>

**문제 3** (논문 비평): Kerbl et al. (2023) 은 왜 covariance 를 직접 최적화하지 않고 quaternion + scaling 으로 인수분해했을까?

<details>
<summary>해설</summary>

**직접 최적화의 문제점**:
1. Constraint 위반: Σ 를 6개 변수로 표현하면, gradient descent 중 negative eigenvalue 가능 → 수치 불안정
2. Parameterization 중복: 회전+scaling 의 freedom 과 naive 6-param 의 mismatch
3. 학습 불안정: 대각 원소가 0 근처일 때 gradient vanish

**Quaternion + Scaling 의 이점**: 명시적 분리로 안정적 학습.

</details>

---

<div align="center">

[◀ 이전](../ch3-nerf/06-instant-ngp.md) | [📚 README](../README.md) | [다음 ▶](./02-spherical-harmonics-color.md)

</div>
