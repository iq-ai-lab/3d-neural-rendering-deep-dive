# 02. Color · Opacity 와 Spherical Harmonics (Kerbl 2023)

## 🎯 핵심 질문

- Spherical Harmonics (SH) 는 구면(sphere) 위의 함수들의 직교 정규 기저(orthonormal basis) 인데, 이것이 view-dependent color 를 어떻게 표현하는가?
- 왜 NeRF 의 view-direction MLP 대신 SH 전개를 사용하면 **폐쇄형(closed-form) 평가** 가 가능하고, 이것이 렌더링 속도에 어떤 영향을 미치는가?
- L=3 (degree 3) 을 선택했을 때 16개 계수 × 3 (RGB) = 48 파라미터 / Gaussian 은 어떻게 나오는가? 더 높은 L 을 사용하지 않는 이유는?
- Spherical Harmonics 의 직교성(orthogonality) 이 학습 안정성에 어떤 역할을 하는가?

---

## 🔍 왜 Spherical Harmonics 가 3DGS 의 색상 표현인가

NeRF 는 view direction **d** 를 positional encoding 후 MLP 에 입력해 view-dependent color c(d) 를 생성합니다. 그러나:

**NeRF 의 한계**:
- **MLP 평가 비용**: 각 ray 의 샘플점 × view MLP → rendering 느림
- **Spectral bias**: 낮은 주파수에 편향 → high-frequency specular detail 어려움
- **학습 불안정**: MLP 의 수렴 느림, hyperparameter 민감

**3DGS 의 솔루션: Spherical Harmonics**:
- **직교 기저**: Fourier 처럼 완전 분해 가능, 서로 직교
- **폐쇄형 평가**: SH(d) 는 간단한 다항식 계산 (MLP 불필요)
- **수렴 빠름**: 각 계수가 **독립적 학습 가능** (직교성)
- **표현력**: L=3 으로 충분한 specular detail 포착

이 문서에서 다루는 내용:
1. Spherical Harmonics 의 정의와 직교성
2. L²(S²) 함수 공간과의 관계
3. 3DGS 에서의 구체적 사용 (L=3, 48 계수)
4. NeRF MLP 와의 정량 비교

---

## 📐 수학적 선행 조건

- 선형대수: 직교 집합, 정규화, 함수 공간 inner product
- Fourier 분석: 삼각함수 직교성, 전개 계수
- Legendre 다항식 및 associated Legendre 함수
- (선택) 조화 함수(harmonic functions), Laplacian 의 고유함수

---

## 📖 직관적 이해

### 함수 공간으로서의 구면

구면 $S^2 = \{(\theta, \phi) : \theta \in [0, \pi], \phi \in [0, 2\pi)\}$ 위의 함수들을 생각해봅시다. 예를 들어:
- $f(\theta, \phi) = \cos\theta$ (남북 방향)
- $f(\theta, \phi) = \sin\theta\cos\phi$ (동서 방향)
- ...

**함수 공간 $L^2(S^2)$**: 구면 위의 제곱 적분 가능 함수들의 집합. Inner product:
$$\langle f, g \rangle = \int_0^{2\pi}\int_0^{\pi} f(\theta, \phi) \overline{g(\theta, \phi)} \sin\theta \, d\theta d\phi$$

### Spherical Harmonics 는 L²(S²) 의 orthonormal basis

마치 $e^{ikx}$ 가 $L^2([0, 2\pi])$ 의 기저인 것처럼, Spherical Harmonics $Y_l^m(\theta, \phi)$ 는 $L^2(S^2)$ 의 **완전 정규 직교 기저**:

$$\langle Y_l^m, Y_{l'}^{m'} \rangle = \delta_{ll'} \delta_{mm'}$$

따라서 **모든** $f \in L^2(S^2)$ 는:
$$f(\theta, \phi) = \sum_{l=0}^{\infty} \sum_{m=-l}^{l} c_{lm} Y_l^m(\theta, \phi)$$

with $\sum |c_{lm}|^2 = \|f\|^2$ (Parseval).

### 3DGS 에서의 적용: View-dependent color

각 Gaussian $i$ 마다, color as function of view direction:
$$\mathbf{c}_i(\mathbf{d}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{i,lm} Y_l^m(\mathbf{d})$$

여기서:
- **d** = unit view direction (구면 좌표 $(\theta, \phi)$ 로 변환 가능)
- **$c_{i,lm}$** = Gaussian $i$ 의 SH 계수 (학습 가능)
- **L=3** = truncation degree

**계수 개수**: $\sum_{l=0}^{3} (2l+1) = 1 + 3 + 5 + 7 = 16$ per color channel → **RGB × 16 = 48** 계수 / Gaussian.

### 직관: Fourier 의 구면 버전

- 1D Fourier: $f(x) = \sum_k c_k e^{ikx}$ (linear 도메인에서)
- 2D Fourier on torus: $f(\phi_1, \phi_2) = \sum_k \sum_j c_{kj} e^{i(k\phi_1 + j\phi_2)}$
- **Spherical Harmonics**: $f(\theta, \phi) = \sum_l \sum_m c_{lm} Y_l^m(\theta, \phi)$ (구면 도메인에서)

낮은 degree $l$ 은 smooth 한 변화 (diffuse), 높은 $l$ 은 sharp 한 detail (specular).

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Spherical Harmonics

Spherical Harmonics degree $l$ 과 order $m$ ($-l \leq m \leq l$) 에 대해:

$$Y_l^m(\theta, \phi) = \sqrt{\frac{2l+1}{4\pi} \frac{(l-|m|)!}{(l+|m|)!}} P_l^{|m|}(\cos\theta) e^{im\phi}$$

여기서:
- **$P_l^{|m|}$** = associated Legendre polynomial
- **$\theta$** = polar angle (z축으로부터), $\theta \in [0, \pi]$
- **$\phi$** = azimuthal angle, $\phi \in [0, 2\pi)$
- Normalization: $\int_{S^2} |Y_l^m|^2 d\Omega = 1$ (unit norm)

### 정의 1.2 — 직교성 (Orthonormality)

$$\int_{S^2} Y_l^m(\Omega) \overline{Y_{l'}^{m'}(\Omega)} d\Omega = \delta_{ll'} \delta_{mm'}$$

여기서 $d\Omega = \sin\theta \, d\theta d\phi$ 는 구면 measure.

### 정의 1.3 — L²(S²) 함수 공간

$$L^2(S^2) := \left\{ f: S^2 \to \mathbb{C} \,\Big|\, \int_{S^2} |f(\Omega)|^2 d\Omega < \infty \right\}$$

Inner product: $\langle f, g \rangle = \int_{S^2} f(\Omega) \overline{g(\Omega)} d\Omega$.

**Completeness**: {$Y_l^m : l \in \mathbb{Z}_{\geq 0}, m \in [-l, l] \cap \mathbb{Z}$} 는 $L^2(S^2)$ 의 complete orthonormal basis.

### 정의 1.4 — SH 전개 (Expansion)

모든 $f \in L^2(S^2)$ 에 대해:
$$f(\Omega) = \sum_{l=0}^{\infty} \sum_{m=-l}^{l} c_l^m Y_l^m(\Omega)$$

여기서:
$$c_l^m = \int_{S^2} f(\Omega) \overline{Y_l^m(\Omega)} d\Omega$$

**Truncation at degree L**: $f \approx \sum_{l=0}^{L} \sum_{m=-l}^{l} c_l^m Y_l^m(\Omega)$ — projection onto first $(L+1)^2$ basis functions.

### 정의 1.5 — 3DGS Color parameterization

각 Gaussian $i$ 마다:
$$\mathbf{c}_i(\mathbf{d}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} \mathbf{C}_i^{lm} Y_l^m(\mathbf{d})$$

여기서:
- **d** = unit view direction (direction from Gaussian to camera)
- **$\mathbf{C}_i^{lm} \in \mathbb{R}^3$** = RGB coefficients for SH basis function $Y_l^m$
- Total: $(L+1)^2 \times 3$ coefficients per Gaussian (L=3 → 16 × 3 = 48)
- $\mathbf{c}_i(\mathbf{d}) \in \mathbb{R}^3$ = RGB color

---

## 🔬 정리와 증명

### 정리 1.1 — SH 의 직교 완전성 (Completeness)

**명제**: Spherical Harmonics $\{Y_l^m : l \geq 0, -l \leq m \leq l\}$ 는 $L^2(S^2)$ 의 complete orthonormal basis.

**증명 스케치** (수학 교과서 참조: Folland "Harmonic Analysis in Phase Space"):

1. **Orthonormality**: 정의에 의해 $\langle Y_l^m, Y_{l'}^{m'} \rangle = \delta_{ll'} \delta_{mm'}$ (확인 가능하나 계산 생략)

2. **Completeness**: Associated Legendre 다항식 {$P_l^{|m|}(\cos\theta)$} 가 $L^2([-1, 1])$ 의 완전 기저이고, $\{e^{im\phi}\}$ 가 $L^2([0, 2\pi])$ 의 완전 기저이므로, 곱 $Y_l^m(\theta, \phi)$ 도 완전 → Cauchy-Schwarz, Weierstrass approximation.

따라서 **모든** continuous function $f: S^2 \to \mathbb{R}$ 은 SH 전개로 표현 가능하고, 균일 수렴. ∎

### 정리 1.2 — Parseval 항등식

$$\|f\|_{L^2(S^2)}^2 = \sum_{l=0}^{\infty} \sum_{m=-l}^{l} |c_l^m|^2$$

**의미**: SH 계수의 크기로 함수의 에너지(분산) 측정 가능.

### 정리 1.3 — Low-frequency 전개의 근사 오차

$f$ 를 degree $L$ 으로 truncate:
$$f_L(\Omega) := \sum_{l=0}^{L} \sum_{m=-l}^{l} c_l^m Y_l^m(\Omega)$$

그러면:
$$\|f - f_L\|_{L^2(S^2)}^2 = \sum_{l > L} \sum_{m} |c_l^m|^2$$

특히, $f$ 가 smooth 할수록 high-degree 계수가 빠르게 감소 → low L 로도 좋은 근사.

---

## 💻 실전 구현 (PyTorch)

### 실험 1 — Spherical Harmonics 기저 함수 평가

```python
import torch
import numpy as np
from scipy.special import legendre as sp_legendre
from scipy.special import lpmv

def spherical_harmonics(l, m, theta, phi):
    """
    Y_l^m(theta, phi) 계산
    
    Args:
        l: degree (int >= 0)
        m: order (int, -l <= m <= l)
        theta: polar angle (0 to pi), shape (N,)
        phi: azimuthal angle (0 to 2pi), shape (N,)
    
    Return:
        Y_l^m: shape (N,), complex or real (we use real convention)
    """
    import math
    
    if abs(m) > l:
        raise ValueError(f"Invalid (l, m) = ({l}, {m})")
    
    # Normalization constant
    norm = np.sqrt((2*l + 1) / (4*np.pi) * 
                   math.factorial(l - abs(m)) / math.factorial(l + abs(m)))
    
    # Associated Legendre polynomial
    P_lm = np.abs(lpmv(abs(m), l, np.cos(theta)))
    
    # Exponential part
    exp_part = np.cos(m * phi) if m >= 0 else np.sin(abs(m) * phi)
    
    result = norm * P_lm * exp_part
    return result

# Test: evaluate SH at specific direction
theta = np.array([np.pi/4, np.pi/3])  # 45 degrees, 60 degrees
phi = np.array([0, np.pi/2])          # 0 degrees, 90 degrees

Y_0_0 = spherical_harmonics(0, 0, theta, phi)
Y_1_1 = spherical_harmonics(1, 1, theta, phi)
print(f"Y_0^0 at (π/4, 0): {Y_0_0[0]:.6f}")
print(f"Y_1^1 at (π/4, 0): {Y_1_1[0]:.6f}")

# Y_0^0 should be 1/sqrt(4π) everywhere (constant)
print(f"Y_0^0 should be constant: {1/np.sqrt(4*np.pi):.6f}")
```

### 실험 2 — View-dependent color 로서의 SH 전개

```python
def pytorch_spherical_harmonics(l, m, theta, phi):
    """
    PyTorch implementation of real SH
    theta: (N,), phi: (N,)
    """
    import math
    import scipy.special as sp
    
    # numpy 로 계산 후 torch 로 변환
    if isinstance(theta, torch.Tensor):
        theta_np = theta.detach().cpu().numpy()
        phi_np = phi.detach().cpu().numpy()
    else:
        theta_np, phi_np = theta, phi
    
    norm = np.sqrt((2*l + 1) / (4*np.pi) * 
                   math.factorial(l - abs(m)) / math.factorial(l + abs(m)))
    P_lm = np.abs(sp.lpmv(abs(m), l, np.cos(theta_np)))
    
    if m >= 0:
        exp_part = np.cos(m * phi_np)
    else:
        exp_part = np.sin(abs(m) * phi_np)
    
    result = torch.from_numpy(norm * P_lm * exp_part).float()
    return result

def sh_color_encoding(view_dir, sh_coeffs, max_degree=3):
    """
    View-dependent color 계산
    
    Args:
        view_dir: (N, 3) unit vectors (dx, dy, dz)
        sh_coeffs: (N, (max_degree+1)^2, 3) SH 계수
        max_degree: L (default 3)
    
    Return:
        colors: (N, 3) RGB
    """
    # view_dir 을 구면좌표로 변환
    # theta = arccos(dz), phi = atan2(dy, dx)
    theta = torch.acos(torch.clamp(view_dir[:, 2], -1, 1))
    phi = torch.atan2(view_dir[:, 1], view_dir[:, 0])
    
    colors = torch.zeros(view_dir.shape[0], 3, dtype=torch.float32)
    
    coeff_idx = 0
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            Y_lm = pytorch_spherical_harmonics(l, m, theta, phi)  # (N,)
            colors += sh_coeffs[:, coeff_idx, :] * Y_lm.unsqueeze(-1)  # (N, 3)
            coeff_idx += 1
    
    return colors

# Test
N = 10
view_dirs = torch.randn(N, 3)
view_dirs = view_dirs / torch.norm(view_dirs, dim=1, keepdim=True)  # normalize

sh_coeffs = torch.randn(N, 16, 3)  # (L=3) -> 16 coefficients, RGB
colors = sh_color_encoding(view_dirs, sh_coeffs, max_degree=3)
print(f"Colors shape: {colors.shape}")  # (N, 3)
print(f"Color values in [0, inf): min={colors.min():.4f}, max={colors.max():.4f}")
```

### 실험 3 — NeRF MLP vs SH 비교

```python
import torch
import torch.nn as nn
import time

class ViewMLP(nn.Module):
    """NeRF-style view-dependent MLP"""
    def __init__(self, input_dim=27, hidden_dim=128):
        super().__init__()
        # 3D direction -> positional encoding (3*L=3*9=27 dims)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)
        self.relu = nn.ReLU()
    
    def forward(self, view_dir):
        x = self.relu(self.fc1(view_dir))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class SHColor(nn.Module):
    """3DGS Spherical Harmonics color"""
    def __init__(self, num_gaussians=1000):
        super().__init__()
        self.sh_coeffs = nn.Parameter(torch.randn(num_gaussians, 16, 3) * 0.1)
    
    def forward(self, view_dir, gauss_idx):
        # view_dir: (N, 3) or (1, 3)
        # gauss_idx: (1,) which Gaussian
        # return: (N, 3)
        return sh_color_encoding(view_dir, self.sh_coeffs[gauss_idx:gauss_idx+1], max_degree=3)

# Benchmark
num_samples = 10000
view_dirs_positional = torch.randn(num_samples, 27)  # pre-encoded
view_dirs_unit = torch.randn(num_samples, 3)
view_dirs_unit = view_dirs_unit / torch.norm(view_dirs_unit, dim=1, keepdim=True)

mlp = ViewMLP(27, 128)
sh = SHColor(1)

# Warm-up
_ = mlp(view_dirs_positional[:100])
_ = sh(view_dirs_unit[:100], 0)

# MLP timing
start = time.time()
for _ in range(100):
    _ = mlp(view_dirs_positional)
mlp_time = time.time() - start

# SH timing
start = time.time()
for _ in range(100):
    _ = sh(view_dirs_unit, 0)
sh_time = time.time() - start

print(f"MLP time (100 iters): {mlp_time:.4f}s")
print(f"SH time (100 iters): {sh_time:.4f}s")
print(f"Speedup: {mlp_time / sh_time:.2f}x")
# Expected: SH ~5-10x faster
```

---

## 🔗 실전 활용

### 1. Coefficient 초기화

- **Zero init**: 모든 $c_{lm} = 0$ (이론적으로 흰색, 실제로는 diffuse)
- **Random init**: 작은 Gaussian distribution (e.g., $\mathcal{N}(0, 0.1)$)
- **COLMAP RGB 기반**: 초기 점 색상에서 SH 계수 추정 (최소 제곱법)

### 2. Gradient 제어

- **DC component ($l=0$)**: diffuse color, 일반적으로 가장 중요
- **Higher harmonics ($l > 0$)**: view-dependent (specular) detail
- Optional: 높은 degree 계수에 낮은 learning rate 적용

### 3. Rendering integration

```python
# Pseudocode
for each ray:
    for each Gaussian along ray:
        view_dir = (camera - gaussian_pos) / norm
        color = sh_color_encoding(view_dir, sh_coeffs)  # 16 component evaluation
        alpha = compute_alpha(gaussian, ray)
        final_color += alpha * color * transmittance
        transmittance *= (1 - alpha)
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|------------|
| L=3 충분 | 매우 high-frequency specular (hot highlights) 는 miss. 필요시 L=4 (25 coeff) |
| Color in [-∞, ∞] 가능 | tone mapping (sigmoid, ReLU 등) 필요해서 loss 에서 brightness control |
| View direction 정확 | Novel view 에서 specular artifact 가능 (view extrapolation) |
| Spherical parameterization | Poles (θ=0, π) 에서 φ singular. 안정화 필요 (clipping, regularization) |
| Per-Gaussian SH 계수 | 매우 많은 Gaussian (e.g., 100k) → 메모리 5MB (100k × 48 × 4 bytes) |

---

## 📌 핵심 정리

$$\boxed{\mathbf{c}(\mathbf{d}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} Y_l^m(\mathbf{d}), \quad L=3 \Rightarrow 16 \text{ basis}, \, 48 \text{ params/Gaussian}}$$

| 양 | 정의 | 역할 |
|----|------|------|
| **$Y_l^m$** | SH basis function | Orthonormal on $S^2$ |
| **$c_{lm}$** | SH coefficient | Learnable, per Gaussian |
| **$\mathbf{d}$** | Unit view direction | Spherical coords (θ, φ) |
| **L=3** | Degree truncation | 16 basis → sufficient specular |
| **Completeness** | $\{Y_l^m\}$ 는 complete basis | All continuous functions expressible |

**vs NeRF MLP**:
- ✓ Closed-form evaluation (다항식만)
- ✓ ~5-10x faster rendering
- ✓ Orthogonal → 안정적 학습
- ✗ Frequency 구조 고정 (adaptivity 없음)

---

## 🤔 생각해볼 문제

**문제 1** (기초): $Y_0^0(\theta, \phi)$ 를 계산하라. 모든 방향에서 상수인가?

<details>
<summary>해설</summary>

$$Y_0^0 = \sqrt{\frac{1}{4\pi}}$$

네, 상수. 이것이 "diffuse" component — view direction 무관.

</details>

**문제 2** (심화): NeRF 의 positional encoding $\gamma(d) = (d, \sin d, \cos d, \sin 2d, \cos 2d, \ldots)$ 과 SH 의 차이점은? 왜 SH 가 더 "자연스러운" 기저인가?

<details>
<summary>해설</summary>

**Positional encoding**: 1D Fourier, Euclidean space 최적화. ℝ³ 의 각 좌표마다 적용 → 큰 차원 (L=4 → 27D)

**SH**: Spherical domain 고유 기저, 구면 위의 함수 표현에 최적화. 직교 완전성으로 각 계수 독립 학습 가능.

SH 가 더 compact (16 vs 27), 더 해석 가능 (각 basis 가 구면 조화(harmonic) 함수).

</details>

**문제 3** (논문 비평): 왜 3DGS 는 opacity α 는 단순 scalar 인데, color 는 48D SH 계수인가?

<details>
<summary>해설</summary>

**Opacity α**: 기하학적 정보 (coverage), view-direction independent → scalar 충분

**Color c(d)**: View-dependent specular reflection → view direction 의 함수 필요 → SH 로 표현

실제로는 α 도 view-dependent 될 수 있으나 (Fresnel), 3DGS 논문에서는 단순화.

</details>

---

<div align="center">

[◀ 이전](./01-anisotropic-gaussian.md) | [📚 README](../README.md) | [다음 ▶](./03-ewa-projection.md)

</div>
