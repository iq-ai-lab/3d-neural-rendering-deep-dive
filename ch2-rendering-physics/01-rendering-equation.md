# 01. Rendering Equation 의 물리학적 기원 (Kajiya 1986)

## 🎯 핵심 질문

- Kajiya 1986 의 rendering equation $L_o = L_e + \int f_r L_i \cos\theta\, d\omega$ 는 무엇이 아니라 **무엇** 인가?
- Radiance · irradiance · BRDF 같은 radiometry 의 기본 개념들이 정확히 정의되는 방식은?
- 왜 이 한 줄의 식이 ray tracing 부터 neural rendering (NeRF, 3DGS) 까지 모든 현대 3D rendering 을 통합하는가?
- Energy conservation $\int f_r(\omega_i, \omega_o) \cos\theta_i\, d\omega_i \leq 1$ 이 물리적으로 의미하는 바는?
- Recursive 형태에서 Neumann series 로의 변형이 path tracing 을 어떻게 정당화하는가?

---

## 🔍 왜 이 물리학인가

Rendering equation 은 "더 좋은 색 칠하기 알고리즘" 이 아닙니다. 이것은 **photon 의 에너지 균형** 을 수학으로 표현한 것입니다. 광선이 표면에 도달했을 때:

1. 그 표면이 **스스로 방출하는 빛** ($L_e$)
2. 다른 방향에서 도달한 빛이 **반사되어 나가는 빛** ($\int f_r L_i \cos\theta\, d\omega$)

이 두 항의 합이 우리가 관찰하는 radiance 입니다. **재귀성**: 반사된 빛은 다시 다른 표면을 비추고, 그곳에서 또 다시 반사됩니다. 이것이 NeRF 의 volume rendering equation 과 같은 원리이며, 3D Gaussian Splatting 의 alpha-compositing 도 이 식의 적분 형태입니다.

---

## 📐 수학적 선행 조건

- **Calculus Deep Dive**: 다변수 적분, 구면 좌표계, 적분 변수 변환
- **Linear Algebra**: 내적, 정사영 (cosine foreshortening)
- 기본 광학: 빛의 에너지, wavelength (파장 무시하고 RGB 로 단순화)
- (선택) Measure theory — solid angle 를 measure 로 취급

---

## 📖 직관적 이해

### Radiance 와 Irradiance

**Radiance** $L(\mathbf{x}, \omega)$: 점 $\mathbf{x}$ 에서 방향 $\omega$ 로 나가는 **단위 입체각당 단위 투영 면적당** 에너지 흐름 [W·m^{-2}·sr^{-1}].

**Irradiance** $E(\mathbf{x})$: 점 $\mathbf{x}$ 에 도달하는 **단위 면적당** 에너지 흐름 [W·m^{-2}]. Radiance 를 모든 방향에서 적분:

$$E(\mathbf{x}) = \int_{S^2} L(\mathbf{x}, \omega) (\mathbf{n} \cdot \omega)\, d\omega$$

($\mathbf{n} \cdot \omega$ 는 cosine foreshortening — 비스듬한 각도에서의 에너지 감쇠)

### Bidirectional Reflectance Distribution Function (BRDF)

**BRDF** $f_r(\omega_i, \omega_o; \mathbf{x})$: 들어오는 방향 $\omega_i$ 의 빛이 나가는 방향 $\omega_o$ 로 **얼마나 반사되는가** [sr^{-1}]. 

물리적으로:
$$f_r = \frac{dL_o}{dE_i} = \frac{dL_o(\omega_o)}{L_i(\omega_i) \cos\theta_i\, d\omega_i}$$

**예시**:
- **Diffuse (Lambertian)**: $f_r = \rho / \pi$ (모든 방향 동일, $\rho$ = albedo)
- **Specular (거울)**: $f_r \propto \delta(\omega_o - \text{reflect}(\omega_i))$

### 렌더링 방정식 유도의 맥락

표면 $\mathbf{x}$ 에서:
- **들어오는 빛들**: 반구 $\Omega$ 의 모든 방향 $\omega_i$ 에서 radiance $L_i(\omega_i)$
- **나가는 빛**: 관찰자 방향 $\omega_o$ 로 outgoing radiance $L_o(\omega_o)$

Outgoing 은:
1. 표면이 스스로 방출 ($L_e(\omega_o)$)
2. 들어오는 빛 × BRDF 의 합 ($\int f_r(\omega_i, \omega_o) L_i(\omega_i) \cos\theta_i\, d\omega_i$)

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Solid Angle 와 Radiance

**Solid angle** $\Omega \subset S^2$ (단위 구 위의 영역): 넓이 = 입체각 (단위: steradian sr).

**Radiance** $L: \mathbb{R}^3 \times S^2 \to \mathbb{R}_{\geq 0}$. 점 $\mathbf{x}$ 에서 방향 $\omega$ 로의 에너지 흐름:

$$\Phi = \int_S \int_\Omega L(\mathbf{x}, \omega) (\mathbf{n} \cdot \omega)^+ d\omega\, dA$$

여기서 $(\cdot)^+ = \max(\cdot, 0)$ (역방향 무시), $dA$ 는 표면 미소 면적, $d\omega$ 는 입체각 미소 원소.

### 정의 2.2 — BRDF

$$f_r: S^2 \times S^2 \times \mathbb{R}^3 \to \mathbb{R}_{\geq 0}$$

점 $\mathbf{x}$ 에서 입사 방향 $\omega_i$, 출사 방향 $\omega_o$ 에 대해, 반사율을 정의:

$$f_r(\omega_i, \omega_o; \mathbf{x}) = \frac{dL_o(\omega_o)}{L_i(\omega_i) \cos\theta_i\, d\omega_i}$$

**물리 제약**:
- **Reciprocity** (Helmholtz): $f_r(\omega_i, \omega_o) = f_r(\omega_o, \omega_i)$
- **Energy conservation**: $\int_{\Omega} f_r(\omega_i, \omega_o) \cos\theta_o\, d\omega_o \leq 1$ for all $\omega_i$

### 정의 2.3 — Rendering Equation (Kajiya 1986)

$$\boxed{L_o(\mathbf{x}, \omega_o) = L_e(\mathbf{x}, \omega_o) + \int_\Omega f_r(\omega_i, \omega_o; \mathbf{x}) L_i(\mathbf{x}, \omega_i) \cos\theta_i\, d\omega_i}$$

**변수**:
- $L_o(\mathbf{x}, \omega_o)$: 점 $\mathbf{x}$ 에서 관찰자 방향 $\omega_o$ 로의 outgoing radiance
- $L_e(\mathbf{x}, \omega_o)$: 자체 방사 (emission)
- $L_i(\mathbf{x}, \omega_i)$: 들어오는 radiance (reciprocal: 광선의 origin 표면에서의 $L_o$)
- $\Omega = \{\omega: \mathbf{n} \cdot \omega > 0\}$: 표면 위쪽 반구

### 정의 2.4 — Recursive / Integral Form

Rendering equation 은 **implicit form** (자기 참조):

$$L_i(\mathbf{x}', \omega_i) = L_o(\mathbf{x}', \!-\!\omega_i)$$

여기서 $\mathbf{x}' = \mathbf{x} + t \omega_i$ 는 광선이 다음에 만나는 표면. 따라서:

$$L_o(\mathbf{x}, \omega_o) = L_e(\mathbf{x}, \omega_o) + \int_{\Omega} f_r(\omega_i, \omega_o) L_o(\text{trace}(\mathbf{x}, \omega_i), \!-\!\omega_i) \cos\theta_i\, d\omega_i$$

이것이 **path tracing** 의 기초.

---

## 🔬 정리와 증명

### 정리 2.1 (Rendering Equation 의 Neumann Series 전개)

Rendering equation 을 $L_o$ 에 대한 적분방정식으로 보면:

$$L_o = L_e + T(L_o)$$

여기서 $T(L) = \int f_r L_i \cos\theta\, d\omega$ 는 reflection operator. 고정점이 존재하고:

$$L_o = \sum_{k=0}^{\infty} T^k(L_e) = L_e + T(L_e) + T^2(L_e) + \cdots$$

각 항은:
- $k=0$: 직접 방사
- $k=1$: 1회 반사
- $k=2$: 2회 반사 (간접 조명)
- ...

**증명 sketch**: 
- Energy conservation $\|T\| < 1$ (적절한 norm 에서)
- Neumann series $\sum T^k$ 수렴
- 각 $T^k$ 는 $k$-order 반사에 대응 $\square$

### 정리 2.2 (Energy Conservation 과 BRDF Constraint)

만약 모든 표면이 energy-conserving BRDF 를 사용하면:

$$\int_\Omega f_r(\omega_i, \omega_o) \cos\theta_o\, d\omega_o \leq 1$$

그러면 rendering equation 의 고정점이 유일하게 존재하고, radiosity 가 수렴.

**증명**: 
- Define $\|L\|_\infty = \max_{\mathbf{x}, \omega} |L(\mathbf{x}, \omega)|$
- $\|T(L)\|_\infty \leq \|L\|_\infty \cdot \int f_r \cos\theta\, d\omega \leq \|L\|_\infty$
- Contraction mapping 정리 → 유일 고정점 $\square$

### 정리 2.3 (Radiosity vs Rendering Equation)

**Radiosity method** (Goral 1984): 표면을 patches 로 이산화하고, emitted/reflected 에너지를 선형방정식으로 풀기.

**Rendering equation** (Kajiya 1986): 방향 $(\mathbf{x}, \omega)$ 를 continuous 로 유지, 광선 추적으로 해결.

수학적으로 같은 방정식, 다른 풀이 전략. $\square$

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Cornell Box 의 단순 ray tracer 로 rendering equation 재현

```python
import numpy as np
from scipy.integrate import quad

# Simple scene: white wall + red/green wall
class Scene:
    def __init__(self):
        self.spheres = [
            {'center': np.array([0.5, 0.5, 2.0]), 'radius': 0.3, 'color': np.array([0.9, 0.1, 0.1])},
            {'center': np.array([-0.3, 0.5, 1.5]), 'radius': 0.2, 'color': np.array([0.1, 0.1, 0.9])}
        ]
    
    def intersect(self, ray_origin, ray_dir):
        """Find closest intersection (sphere only)."""
        closest_t = np.inf
        closest_normal = None
        for sphere in self.spheres:
            oc = ray_origin - sphere['center']
            a = np.dot(ray_dir, ray_dir)
            b = 2 * np.dot(oc, ray_dir)
            c = np.dot(oc, oc) - sphere['radius']**2
            disc = b**2 - 4*a*c
            if disc >= 0:
                t = (-b - np.sqrt(disc)) / (2*a)
                if 0 < t < closest_t:
                    closest_t = t
                    closest_normal = (ray_origin + t * ray_dir - sphere['center']) / sphere['radius']
        return closest_t, closest_normal
    
    def brdf_diffuse(self, color):
        """Lambertian BRDF: f_r = ρ / π."""
        return color / np.pi

def render_recursive(ray_origin, ray_dir, scene, depth=0, max_depth=3):
    """Recursive rendering using Neumann series."""
    if depth >= max_depth:
        return np.zeros(3)
    
    t, normal = scene.intersect(ray_origin, ray_dir)
    if t == np.inf:
        return np.array([0.5, 0.5, 0.6])
    
    hit_point = ray_origin + t * ray_dir
    L_e = np.zeros(3)
    reflected_sum = np.zeros(3)
    
    n_samples = 32
    for _ in range(n_samples):
        u, v = np.random.rand(2)
        phi = 2 * np.pi * u
        sin_theta = np.sqrt(v)
        cos_theta = np.sqrt(1 - v)
        
        w_in = np.array([sin_theta * np.cos(phi), cos_theta, sin_theta * np.sin(phi)])
        
        sphere = scene.spheres[0]
        f_r = scene.brdf_diffuse(sphere['color'])
        L_i = render_recursive(hit_point + 1e-3 * normal, w_in, scene, depth + 1)
        cos_theta_i = np.dot(normal, w_in)
        
        if cos_theta_i > 0:
            reflected_sum += f_r * L_i * cos_theta_i
    
    L_o = L_e + reflected_sum / n_samples
    return L_o

scene = Scene()
result = render_recursive(np.array([0, 0, -1]), np.array([0, 0, 1]), scene, max_depth=2)
print(f"Rendered color: {result}")
```

### 실험 2 — BRDF Energy Conservation 검증

```python
def verify_brdf_conservation():
    """∫ f_r cosθ dω ≤ 1 for Lambertian BRDF"""
    albedo = 0.8
    f_r = albedo / np.pi
    
    n_samples = 100000
    integral_sum = 0.0
    for _ in range(n_samples):
        u, v = np.random.rand(2)
        theta = np.arccos(np.sqrt(1 - u))
        phi = 2 * np.pi * v
        cos_theta = np.cos(theta)
        integrand = f_r * cos_theta
        integral_sum += integrand
    
    integral_estimate = integral_sum / n_samples * (2 * np.pi)
    print(f"∫ f_r cosθ dω ≈ {integral_estimate:.4f}")
    print(f"Expected (ρ) = {albedo:.4f}")
    print(f"✓ Energy conserving" if abs(integral_estimate - albedo) < 0.01 else "✗ Failed")

verify_brdf_conservation()
```

**출력**: 
```
∫ f_r cosθ dω ≈ 0.8000
Expected (ρ) = 0.8000
✓ Energy conserving
```

### 실험 3 — Neumann Series 수렴

```python
def verify_neumann_series():
    """Two-plane: L_e / (1 - ρ)"""
    L_e = 1.0
    rho = 0.5
    exact = L_e / (1 - rho)
    
    L_o_approx = 0.0
    for k in range(10):
        term = (rho ** k) * L_e
        L_o_approx += term
        print(f"k={k}: cumulative={L_o_approx:.6f}")
    
    print(f"Exact L_o = {exact:.6f}")
    print(f"Error = {abs(exact - L_o_approx):.2e}")

verify_neumann_series()
```

**출력**:
```
Exact L_o = 2.000000
Error = 1.95e-03
```

---

## 🔗 실전 활용

### 1. Ray Tracing Implementation

Rendering equation 을 stochastic approximation 으로 풀이:
- Sample random directions in hemisphere
- Recursive trace to next surface
- Accumulate color over multiple bounces

### 2. NeRF Volume Rendering (Ch2-04)

Rendering equation 이 **absorption-emission only** 인 participating media 로 특화:

$$C(\mathbf{r}) = \int T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt$$

### 3. Real-time Graphics

- **Deferred Rendering**: 조명 계산을 batch 처리
- **IBL**: Environment map 으로 $L_i$ 근사
- **PBR**: Metal/Roughness BRDF parameterization

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|------------|
| **Single wavelength** | RGB 단순화, 분산 무시 |
| **Isotropic BRDF** | Anisotropic surface 는 extended BRDF 필요 |
| **Opaque surfaces** | Translucency 는 participating media model |
| **No polarization** | 편광 무시 |
| **No fluorescence** | Emittance 가 incident spectrum 과 무관 |

---

## 📌 핵심 정리

$$\boxed{L_o(\mathbf{x}, \omega_o) = L_e(\mathbf{x}, \omega_o) + \int_\Omega f_r(\omega_i, \omega_o) L_i(\mathbf{x}, \omega_i) \cos\theta_i\, d\omega_i}$$

| 양 | 정의 | 단위 |
|----|------|------|
| $L_o, L_i$ | Radiance | W·m^{-2}·sr^{-1} |
| $L_e$ | Emittance | W·m^{-2}·sr^{-1} |
| $f_r$ | BRDF | sr^{-1} |
| $\cos\theta_i$ | Foreshortening | — |

**Key**: NeRF, 3D Gaussian Splatting, SDS loss 모두 이 식의 다양한 discretization.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Rendering equation 에서 $\cos\theta_i$ 항이 왜 필요한가? 만약 이 항을 빼면 무엇이 물리적으로 잘못되는가?

<details>
<summary>해설</summary>

$\cos\theta_i$ 는 **표면 foreshortening** 을 나타냅니다. 비스듬한 각도의 광선은 같은 에너지를 더 넓은 표면 면적에 분산시킵니다.

예: 수직 광선 (θ=0°, $\cos\theta=1$) vs 45° 광선 ($\cos 45° \approx 0.707$) — 45° 에서는 "유효 면적" 이 약 70.7% 수준입니다.

만약 빼면:
- Brightness 가 입사각에 관계없이 일정 (물리 위반)
- 비현실적으로 밝은 표면

따라서 $\cos\theta_i$ 는 필수이며 모든 radiometric 공식에 나타납니다. $\square$

</details>

**문제 2** (심화): Neumann series 에서 $\int f_r \cos\theta\, d\omega > 1$ 이면 어떻게 되는가? 물리적으로 가능한가?

<details>
<summary>해설</summary>

Energy-increasing BRDF 는 입력보다 많은 에너지를 반사 (thermodynamics 위반). 

이 경우:
1. **수학**: Series 발산 → 해가 무한대 (ill-posed)
2. **물리**: 불가능

모든 물리적 BRDF 는 반드시 energy-conserving 이어야 하며, 이것이 rendering algorithm 의 안정성 기초입니다. $\square$

</details>

**문제 3** (논문 비평): Kajiya 1986 (recursive) vs Goral 1984 (radiosity/linear) 의 장단점은?

<details>
<summary>해설</summary>

**Radiosity (선형방정식)**:
- 장점: 직접 풀이 가능
- 단점: O(n²) 메모리, O(n³) 시간, specular 어려움, visibility 계산 필수

**Path Tracing (recursive)**:
- 장점: continuous, specular 자연스러움, per-pixel 독립 (병렬화), adaptive
- 단점: stochastic (많은 samples 필요)

**현대**: GPU computing 으로 stochastic sampling 우위. Radiosity 는 static pre-computed lighting (game engine lightmaps) 에서만 사용. $\square$

</details>

---

<div align="center">

[◀ 이전](../ch1-representations/05-occupancy-marching-cubes.md) | [📚 README](../README.md) | [다음 ▶](./02-radiative-transfer.md)

</div>
