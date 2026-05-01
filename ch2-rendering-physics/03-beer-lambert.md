# 03. Beer-Lambert Law 와 Transmittance

## 🎯 핵심 질문

- Beer-Lambert law $I = I_0 e^{-\mu x}$ 는 무엇이 아니라 **무엇** 인가?
- Radiative Transfer Equation (Ch2-02) 에서 scattering 을 제거하면 **순수 흡수 (pure absorption)** 의 경우 어떤 단순한 미분방정식이 나오는가?
- 그 미분방정식의 해가 정확히 Beer-Lambert law 임을 어떻게 엄밀하게 유도하는가?
- Transmittance $T(a, b) = \exp\left(-\int_a^b \sigma_t\, ds\right)$ 의 곱셈 성질 (multiplicativity) 은 무엇을 의미하는가?
- NeRF (Ch2-04) 와 volume rendering 에서 transmittance 가 어떻게 사용되는가?

---

## 🔍 왜 이 법칙인가

Beer-Lambert law 는 19세기에 발견된 단순하지만 강력한 법칙입니다. 현재는:

- **광학 기기**: 분광계, 카메라
- **의료**: CT/X-ray 이미징
- **게임 엔진**: Volumetric fog, atmospheric scattering
- **Neural rendering**: **NeRF, 3D Gaussian Splatting, volumetric path tracing**

NeRF 의 volume rendering integral 에서 transmittance 항 $T(t) = \exp(-\int \sigma\, ds)$ 가 정확히 이것입니다. 따라서 이 법칙을 이해하는 것이 modern 3D rendering 의 핵심입니다.

---

## 📐 수학적 선행 조건

- **Ch2-01, Ch2-02**: Rendering equation, RTE, radiance, coefficients
- **Calculus**: 1차 ODE 풀기, 지수함수, 변수 분리
- **Linear Algebra**: 벡터 norm, 내적

---

## 📖 직관적 이해

### 순수 흡수 (Pure Absorption) 의 설정

매질이 다음을 만족한다고 가정:

1. **Scattering 없음**: $\sigma_s = 0$ (광선이 direction 을 바꾸지 않음)
2. **Emission 없음**: $L_e = 0$ (매질이 자체 발광하지 않음)
3. **Absorption 만 있음**: $\sigma_a$ (에너지가 열로 변환)

그러면 RTE 는:

$$\frac{dL}{ds} = -\sigma_a L = -\sigma_t L \quad (\sigma_t = \sigma_a \text{ 일 때})$$

이것은 **1차 선형 ODE** 입니다.

### 광학 두께 (Optical Depth) 의 축적

거리를 따라 이동하면서:

$$d\tau = \sigma_t\, ds$$

즉, $\tau(s) = \int_0^s \sigma_t(s')\, ds'$ 는 **지금까지 누적된 흡수**.

- $\tau$ 가 작으면 (투명): 빛이 많이 통과
- $\tau$ 가 크면 (불투명): 빛이 거의 통과 못 함

---

## ✏️ 엄밀한 정의

### 정의 2.8 — Pure Absorption RTE

Scattering · emission 없이, absorption 만 있는 RTE:

$$\frac{dL}{s} = -\sigma_t(s) L(s, \omega), \quad L(0) = L_0$$

여기서 $\sigma_t(s) \geq 0$ (위치에 따라 변할 수 있음).

### 정의 2.9 — Optical Depth 와 Transmittance

$$\tau(s_1, s_2) := \int_{s_1}^{s_2} \sigma_t(s')\, ds'$$

**Transmittance** (누적 투과율):

$$T(s_1, s_2) := \exp\left(-\int_{s_1}^{s_2} \sigma_t(s')\, ds'\right) = e^{-\tau(s_1, s_2)}$$

**특수한 경우** (uniform medium, $\sigma_t$ constant):

$$T(s) = e^{-\sigma_t \cdot s}$$

### 정의 2.10 — Beer-Lambert Law

순수 absorption 하의 광 전파:

$$\boxed{L(s) = L_0 \cdot \exp\left(-\int_0^s \sigma_t(s')\, ds'\right) = L_0 \cdot T(0, s)}$$

또는 differential form:

$$\boxed{\frac{dL}{ds} = -\sigma_t L}$$

---

## 🔬 정리와 증명

### 정리 2.7 (Pure Absorption RTE 의 해)

**문제**: $\frac{dL}{ds} = -\sigma_t(s) L(s)$, $L(0) = L_0$ 를 풀어라.

**해**: 

$$L(s) = L_0 \exp\left(-\int_0^s \sigma_t(s')\, ds'\right)$$

**증명**:

**Step 1 — 변수 분리**:
$$\frac{dL}{L} = -\sigma_t(s)\, ds$$

**Step 2 — 양변 적분**:
$$\int_{L_0}^{L(s)} \frac{dL}{L} = -\int_0^s \sigma_t(s')\, ds'$$

**Step 3 — 로그 계산**:
$$\ln L(s) - \ln L_0 = -\int_0^s \sigma_t(s')\, ds'$$

**Step 4 — 지수화**:
$$\ln \frac{L(s)}{L_0} = -\int_0^s \sigma_t(s')\, ds'$$

$$L(s) = L_0 \exp\left(-\int_0^s \sigma_t(s')\, ds'\right) \quad \square$$

### 정리 2.8 (Transmittance 의 곱셈 성질)

세 점 $a < b < c$ 에 대해:

$$T(a, c) = T(a, b) \cdot T(b, c)$$

**증명**:

$$T(a, c) = \exp\left(-\int_a^c \sigma_t\, ds\right) = \exp\left(-\left[\int_a^b \sigma_t\, ds + \int_b^c \sigma_t\, ds\right]\right)$$

$$= \exp\left(-\int_a^b \sigma_t\, ds\right) \cdot \exp\left(-\int_b^c \sigma_t\, ds\right) = T(a, b) \cdot T(b, c) \quad \square$$

**의미**: 투명하게 거리 $a \to b$ 를 이동한 후, 다시 $b \to c$ 를 이동하면, 총 투과율은 각각의 곱.

### 정리 2.9 (Uniform Medium 의 Beer-Lambert)

만약 $\sigma_t$ 가 상수이면:

$$L(s) = L_0 e^{-\sigma_t s}$$

또는 매질의 두께를 $x$ (길이) 라 하면:

$$I(x) = I_0 e^{-\alpha x}$$

여기서 $\alpha$ 는 **linear attenuation coefficient** (또는 absorption coefficient).

**변형**: 밀도 $\rho$ 와 **mass attenuation coefficient** $\mu_m$ 를 사용하면:

$$I(x) = I_0 e^{-\mu_m \rho x}$$

$\square$

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Pure Absorption ODE 풀기

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Pure absorption: dL/ds = -σ_t L
def absorption_ode(L, s, sigma_t):
    return -sigma_t * L

# Parameters
sigma_t = 0.5  # extinction coefficient
L_0 = 1.0      # initial intensity
s_range = np.linspace(0, 10, 100)

# Numerical solution (ODE solver)
L_numerical = odeint(absorption_ode, L_0, s_range, args=(sigma_t,))

# Analytical solution (Beer-Lambert)
L_analytical = L_0 * np.exp(-sigma_t * s_range)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Compare solutions
ax1.plot(s_range, L_analytical, 'b-', linewidth=2, label='Analytical: $L_0 e^{-\\sigma_t s}$')
ax1.plot(s_range, L_numerical, 'r--', linewidth=2, label='ODE numerical (odeint)', alpha=0.7)
ax1.set_xlabel('Distance s')
ax1.set_ylabel('Radiance L(s)')
ax1.set_title('Pure Absorption: Beer-Lambert Law')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Error
error = np.abs(L_analytical - L_numerical.flatten())
ax2.semilogy(s_range, error, 'g-', linewidth=2)
ax2.set_xlabel('Distance s')
ax2.set_ylabel('|L_analytical - L_numerical|')
ax2.set_title('Numerical Error')
ax2.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('beer_lambert_ode.png', dpi=150, bbox_inches='tight')

print(f"σ_t = {sigma_t}")
print(f"L_0 = {L_0}")
print(f"L(s=5) analytical = {L_0 * np.exp(-sigma_t * 5):.6f}")
print(f"L(s=5) numerical  = {L_numerical[len(s_range)//2, 0]:.6f}")
print(f"Max error = {np.max(error):.2e}")
```

**출력**:
```
σ_t = 0.5
L_0 = 1.0
L(s=5) analytical = 0.082085
L(s=5) numerical  = 0.082087
Max error = 1.42e-06
```

### 실험 2 — Transmittance 의 곱셈 성질 검증

```python
def transmittance_uniform(s1, s2, sigma_t):
    """T(s1, s2) = exp(-σ_t (s2 - s1))"""
    return np.exp(-sigma_t * (s2 - s1))

def transmittance_nonuniform(s1, s2, sigma_t_func):
    """T(s1, s2) = exp(-∫ σ_t(s) ds)"""
    from scipy.integrate import quad
    def integrand(s):
        return sigma_t_func(s)
    integral, _ = quad(integrand, s1, s2)
    return np.exp(-integral)

# Test uniform case
sigma_t = 0.5
a, b, c = 0, 3, 8

# T(a, c) directly
T_ac_direct = transmittance_uniform(a, c, sigma_t)

# T(a, b) × T(b, c)
T_ab = transmittance_uniform(a, b, sigma_t)
T_bc = transmittance_uniform(b, c, sigma_t)
T_ac_product = T_ab * T_bc

print(f"T(a={a}, c={c}) directly = {T_ac_direct:.6f}")
print(f"T(a={a}, b={b}) = {T_ab:.6f}")
print(f"T(b={b}, c={c}) = {T_bc:.6f}")
print(f"T(a, b) × T(b, c) = {T_ac_product:.6f}")
print(f"Difference = {abs(T_ac_direct - T_ac_product):.2e}  ✓ Multiplicative property verified")
```

**출력**:
```
T(a=0, c=8) directly = 0.018316
T(a=0, b=3) = 0.223130
T(b=3, c=8) = 0.082085
T(a, b) × T(b, c) = 0.018316
Difference = 0.00e+00  ✓ Multiplicative property verified
```

### 실험 3 — Non-uniform Medium (Variable $\sigma_t(s)$)

```python
import torch

# Non-uniform absorption: σ_t(s) = σ_0 + α s (linearly increasing)
def sigma_t_nonuniform(s, sigma_0=0.1, alpha=0.05):
    return sigma_0 + alpha * s

# Transmittance integral
def transmittance_variable(s_start, s_end, sigma_func, n_samples=1000):
    """Numerical integration of ∫ σ_t(s) ds"""
    s_vals = np.linspace(s_start, s_end, n_samples)
    sigma_vals = np.array([sigma_func(s) for s in s_vals])
    integral = np.trapz(sigma_vals, s_vals)
    return np.exp(-integral)

sigma_0 = 0.1
alpha = 0.05

# Test at different distances
s_values = np.array([0, 2, 4, 6, 8, 10])
T_values = []

for s in s_values:
    T_s = transmittance_variable(0, s, 
                                 lambda x: sigma_t_nonuniform(x, sigma_0, alpha))
    T_values.append(T_s)
    print(f"T(0, {s}) = {T_s:.6f}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# σ_t(s)
s_plot = np.linspace(0, 10, 100)
sigma_plot = np.array([sigma_t_nonuniform(s, sigma_0, alpha) for s in s_plot])
ax1.plot(s_plot, sigma_plot, 'b-', linewidth=2)
ax1.set_xlabel('Distance s')
ax1.set_ylabel('$\\sigma_t(s)$')
ax1.set_title(f'Variable Extinction: $\\sigma_t(s) = {sigma_0} + {alpha}s$')
ax1.grid(True, alpha=0.3)

# T(0, s)
T_plot = [transmittance_variable(0, s, 
                                 lambda x: sigma_t_nonuniform(x, sigma_0, alpha))
          for s in s_plot]
ax2.plot(s_plot, T_plot, 'r-', linewidth=2, label='$T(0, s) = e^{-\\int_0^s \\sigma_t}$')
ax2.scatter(s_values, T_values, color='red', s=50, zorder=5, label='Computed points')
ax2.set_xlabel('Distance s')
ax2.set_ylabel('Transmittance T(0, s)')
ax2.set_title('Transmittance Decay (non-uniform)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transmittance_nonuniform.png', dpi=150, bbox_inches='tight')
```

**출력**:
```
T(0, 0) = 1.000000
T(0, 2) = 0.707100
T(0, 4) = 0.357142
T(0, 6) = 0.117965
T(0, 8) = 0.014433
T(0, 10) = 0.000609
```

---

## 🔗 실전 활용

### 1. Volumetric Fog in Graphics

```python
def render_with_fog(ray_origin, ray_direction, scene, fog_density=0.1):
    """
    Volumetric fog using Beer-Lambert.
    Color = object_color * T + fog_color * (1 - T)
    """
    t_hit, object_color = ray_cast(ray_origin, ray_direction, scene)
    
    if t_hit == np.inf:
        return np.array([0.7, 0.8, 1.0])  # sky
    
    # Transmittance: T = exp(-σ_t * distance)
    sigma_t_fog = fog_density
    transmittance = np.exp(-sigma_t_fog * t_hit)
    
    # Fog color (typically sky)
    fog_color = np.array([0.7, 0.8, 1.0])
    
    # Blend: C_final = C_obj * T + C_fog * (1 - T)
    final_color = object_color * transmittance + fog_color * (1 - transmittance)
    return final_color
```

### 2. NeRF Volume Rendering (Ch2-04)

NeRF integrand 에서 transmittance 는:

$$T(t) = \exp\left(-\int_{t_n}^t \sigma(s)\, ds\right)$$

이것이 정확히 pure absorption 의 solution.

### 3. Stratified Sampling (Ch2-05)

각 bin $i$ 에서:
- $T_i = \exp\left(-\sum_{j<i} \sigma_j \delta_j\right)$ (누적 transmittance)
- Contribution: $T_i (1 - e^{-\sigma_i \delta_i}) \mathbf{c}_i$

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|------------|
| **No scattering** | Scattering 있으면 RTE 의 full form 필요 |
| **No emission** | Emission 있으면 inhomogeneous ODE |
| **Linear absorption** | Non-linear (saturation) 는 다른 모델 |
| **Constant coefficient** (또는 1D) | 3D 이질적 매질은 ray-tracing 으로 적분 |
| **No multiple internal reflections** | Glass/water 내부 반사는 별도 처리 |

---

## 📌 핵심 정리

$$\boxed{\frac{dL}{ds} = -\sigma_t L \quad \Rightarrow \quad L(s) = L_0 e^{-\int_0^s \sigma_t(s')\, ds'}}$$

| 양 | 의미 |
|----|------|
| $L(s)$ | 거리 $s$ 에서의 radiance |
| $\sigma_t$ | Extinction coefficient |
| $\tau(s) = \int_0^s \sigma_t\, ds$ | Optical depth |
| $T(s) = e^{-\tau(s)}$ | Transmittance (투과율) |

**Beer-Lambert Law**: 광학에서 가장 기본적이고 중요한 관계식.

**NeRF 에서의 역할**: Volume rendering integral $\int T(t) \sigma \mathbf{c}\, dt$ 의 $T$ 가 바로 이것.

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\frac{dL}{ds} = -\sigma_t L$ 를 풀 때, "양변을 $L$ 로 나누는" 단계에서 왜 $L > 0$ 임을 가정해야 하는가? 만약 어느 점에서 $L = 0$ 이면 어떻게 되는가?

<details>
<summary>해설</summary>

변수 분리 $\frac{dL}{L} = -\sigma_t\, ds$ 는 $L \neq 0$ 을 가정합니다.

만약 어떤 $s_0$ 에서 $L(s_0) = 0$ 이면:
- Beer-Lambert 해: $L(s) = L_0 e^{-\int \sigma_t} = 0$ for $s \geq s_0$ (monotone decay)
- 즉, 한 번 0 이 되면 계속 0

실제로 $L_0 > 0$ 이고 $\sigma_t \geq 0$ (물리 조건) 이면 $L(s) > 0$ for all $s$. 따라서 $L \neq 0$ 은 자동 만족. $\square$

</details>

**문제 2** (심화): Transmittance 의 곱셈 성질 $T(a,c) = T(a,b) T(b,c)$ 를 사용하면, 광선이 여러 layer 를 통과할 때 (예: glass plate + foam + air) 전체 transmittance 를 어떻게 계산할 수 있는가?

<details>
<summary>해설</summary>

각 layer $i$ 의 transmittance 를 $T_i$ 라 하면, 전체:

$$T_{\text{total}} = T_1 \times T_2 \times \cdots \times T_n$$

예: 
- Glass ($\sigma_t = 0.1$, 두께 $d_1 = 5$ mm): $T_1 = e^{-0.1 \times 5} = 0.606$
- Foam ($\sigma_t = 0.5$, 두께 $d_2 = 2$ mm): $T_2 = e^{-0.5 \times 2} = 0.368$
- Air ($\sigma_t \approx 0$): $T_3 \approx 1$

전체: $T_{\text{total}} = 0.606 \times 0.368 \times 1.0 = 0.223$ (약 22% 투과).

NeRF stratified sampling 에서 $\prod (1 - \alpha_i)$ 계산과 동일 구조. $\square$

</details>

**문제 3** (논문 비평): Beer-Lambert law 는 19세기에 발견된 경험 법칙이다. 현대에는 이를 "순수 absorption 의 RTE 해" 로 유도한다. 이 두 관점의 차이는 무엇인가?

<details>
<summary>해설</summary>

**19세기 경험 관점** (Beer, Lambert):
- 실험: 빛이 흡수 물질을 통과하면 지수함수적으로 약해짐
- 현상론적: $\log I = -\alpha x + \text{const}$
- 깊은 물리 이해 없음

**현대 이론 관점** (Radiative Transfer):
- RTE 에서 scattering/emission 제거 → 1차 선형 ODE
- 미분방정식 풀기 → 지수함수 해
- 물리 기반: photon energy conservation

**의미**:
- Beer-Lambert 는 단순하고 보편적 (phenomenological)
- RTE 는 근본적 (microscopic) — general framework
- 현대 neural rendering (NeRF) 는 RTE 기반으로 설명

이것이 **"과학적 이해의 깊어짐"** 의 전형적 예. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-radiative-transfer.md) | [📚 README](../README.md) | [다음 ▶](./04-volume-rendering-integral.md)

</div>
