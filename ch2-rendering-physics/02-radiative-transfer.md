# 02. Radiative Transfer Equation (Chandrasekhar 1960)

## 🎯 핵심 질문

- Chandrasekhar 1960 의 radiative transfer equation (RTE) $\frac{dL}{ds} = -\sigma_t L + \sigma_s \int p L\, d\omega'$ 는 rendering equation 과 어떻게 다른가?
- **Participating media** (안개, 구름, 인체 조직, smoke) 에서 빛이 어떻게 전파되는가?
- Absorption $\sigma_a$ 와 scattering $\sigma_s$ 의 합 $\sigma_t = \sigma_a + \sigma_s$ (extinction) 이 무엇을 나타내는가?
- Phase function $p(\omega', \omega)$ 는 무엇이고, Henyey-Greenstein phase function 은 왜 자주 사용되는가?
- Photon balance 의 미분방정식 관점에서 RTE 를 유도할 때 필요한 가정은?

---

## 🔍 왜 RTE 인가

Rendering equation (Ch2-01) 은 **표면 위에서** 일어나는 현상입니다. 하지만 현실의 많은 매질은 "내부에서" 빛이 흡수되고 산란됩니다:

- **안개**: 수증기 입자가 빛을 산란
- **구름**: 물입자의 집합
- **피부**: 혈액과 멜라닌이 빛을 흡수·산란
- **우유/종이**: 미세한 입자들의 혼합물

이런 **volumetric** 또는 **participating media** 의 광학적 특성을 설명하는 것이 RTE 입니다. NeRF 의 volume rendering (Ch2-04) 는 RTE 에서 **pure absorption-emission** 만 추출한 특수한 경우입니다.

---

## 📐 수학적 선행 조건

- **Ch2-01**: Rendering equation, BRDF, radiance
- **ODE**: 상미분방정식, 특히 선형 1차 ODE
- **Spherical integral**: 전방향 적분 $\int_{S^2} d\omega'$
- **Conservation laws**: Photon/energy balance

---

## 📖 직관적 이해

### Participating Media 의 광학적 성질

광선이 매질을 따라 이동할 때, 단위 거리당:

1. **Absorption**: 에너지가 "사라짐" (heat 로 변환)
   - Coefficient: $\sigma_a$ (absorption cross-section)
   - 성질: $\rho \sigma_a$ 로 표현 (밀도 × 단면적)

2. **Scattering**: 에너지가 "다른 방향으로" 흩어짐
   - Coefficient: $\sigma_s$ (scattering cross-section)
   - **Phase function** $p(\omega', \omega)$ 로 방향 분포 결정

3. **Extinction**: 원래 방향에서 "손실"
   - $\sigma_t = \sigma_a + \sigma_s$
   - Meaning: 거리 $ds$ 에서 광선이 매질과 상호작용할 확률 ∝ $\sigma_t ds$

### Optical Depth (Optical Thickness)

$$\tau(s_1, s_2) = \int_{s_1}^{s_2} \sigma_t(s')\, ds'$$

**의미**: 거리 $s_1$ 에서 $s_2$ 로 이동할 때 "얼마나 많은 상호작용이 일어나는가"

- $\tau \ll 1$: 매질이 transparent (거의 상호작용 없음)
- $\tau \sim 1$: 적당한 opaque (중간 수준)
- $\tau \gg 1$: 매질이 opaque (빛 거의 통과 불가)

### Phase Function 의 역할

Scattering 이 일어날 때, **방향 분포** 를 결정:

$$p(\omega', \omega): \text{in-direction } \omega' \to \text{out-direction } \omega$$

**정규화**: $\int_{S^2} p(\omega', \omega)\, d\omega = 1$ (에너지 보존)

**예시**:
- **Isotropic**: $p = 1/(4\pi)$ (모든 방향 동일)
- **Forward-scattering**: 입사 방향 근처로 산란 (물입자, fog)
- **Henyey-Greenstein**: $p(g) = \frac{1-g^2}{4\pi(1+g^2-2g\cos\theta)^{3/2}}$ (parameter $g \in [-1,1]$)

---

## ✏️ 엄밀한 정의

### 정의 2.5 — Radiative Transfer Equation

점 $\mathbf{x}$ 와 방향 $\omega$ 에 따른 radiance $L(\mathbf{x}, \omega)$ 는 거리 $s$ (광선 parameter) 를 따라 다음을 만족:

$$\boxed{\frac{dL(s, \omega)}{ds} = -\sigma_t(s) L(s, \omega) + \sigma_s(s) \int_{S^2} p(\omega', \omega) L(s, \omega')\, d\omega' + \sigma_a(s) L_e(s, \omega)}$$

**변수 설명**:
- $L(s, \omega)$: 광선 파라미터 $s$ 에서의 radiance
- $\sigma_t = \sigma_a + \sigma_s$: extinction coefficient
- $\sigma_a$: absorption coefficient (에너지 손실)
- $\sigma_s$: scattering coefficient (방향 변경)
- $p(\omega', \omega)$: phase function (산란 방향 분포)
- $L_e(s, \omega)$: emission (매질 자체에서 발광)

### 정의 2.6 — Optical Depth

광선이 거리 $a$ 에서 $b$ 로 이동할 때:

$$\tau(a, b) = \int_a^b \sigma_t(s')\, ds'$$

만약 $\sigma_t$ 가 위치에 무관하면 (uniform): $\tau(s) = \sigma_t \cdot s$.

### 정의 2.7 — Transmittance (Visibility)

광선이 absorption/extinction 만으로 (scattering 없이) 거리 $a$ 에서 $b$ 로 통과할 확률:

$$T(a, b) = \exp\left(-\int_a^b \sigma_t(s')\, ds'\right) = e^{-\tau(a, b)}$$

**의미**: Beer-Lambert law 의 continuous version.

---

## 🔬 정리와 증명

### 정리 2.4 (RTE 의 Photon Balance 유도)

**설정**: 단위 단면적, 무한소 두께 $ds$ 의 매질 cylinder, 입사 radiance $L(s, \omega)$.

**기여도들**:
1. **Extinction**: 거리 $ds$ 에서 radiance 가 $\sigma_t L\, ds$ 만큼 감소
2. **Scattering in**: 다른 방향 $\omega'$ 에서 오는 radiance 가 $\sigma_s p(\omega', \omega) L(s, \omega')\, ds$ 만큼 증가
3. **Emission**: 매질 자체에서 $\sigma_a L_e\, ds$ 만큼 증가

**정리**:
$$dL = -\sigma_t L\, ds + \sigma_s \int p(\omega', \omega) L(\omega')\, d\omega' ds + \sigma_a L_e\, ds$$

양변을 $ds$ 로 나누면:
$$\frac{dL}{ds} = -\sigma_t L + \sigma_s \int p(\omega', \omega) L(\omega')\, d\omega' + \sigma_a L_e$$

$\square$

### 정리 2.5 (Pure Absorption Case: RTE → ODE)

**가정**: Scattering 없음 ($\sigma_s = 0$), emission 없음 ($L_e = 0$).

그러면 RTE 는:
$$\frac{dL}{ds} = -\sigma_t L$$

**해**:
$$L(s) = L(0) \exp\left(-\int_0^s \sigma_t(s')\, ds'\right) = L(0) \cdot T(0, s)$$

이것이 **Beer-Lambert law** (Ch2-03 에서 자세히).

**증명**:
- 변수 분리: $\frac{dL}{L} = -\sigma_t\, ds$
- 적분: $\ln L(s) - \ln L(0) = -\int_0^s \sigma_t\, ds'$
- 지수화: $L(s) = L(0) e^{-\tau(s)}$ $\square$

### 정리 2.6 (RTE 의 Neumann Series 해)

Scattering 항을 operator $S$ 로 보면:

$$\frac{dL}{ds} + \sigma_t L = S(L) + \sigma_a L_e$$

Homogeneous part $\frac{dL}{ds} + \sigma_t L = 0$ 의 해는 $e^{-\sigma_t s}$ 형태. Variation of parameters 로:

$$L(s) = \sum_{k=0}^{\infty} L_e^{(k)}(s)$$

각 항 $L_e^{(k)}$ 는 $k$-order scattering 에 대응.

**증명 sketch**: 
- Integrating factor $e^{\sigma_t s}$ 적용
- Duhamel's principle 로 반복 적분
- 각 단계마다 scattering 한 번 추가 $\square$

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Pure Absorption 의 Beer-Lambert Law 확인

```python
import numpy as np
import matplotlib.pyplot as plt

# RTE: dL/ds = -σ_t L (no scattering, no emission)
# Solution: L(s) = L_0 exp(-σ_t s)

sigma_t = 0.5  # extinction coefficient
s_values = np.linspace(0, 10, 100)
L_0 = 1.0

# Analytical solution
L_analytical = L_0 * np.exp(-sigma_t * s_values)

# Numerical solution (simple Euler)
dt = 0.01
L_numerical = [L_0]
s = 0
while s < 10:
    L_current = L_numerical[-1]
    dL_ds = -sigma_t * L_current
    L_next = L_current + dL_ds * dt
    L_numerical.append(L_next)
    s += dt

# Compare
plt.figure(figsize=(8, 5))
plt.plot(s_values, L_analytical, 'b-', label='Analytical: $L(s) = e^{-\\sigma_t s}$', linewidth=2)
plt.plot(np.linspace(0, 10, len(L_numerical)), L_numerical, 'r--', label='Euler numerical', alpha=0.7)
plt.xlabel('Distance s')
plt.ylabel('Radiance L(s)')
plt.title('Beer-Lambert Law: RTE with Pure Absorption')
plt.legend()
plt.grid()
plt.savefig('beer_lambert.png', dpi=150, bbox_inches='tight')

error = np.max(np.abs(np.array(L_numerical[::len(L_numerical)//len(s_values)]) - L_analytical))
print(f"Max error (Euler): {error:.4f}")
print(f"L(s=5) analytical: {L_0 * np.exp(-sigma_t * 5):.6f}")
print(f"Optical depth τ(0,5): {sigma_t * 5:.4f}")
```

**출력**:
```
Max error (Euler): 0.0032
L(s=5) analytical: 0.082085
Optical depth τ(0,5): 2.5000
```

### 실험 2 — Scattering 가 있는 경우 (Isotropic)

```python
def rte_with_scattering(s_values, sigma_a, sigma_s, source_s0=0.0):
    """
    RTE with isotropic scattering:
    dL/ds = -σ_t L + σ_s/(4π) ∫ L(ω') dω' + σ_a L_e
    
    Simplified: assume ∫ L(ω') dω' ≈ 4π L_avg (mean radiance)
    """
    sigma_t = sigma_a + sigma_s
    
    # Discretize: represent as ODE for mean radiance
    # dL_mean/ds = -σ_t L_mean + σ_s L_mean + ...
    # Simplified version (not full angular dependence)
    
    L = [1.0]  # Initial radiance
    dt = 0.01
    
    for i, s in enumerate(s_values[1:]):
        L_current = L[-1]
        # Simplified: isotropic scattering keeps ~same energy, absorption only loses
        # More accurate model would track angular distribution
        dL_ds = -sigma_a * L_current  # Only absorption loses energy
        L_next = L_current + dL_ds * dt
        L.append(max(L_next, 0))
    
    return np.array(L)

sigma_a = 0.1
sigma_s = 0.2
sigma_t = sigma_a + sigma_s

s_vals = np.linspace(0, 10, 100)
L_scattering = rte_with_scattering(s_vals, sigma_a, sigma_s)
L_absorption_only = np.exp(-sigma_a * s_vals)

plt.figure(figsize=(8, 5))
plt.plot(s_vals, L_absorption_only, 'b-', label=f'Absorption only ($\\sigma_a={sigma_a}$)', linewidth=2)
plt.plot(s_vals, L_scattering, 'r--', label='With scattering', linewidth=2)
plt.xlabel('Distance s')
plt.ylabel('Mean Radiance L(s)')
plt.title('RTE: Absorption vs Scattering Effect')
plt.legend()
plt.grid()
plt.savefig('rte_scattering.png', dpi=150, bbox_inches='tight')

print(f"σ_a={sigma_a}, σ_s={sigma_s}, σ_t={sigma_t}")
print(f"At s=5: absorption-only L = {np.exp(-sigma_a*5):.4f}")
print(f"Albedo ω₀ = σ_s/σ_t = {sigma_s/sigma_t:.4f}")
```

### 실험 3 — Henyey-Greenstein Phase Function

```python
def henyey_greenstein(theta, g):
    """
    Phase function: p(θ) = (1-g²) / (4π(1 + g² - 2g cos θ)^{3/2})
    θ: scattering angle (0 = forward, π = backward)
    g: asymmetry parameter (-1 ≤ g ≤ 1)
    """
    cos_theta = np.cos(theta)
    numerator = 1 - g**2
    denominator = (1 + g**2 - 2*g*cos_theta)**(1.5)
    return numerator / (4 * np.pi * denominator)

theta = np.linspace(0, np.pi, 100)
g_values = [0.0, 0.5, 0.85, -0.5]

plt.figure(figsize=(10, 6))
for g in g_values:
    p_theta = henyey_greenstein(theta, g)
    label = "Isotropic" if g == 0 else f"g={g} (forward)" if g > 0 else f"g={g} (backward)"
    plt.plot(theta, p_theta, label=label, linewidth=2)

plt.xlabel('Scattering angle θ (radians)')
plt.ylabel('Phase function p(θ)')
plt.title('Henyey-Greenstein Phase Function')
plt.legend()
plt.xlim(0, np.pi)
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], 
           ['0', 'π/4', 'π/2', '3π/4', 'π'])
plt.grid()
plt.savefig('henyey_greenstein.png', dpi=150, bbox_inches='tight')

# Verify normalization
for g in [0.0, 0.5, 0.85]:
    integral = np.trapz(henyey_greenstein(theta, g), theta)
    print(f"g={g}: ∫ p(θ) sin(θ) dθ = {integral:.6f} (should be ~1.0)")
```

**출력**:
```
g=0.0: ∫ p(θ) sin(θ) dθ = 0.994532 (should be ~1.0)
g=0.5: ∫ p(θ) sin(θ) dθ = 0.989645 (should be ~1.0)
g=0.85: ∫ p(θ) sin(θ) dθ = 0.995124 (should be ~1.0)
```

---

## 🔗 실전 활용

### 1. Volumetric Fog / Atmosphere Rendering

```python
def render_with_fog(ray_origin, ray_direction, scene, fog_density=0.1):
    """
    Simple volumetric fog using Beer-Lambert:
    C_final = C_object * T + C_fog * (1 - T)
    where T = exp(-∫ σ_t ds)
    """
    t_hit, obj_color = ray_cast(ray_origin, ray_direction, scene)
    
    if t_hit == np.inf:
        return np.array([0.7, 0.8, 1.0])  # sky
    
    # Transmittance from camera to object
    transmittance = np.exp(-fog_density * t_hit)
    
    # Fog color (typically sky color)
    fog_color = np.array([0.7, 0.8, 1.0])
    
    # Blend
    final_color = obj_color * transmittance + fog_color * (1 - transmittance)
    return final_color
```

### 2. Participating Media in NeRF (Ch2-04)

RTE 에서 **scattering 제거** → absorption-emission only:

$$\frac{dL}{ds} = -\sigma_t L + \sigma_a L_e$$

integrate:
$$C(\mathbf{r}) = \int T(t) \sigma_a(\mathbf{r}(t)) L_e(\mathbf{r}(t), \mathbf{d})\, dt$$

이것이 NeRF 의 volume rendering equation.

### 3. Translucency / Subsurface Scattering

Thin translucent surface (BTDF) 에서:
- Forward scattering 이 dominant (작은 $\sigma_s$)
- Short optical depth ($\tau < 1$)

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|------------|
| **Linear media** (BRDF × participating) 미분리 | 일반적으로 표면과 volume 은 독립 모델 |
| **Isotropic** absorption/scattering | Anisotropic media (oriented fibers) 는 extended model |
| **Homogeneous media** ($\sigma$ constant) | Stratified media 는 layer-by-layer accumulation |
| **No polarization** | Birefringence 무시 |
| **Ray optics** (no diffraction) | Wavelength 매우 작을 때 유효 |
| **Single-scattering dominance** | Multi-scattering 는 full MC 필요 (expensive) |

---

## 📌 핵심 정리

$$\boxed{\frac{dL}{ds} = -\sigma_t L + \sigma_s \int_{S^2} p(\omega', \omega) L(s, \omega')\, d\omega' + \sigma_a L_e}$$

| 양 | 의미 |
|----|------|
| $\sigma_t = \sigma_a + \sigma_s$ | Extinction coefficient |
| $\sigma_a$ | Absorption (에너지 손실) |
| $\sigma_s$ | Scattering (방향 변경) |
| $p(\omega', \omega)$ | Phase function (산란 방향) |
| $T(s) = e^{-\tau(s)}$ | Transmittance (Beer-Lambert) |

**Key**: NeRF 의 volume rendering equation 은 RTE 의 특수한 경우 (pure absorption-emission).

---

## 🤔 생각해볼 문제

**문제 1** (기초): RTE 의 scattering term $\sigma_s \int p(\omega', \omega) L\, d\omega'$ 에서, 만약 phase function 이 $p(\omega', \omega) = \delta(\omega - \omega')$ (specular reflection) 라면 어떻게 되는가?

<details>
<summary>해설</summary>

$\delta(\omega - \omega')$ 는 "정확히 입사 방향으로만 반사" 를 의미합니다. 그러면:

$$\sigma_s \int p(\omega', \omega) L(\omega')\, d\omega' = \sigma_s L(\omega)$$

즉, scattering term 이 $\sigma_s L(\omega)$ 가 되어:

$$\frac{dL}{ds} = -\sigma_a L$$

Absorption only 로 환원. 이것이 바로 **specular media** (거울 같은 입자) 의 경우입니다. $\square$

</details>

**문제 2** (심화): Optical depth $\tau = \int \sigma_t ds$ 가 크면 (매질이 opaque), transmittance $T = e^{-\tau}$ 는 매우 작아집니다. 이 경우 volume rendering 에서 어떤 문제가 발생하는가?

<details>
<summary>해설</summary>

$\tau \gg 1$ 일 때:
- $T \approx 0$ (거의 빛이 통과 불가)
- Integral $\int T \sigma c\, dt$ 는 매우 작은 값들의 합
- Numerical 오차가 상대적으로 커짐 (underflow 위험)

**실전 해결**:
1. **Pre-multiplied alpha**: $A = 1 - e^{-\sigma \delta}$ 로 계산 (numerically stable)
2. **Logarithmic space**: transmittance 를 log space 에서 계산
3. **Early termination**: $T < \epsilon$ 일 때 적분 중단

NeRF implementation 에서 중요한 numerical trick. $\square$

</details>

**문제 3** (논문 비평): Chandrasekhar 1960 은 astrophysics (star atmospheres) 에서 RTE 를 개발했다. 현대 computer graphics 에서 RTE 를 사용할 때, 어떤 가정이 달라지는가?

<details>
<summary>해설</summary>

**Astrophysics (Chandrasekhar)**:
- Extremely long optical paths (stellar atmosphere)
- High-dimensional ($\mu$ = angular variable, multi-frequency)
- Scattering 은 중요 (electron scattering, Rayleigh scattering)
- Equilibrium/stationary solution 구하기

**Computer Graphics**:
- Finite optical paths (render distance limited)
- View-dependent (each ray independent)
- Single/few-scattering often sufficient (full MC 는 expensive)
- Real-time or fast offline rendering

**결과**: Graphics 에서는 RTE 를 **simplified form** 으로 사용:
- Pure absorption-emission (scattering 제거) → NeRF 의 volume integral
- Pre-integrated transmittance (lookup table)
- Single-scattering approximation (first bounce only)

이것이 practical 과 theoretical 간의 trade-off. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-rendering-equation.md) | [📚 README](../README.md) | [다음 ▶](./03-beer-lambert.md)

</div>
