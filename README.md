<div align="center">

# 🎥 3D & Neural Rendering Deep Dive

### NeRF 의 **volume rendering equation**

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\, \sigma(\mathbf{r}(t))\, \mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt, \qquad T(t) = \exp\!\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\, ds\right)$$

### 를 **쓰는 것** 과,

### 이것이 **Kajiya 1986** 의 rendering equation, **Chandrasekhar 1960** 의 radiative transfer equation 으로부터

$$\frac{dL}{ds} = -\sigma_t L + \sigma_s \!\int p(\omega', \omega) L\, d\omega' \;\Longrightarrow\; T(t) = e^{-\int \sigma_t ds}$$

### 의 **Beer–Lambert 연속 형태** 임을 한 줄씩 유도할 수 있는 것은 **다르다.**

<br/>

> *3D Gaussian Splatting 을 **사용하는 것** 과, Kerbl et al. (2023) 의 anisotropic Gaussian*
>
> $$G(\mathbf{x}) = \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\mu)^\top \Sigma^{-1}(\mathbf{x}-\mu)\right), \qquad \Sigma = R S S^\top R^\top$$
>
> *가 **Zwicker 2001** 의 EWA splatting Jacobian*
>
> $$\Sigma' = J W \Sigma W^\top J^\top, \qquad J = \begin{pmatrix} f/z & 0 & -fx/z^2 \\ 0 & f/z & -fy/z^2 \\ \cdots \end{pmatrix}$$
>
> *로 2D screen 에 projection 되고, **tile-based alpha-compositing***
>
> $$C = \sum_i \alpha_i T_i c_i, \qquad T_i = \prod_{j \lt i}(1-\alpha_j)$$
>
> *가 왜 NeRF 의 분 단위 렌더링을 100+ FPS 로 바꾸는지 유도할 수 있는 것은 다르다.*
>
> *DreamFusion 의 SDS loss 를 **호출하는 것** 과, Poole et al. (2023) 의*
>
> $$\nabla_\theta \mathcal{L}_{\mathrm{SDS}} = \mathbb{E}_{t, \epsilon, \pi}\!\left[ w(t)\bigl(\hat{\epsilon}_\phi(z_t; y, t) - \epsilon\bigr)\, \frac{\partial z}{\partial \theta} \right]$$
>
> *가 왜 U-Net 의 Jacobian 을 **명시적으로 계산하지 않고도** 3D parameter 를 update 하는지, 그리고 왜 mode-seeking · over-saturation 을 일으켜 **VSD (ProlificDreamer 2023)** 로 보정되어야 하는지 따라가는 것은 다르다.*
>
> *NeRF 의 ReLU MLP 가 **positional encoding***
>
> $$\gamma(p) = (\sin 2^0 \pi p,\ \cos 2^0 \pi p,\ \dots)$$
>
> *없이는 high-frequency 디테일을 **이론적으로** 학습 못한다는 것 — Rahaman 2019 의 spectral bias, Tancik 2020 의 NTK spectrum — 을 알고 쓰는 것은 다르다.*

<br/>

**다루는 알고리즘 (이론 계보순)**

Kajiya 1986 *Rendering Equation* · Chandrasekhar 1960 *Radiative Transfer* · Zwicker 2001 *EWA Splatting* · Qi 2017 *PointNet* · Park 2019 *DeepSDF* · Mescheder 2019 *Occupancy Networks* · Mildenhall 2020 *NeRF* · Rahaman 2019 *Spectral Bias* · Tancik 2020 *Fourier Features* · Barron 2021 *Mip-NeRF* · Müller 2022 *Instant-NGP* · Kerbl 2023 *3D Gaussian Splatting* · Park 2021 *Nerfies / HyperNeRF* · Wu 2024 / Yang 2024 *4D Gaussian Splatting* · Poole 2023 *DreamFusion / SDS* · Wang 2023 *ProlificDreamer / VSD* · Liu 2023 *Zero-1-to-3* · Shi 2024 *MVDream* · Voleti 2024 *SV3D* · Li 2024 *Instant3D* · Hong 2024 *LRM* · Wang 2024 *DUSt3R / MASt3R*

<br/>

**핵심 질문**

> 3D & Neural Rendering 은 왜 **"volume rendering equation 의 다른 discretization"** 들의 모음이고, **NeRF · Mip-NeRF · Instant-NGP · 3D Gaussian Splatting · 4D GS · DreamFusion · LRM** 이 각각 어떤 수학적 동기 (Beer–Lambert · Fourier feature · hash encoding · EWA Jacobian · SDS gradient · transformer prior) 에서 도출되었는가 — Kajiya 1986 의 rendering equation 부터 Apple Vision Pro 의 spatial computing 까지 한 줄씩 유도합니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![nerfstudio](https://img.shields.io/badge/nerfstudio-1.0-FF6F00?style=flat-square)](https://docs.nerf.studio/)
[![gsplat](https://img.shields.io/badge/gsplat-0.1.6-009688?style=flat-square)](https://github.com/nerfstudio-project/gsplat)
[![threestudio](https://img.shields.io/badge/threestudio-0.1-7E57C2?style=flat-square)](https://github.com/threestudio-project/threestudio)
[![Mitsuba](https://img.shields.io/badge/Mitsuba-3.5-4CAF50?style=flat-square)](https://www.mitsuba-renderer.org/)
[![Docs](https://img.shields.io/badge/Docs-34개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems·Definitions-280+개-success?style=flat-square)](./README.md)
[![Proofs](https://img.shields.io/badge/엄밀한_증명-130+개-9c27b0?style=flat-square)](./README.md)
[![Reproductions](https://img.shields.io/badge/Paper_reproductions-15개-critical?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-102개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

3D & Neural Rendering 자료는 대부분 **"NeRF 가 MLP 로 3D 를 학습한다"** 또는 **"3D Gaussian Splatting 이 NeRF 보다 빠르다"** 에서 멈춥니다. 하지만 NeRF 의 volumetric integral 이 왜 1986년 Kajiya 의 물리학 기반 rendering equation 의 **연속 형태** 인지, positional encoding 이 왜 단순 trick 이 아니라 **NTK spectrum 확장** 의 직접적 응용인지, 3D Gaussian 의 anisotropic covariance 가 왜 $\Sigma = RSS^\top R^\top$ 로 분해되어야 PSD 가 보장되는지, EWA Jacobian 이 왜 perspective projection 의 **local linearization** 인지, SDS loss 가 왜 score-matching 과 다르게 U-Net gradient 없이 작동하는지 — 이런 "왜" 는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "NeRF 는 MLP 로 3D scene 을 학습한다" | **Mildenhall 2020** — MLP $F_\theta:(\mathbf{x},\mathbf{d}) \to (\sigma, \mathbf{c})$. 그러나 그 뒤의 렌더링 적분 $C(\mathbf{r}) = \int T(t)\sigma\mathbf{c}\,dt$ 는 **Kajiya 1986** 의 rendering equation 에서 **Chandrasekhar 1960** 의 radiative transfer 로 환원, 순수 absorption 의 경우 **Beer–Lambert** 의 연속 미분 형태 $T(t)=\exp(-\int\sigma\,ds)$ 가 정확히 도출 $\square$. Stratified sampling 으로 $\hat{C} = \sum T_i (1-e^{-\sigma_i\delta_i}) c_i$ 의 numerical integration |
| "Positional encoding 으로 detail 이 살아난다" | **Rahaman 2019 / Tancik 2020** — ReLU MLP 의 NTK 가 low-frequency 에 집중 → high-frequency target 학습 불가 (**spectral bias** $\square$). $\gamma(p) = (\sin 2^l \pi p, \cos 2^l \pi p)_{l=0}^{L-1}$ 가 NTK 의 spectrum 을 $L$ 단계로 균일하게 확장 → high-frequency 학습 가능. **NTK 해석**: $K_{\gamma}(\mathbf{x},\mathbf{x}') = \sum_l \cos(2^l\pi(\mathbf{x}-\mathbf{x}'))$ |
| "Hierarchical sampling 으로 효율적이다" | **Mildenhall 2020 § 5.2** — Coarse network 에서 64 샘플 uniform stratified, 그 weight $w_i = T_i \alpha_i$ 를 PDF 로 inverse-CDF importance sampling 하여 fine network 에서 128 샘플. Loss $\mathcal{L} = \sum_\mathbf{r} \\\|\hat{C}_c - C\\\|^2 + \\\|\hat{C}_f - C\\\|^2$. **이론**: importance weighting 으로 variance 감소, occluded 영역 sample 절약 |
| "Mip-NeRF 는 anti-aliasing 이 좋다" | **Barron 2021** — Pixel 을 ray 가 아닌 **cone** 으로 모델, 각 sample 을 frustum 으로 통합. **Integrated Positional Encoding**: $\gamma(\mu, \Sigma) = \mathbb{E}_{x \sim \mathcal{N}(\mu,\Sigma)}[\gamma(x)]$ — frequency 가 cone 크기보다 크면 자동으로 attenuate, 다중 해상도에서 일관된 rendering. Multi-scale 학습 안정성 |
| "Instant-NGP 는 1000배 빠르다" | **Müller 2022** — Multi-resolution **hash encoding**: $L$ levels × $T$ entries hash table, feature lookup $O(L)$, MLP 는 작게 (2 layers × 64). **Collision tolerance** in feature space — gradient 가 사용된 cell 에만 흐름. Memory $O(LT)$ 가 dense voxel $O(N^3)$ 대비 우수, 5초~5분 학습 |
| "3D Gaussian Splatting 이 빠르다" | **Kerbl 2023** — Scene = $N$ 개 3D Gaussian. **Anisotropic covariance**: $\Sigma = R S S^\top R^\top$ 로 quaternion $R$ + scale $S$ 분해 → PSD 자동 보장 $\square$. View-dependent color = Spherical Harmonics $c(\mathbf{d}) = \sum c_{lm} Y_{lm}(\mathbf{d})$. **Tile-based rasterization** (16×16 tile, depth-sorted, parallel alpha-composite) 으로 100+ FPS |
| "EWA splatting 으로 projection 한다" | **Zwicker 2001** — 3D Gaussian → 2D screen 의 정확한 projection 은 비선형 (perspective). **Local linearization**: viewing transform $W$, perspective Jacobian $J$ 로 $\Sigma' = J W \Sigma W^\top J^\top$ — 2D footprint 가 다시 Gaussian. $J = \begin{psmallmatrix} f/z & 0 & -fx/z^2 \\\\ 0 & f/z & -fy/z^2 \end{psmallmatrix}$ 가 "$u = fx/z, v = fy/z$" 의 점미분 $\square$ |
| "Densification 으로 디테일이 산다" | **Kerbl 2023 § 5** — Adaptive density control: ① 큰 view-space gradient ($\\\|\nabla_{\mu}\\\|$) 인 Gaussian 을 **clone** (under-reconstruction) 또는 **split** (over-reconstruction) 으로 분할. ② 낮은 opacity ($\alpha < \epsilon$) Gaussian 을 prune. 학습 dynamics 가 quasi-greedy mesh refinement 와 유사 |
| "DreamFusion 은 2D diffusion 으로 3D 만든다" | **Poole 2023** — 3D parameter $\theta$ (NeRF), rendered view $z = g(\theta, \pi)$. **SDS gradient**: $\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t,\epsilon,\pi}[w(t)(\hat{\epsilon}_\phi(z_t;y,t) - \epsilon) \partial z/\partial \theta]$. **핵심 트릭**: U-Net Jacobian $\partial \hat{\epsilon}/\partial z_t$ 를 identity 로 근사 → diffusion model 은 **frozen score 평가기**, gradient 는 differentiable renderer 로만 흐름 $\square$ |
| "ProlificDreamer 는 SDS 보다 다양하다" | **Wang 2023** — SDS 의 mode-seeking (KL$(p\|q)$ 의 asymmetry) 분석: $\mathbb{E}[\hat{\epsilon}]$ 이 mean 에 수렴 → over-saturated, cartoon-like. **VSD**: $\nabla_\theta \mathcal{L}_{\text{VSD}} = \mathbb{E}[\hat{\epsilon}_\phi(z_t;y,t) - \hat{\epsilon}_\psi(z_t;y,t,\theta)]$ — particle-specific score $\hat{\epsilon}_\psi$ 를 LoRA 로 fine-tune, mode-covering. 다양성·선명도 향상 |
| "4D GS 는 dynamic scene 을 지원한다" | **Wu 2024 / Yang 2024** — 3D GS 에 시간 차원 추가. ① **Per-Gaussian trajectory**: polynomial $\mu(t) = \mu_0 + \sum a_k t^k$ 또는 **HexPlane** (3D + 3D plane decomposition). ② **Deformation field**: $D(\mathbf{x}, t) = \Delta\mathbf{x}$ 로 canonical Gaussian 을 변형. **Park 2021 (Nerfies)** 의 deformable NeRF 의 GS 일반화, real-time dynamic |
| "LRM 은 single image 로 3D 를 만든다" | **Hong 2024** — Transformer 기반 image → triplane NeRF. 큰 dataset (Objaverse 등) 에 priors 학습 후 feed-forward inference (수 초). **DUSt3R / MASt3R (Wang 2024)**: uncalibrated stereo 로 dense pointmap 직접 예측. **3D foundation model** 의 출현 — 공간적 prior 를 generative 가 아닌 reconstructive 로 학습 |
| 기법의 나열 | NumPy + PyTorch + nerfstudio + gsplat + threestudio + Mitsuba 로 **Volume rendering 적분 손 구현** · **Spectral bias 시연** · **NeRF 작은 scene 바닥부터** · **3D Gaussian EWA projection 직접 계산** · **Tile rasterizer 타이밍** · **SDS / VSD 2D 토이 재현** · **Marching Cubes mesh 추출** · **Instant-NGP hash encoding ablation** 까지 직접 구현해 수학적 주장을 눈으로 확인 |

---

## 📌 선행 레포 & 후속 방향

```
[Linear Algebra Deep Dive]    ─┐
[Calculus Deep Dive]           ─┤
[Diffusion Model Deep Dive]    ─┼─►  이 레포  ──► [Embodied AI · Spatial Computing]
[CNN Deep Dive]                ─┤   "왜 NeRF · 3DGS · SDS 가              Robot simulation / VR-AR
[Generative Model Deep Dive]   ─┘    volume rendering 의 다른 형태인가"   / Autonomous driving / Digital twin
         │
         ├── [Linear Algebra]            Eigendecomposition Σ = RSS^TR^T · Jacobian → Ch1, Ch4
         ├── [Calculus]                  Volume integral · chain rule for rendering → Ch2, Ch3
         ├── [Diffusion Model]           Score matching · DDPM · CFG → Ch6 SDS · VSD
         ├── [CNN]                       U-Net for diffusion · feature hierarchy → Ch3-06, Ch6
         └── [Generative Model]          VAE · 3D generation 맥락 · mode collapse → Ch6
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Linear Algebra Deep Dive** (eigendecomposition, projection, Jacobian), **Calculus Deep Dive** (Riemann integral, chain rule, gradient), **Diffusion Model Deep Dive** (score matching, DDPM, classifier-free guidance) 를 선행 지식으로 전제합니다. **CNN Deep Dive** (U-Net, feature hierarchy) 는 Ch3-06 의 Instant-NGP feature grid 와 Ch6 의 diffusion U-Net 분석에서, **Generative Model Deep Dive** (VAE, mode collapse) 는 Ch6 의 SDS mode-seeking 해석에서 권장됩니다.

> 💡 **이 레포의 핵심 기여**: Chapter 2 (Physics of Rendering) 와 Chapter 3 (NeRF) 는 현대 neural rendering 을 이해하는 **두 핵심 축**입니다. 전자는 "volume rendering equation 이 어디서 왔는가" 의 1986 Kajiya · 1960 Chandrasekhar 의 물리학적 토대 (모든 NeRF · 3DGS · SDS 가 그 응용), 후자는 "왜 ReLU MLP 가 spectral bias 를 가지는가" 의 NTK 해석 (Fourier feature · hash encoding 의 자연스러운 동기) 을 다룹니다. 이 두 축을 완전히 이해한 후 Chapter 4 (3D GS) 와 Chapter 6 (Text-to-3D) 를 읽으면 EWA Jacobian · SDS gradient 의 설계 결정 맥락이 선명해집니다.

> 🟡 **이 레포의 성격**: 여기서 다루는 일부 주제 — **NeRF vs 3D Gaussian Splatting 의 최종 승자**, **SDS vs VSD 의 mode-seeking 해법**, **LRM 같은 3D foundation model 의 한계**, **4D / dynamic scene 의 representation 표준화** — 는 **현재 진행 중인 연구 영역** 입니다. 레포는 "정답" 이 아니라 **"고전 물리학 기반 rendering 과 현대 neural 3D 사이의 지도"** 를 제공합니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-3D_Representations-7E57C2?style=for-the-badge)](./ch1-representations/01-explicit-vs-implicit.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-Rendering_Physics-7E57C2?style=for-the-badge)](./ch2-rendering-physics/01-rendering-equation.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-NeRF-7E57C2?style=for-the-badge)](./ch3-nerf/01-nerf-architecture.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-3D_Gaussian_Splatting-7E57C2?style=for-the-badge)](./ch4-gaussian-splatting/01-anisotropic-gaussian.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-Dynamic_4D-7E57C2?style=for-the-badge)](./ch5-dynamic-4d/01-deformable-nerf.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Text--to--3D-7E57C2?style=for-the-badge)](./ch6-text-to-3d/01-problem-setup.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-3D_Foundation-7E57C2?style=for-the-badge)](./ch7-foundation/01-large-reconstruction-models.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: 3D 표현 방식의 수학

> **핵심 질문:** Mesh · Point Cloud · Voxel (Explicit) 와 SDF · Occupancy · Neural Implicit (Implicit) 의 본질적 차이는 무엇이며, 각각의 memory · quality · editability trade-off 는 어디서 오는가? PointNet 의 permutation-invariant function $f(\\{x_i\\}) = g(\max_i h(x_i))$ 가 왜 set function 의 universal approximator 인가? SDF 의 eikonal equation $\\\|\nabla\phi\\\| = 1$ 이 sphere tracing 의 수학적 토대인 이유는? Marching Cubes 가 occupancy / SDF 에서 mesh 를 어떻게 추출하는가?

<details>
<summary><b>Explicit vs Implicit 부터 DeepSDF 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. 3D Representation 의 분류 — Explicit vs Implicit](./ch1-representations/01-explicit-vs-implicit.md) | **Explicit**: Mesh (vertices + faces), Point Cloud, Voxel — 위상이 자료 구조에 명시. **Implicit**: $\phi: \mathbb{R}^3 \to \mathbb{R}$ 의 level set $\\{x: \phi(x) = 0\\}$ 으로 surface 표현. **Trade-off 표**: memory $O(N)$ (mesh) vs $O(N^3)$ (voxel) vs $O(\theta)$ (neural), editability mesh 우위, smooth topology change 는 implicit 우위 |
| [02. Mesh 와 Triangle Rendering Pipeline](./ch1-representations/02-mesh-rasterization.md) | **Vertices + Faces** 표현, **Barycentric coordinates** $\mathbf{p} = \alpha\mathbf{v}_1 + \beta\mathbf{v}_2 + \gamma\mathbf{v}_3$ ($\alpha+\beta+\gamma=1$) 로 attribute 보간. **Phong shading**: ambient + diffuse + specular. **Z-buffer** 알고리즘 — depth test 의 hidden surface removal $\square$. Classical rasterization pipeline 의 수학 (model · view · projection 행렬) |
| [03. Point Cloud 와 PointNet (Qi 2017)](./ch1-representations/03-point-cloud-pointnet.md) | **Permutation-invariant function**: $f(\\{x_1,\ldots,x_n\\}) = g(\max_i h(x_i))$ — symmetric aggregation (max). **정리** (Qi 2017, Th. 1): 충분히 큰 $h, g$ 가 임의 continuous set function 을 균일 근사 $\square$. **Universal approximation of set functions**. T-Net 으로 input transform, segmentation 까지 확장 |
| [04. Signed Distance Function (SDF) 와 Sphere Tracing](./ch1-representations/04-signed-distance-function.md) | **정의**: $\phi(\mathbf{x}) = \pm \min_{\mathbf{y} \in \partial\Omega} \\\|\mathbf{x} - \mathbf{y}\\\|$. **Eikonal equation** $\\\|\nabla \phi\\\| = 1$ — true distance function 의 PDE 특성 $\square$. **Sphere tracing**: $\mathbf{x}_{k+1} = \mathbf{x}_k + \phi(\mathbf{x}_k) \mathbf{d}$ — Lipschitz 1 의 직접 활용으로 ray-surface intersection. **DeepSDF (Park 2019)**: latent code 조건부 $\phi_\theta(\mathbf{x}; z)$ |
| [05. Occupancy Networks 와 Marching Cubes (Mescheder 2019)](./ch1-representations/05-occupancy-marching-cubes.md) | **Occupancy**: $f_\theta(\mathbf{x}) \in [0, 1]$ — surface 를 $f_\theta(\mathbf{x}) = 0.5$ level set 으로. **SDF vs Occupancy 비교**: SDF 는 distance 정보 (sphere tracing 가능), occupancy 는 binary classification (학습 안정). **Marching Cubes** (Lorensen 1987): $2^8 = 256$ vertex configuration 표 → mesh 추출 알고리즘 $\square$. Implicit → Explicit 변환의 표준 |

</details>

<br/>

### 🔹 Chapter 2: Physics of Rendering — Volume Rendering Equation

> **핵심 질문:** Kajiya 1986 의 rendering equation $L_o = L_e + \int_\Omega f_r L_i \cos\theta\, d\omega$ 는 어떻게 radiometry 의 모든 rendering 을 통합하는가? Chandrasekhar 1960 의 radiative transfer equation $dL/ds = -\sigma_t L + \sigma_s \int p L\, d\omega'$ 가 안개 · 구름 · 인체 조직 같은 participating media 의 빛 전파를 어떻게 설명하는가? Beer–Lambert law 의 미분 형태에서 NeRF 의 transmittance $T(t) = \exp(-\int \sigma\, ds)$ 가 어떻게 나오는가? Stratified sampling 의 numerical integration $\hat{C} = \sum T_i (1-e^{-\sigma_i\delta_i}) c_i$ 의 오차 분석은?

<details>
<summary><b>Rendering Equation 부터 Stratified Sampling 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Rendering Equation 의 물리학적 기원 (Kajiya 1986)](./ch2-rendering-physics/01-rendering-equation.md) | **Radiometry 기본**: radiance $L$, irradiance $E$, BRDF $f_r$. **Kajiya's Equation**: $L_o(\mathbf{x},\omega_o) = L_e + \int_\Omega f_r(\omega_i, \omega_o) L_i(\omega_i)\cos\theta_i\, d\omega_i$ — emitted + reflected. **Energy conservation**: $\int f_r \cos\theta\, d\omega \leq 1$. Recursive (Neumann series) form, path tracing 의 출발점 |
| [02. Radiative Transfer Equation (Chandrasekhar 1960)](./ch2-rendering-physics/02-radiative-transfer.md) | **RTE**: $\frac{dL}{ds}(\mathbf{x},\omega) = -\sigma_t L + \sigma_s \int_{S^2} p(\omega',\omega) L(\mathbf{x},\omega')\, d\omega' + \sigma_a L_e$. **Participating media** (안개·구름·인체): absorption $\sigma_a$, scattering $\sigma_s$, extinction $\sigma_t = \sigma_a + \sigma_s$, phase function $p$ (Henyey-Greenstein 등). **유도**: 미소 cylinder 에서 photon balance — energy conservation $\square$ |
| [03. Beer–Lambert Law 와 Transmittance](./ch2-rendering-physics/03-beer-lambert.md) | **Pure absorption** ($\sigma_s = 0$): RTE 가 $dL/ds = -\sigma_t L$ 로 환원. **해**: $L(s) = L_0 \exp(-\int_0^s \sigma_t ds')$ — Beer–Lambert law $\square$. **Transmittance**: $T(a,b) = \exp(-\int_a^b \sigma_t ds)$. 광학 두께 $\tau = \int \sigma_t ds$, 다중 layer 의 곱 $T_{a \to c} = T_{a \to b} T_{b \to c}$ |
| [04. Volume Rendering Integral — NeRF Form](./ch2-rendering-physics/04-volume-rendering-integral.md) | **NeRF 식**: $C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t),\mathbf{d})\,dt$, $T(t) = \exp(-\int_{t_n}^t \sigma\,ds)$. **유도**: emission-absorption-only 가정 (no scattering) 하의 RTE 에서 직접 도출 $\square$. $\sigma$ 가 differential opacity, $\mathbf{c}$ 가 radiance contribution, $T$ 가 누적 transmittance — 모든 양이 물리학적 의미를 가짐 |
| [05. Stratified Sampling 과 Numerical Integration](./ch2-rendering-physics/05-stratified-sampling.md) | **Discretization**: $[t_n, t_f]$ 를 $N$ bin 으로 나누고 각 bin 에서 uniform sampling: $t_i \sim \mathcal{U}[t_n + (i-1)\Delta, t_n + i\Delta]$. **Discrete form**: $\hat{C} = \sum_{i=1}^N T_i(1-\exp(-\sigma_i\delta_i))\,\mathbf{c}_i$, $T_i = \exp(-\sum_{j<i}\sigma_j\delta_j)$. **오차 분석**: $O(1/N)$ for stratified vs $O(1/\sqrt{N})$ for naive Monte Carlo $\square$ |

</details>

<br/>

### 🔹 Chapter 3: NeRF 와 Neural Volume Rendering

> **핵심 질문:** Mildenhall 2020 의 NeRF MLP $F_\theta:(\mathbf{x},\mathbf{d}) \to (\sigma,\mathbf{c})$ 가 왜 view-independent density 와 view-dependent color 로 구조화되는가? Rahaman 2019 의 spectral bias 가 왜 ReLU MLP 의 본질적 한계인가, 그리고 Tancik 2020 의 Fourier feature 가 NTK 관점에서 왜 그것을 해결하는가? Hierarchical sampling 의 importance weighting 이 어떤 variance 감소를 가져오는가? Mip-NeRF 의 IPE 와 Instant-NGP 의 hash encoding 이 어떻게 NeRF 의 두 약점 (anti-aliasing, 학습 속도) 을 해결하는가?

<details>
<summary><b>NeRF Architecture 부터 Instant-NGP 까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. NeRF 의 아키텍처 (Mildenhall 2020)](./ch3-nerf/01-nerf-architecture.md) | **Network**: 8-layer × 256 width MLP, skip connection at layer 5. **Density branch**: $\sigma$ 는 view direction $\mathbf{d}$ 와 **무관** (geometry 의 view-invariance 강제). **Color branch**: $\mathbf{c}$ 는 view-dependent (specular highlight 모델링). **Sigmoid · ReLU** activation 선택 이유, scene-specific overfitting 의 의도 |
| [02. Positional Encoding 과 Spectral Bias](./ch3-nerf/02-positional-encoding-spectral-bias.md) | **정의**: $\gamma(p) = (\sin 2^l\pi p, \cos 2^l\pi p)_{l=0}^{L-1}$, $L=10$ for position, $L=4$ for direction. **Spectral bias** (Rahaman 2019): ReLU MLP 의 NTK eigenvalue 가 low frequency 에 dominant → high-freq target 학습이 exponentially slow $\square$. **NTK 해석** (Tancik 2020): $K_{\gamma}(\mathbf{x},\mathbf{x}') = \sum_l \cos(2^l\pi(\mathbf{x}-\mathbf{x}'))$ 가 spectrum 균일화 |
| [03. Hierarchical Sampling — Coarse-to-Fine](./ch3-nerf/03-hierarchical-sampling.md) | **Coarse network**: $N_c = 64$ uniform stratified, weight $w_i^c = T_i(1-e^{-\sigma_i\delta_i})$ 계산. **Fine network**: $\\{w_i^c\\}$ 를 PDF 로 normalize 후 inverse-CDF sampling 으로 $N_f = 128$ 추가 sample. **이론**: importance sampling 의 variance 가 $\text{Var}_{\text{IS}} \leq c \cdot \text{Var}_{\text{uniform}}$ ($c < 1$ if PDF is proportional to integrand) $\square$. Loss = coarse + fine 의 합 |
| [04. NeRF Loss · Training · Convergence](./ch3-nerf/04-loss-training.md) | **Loss**: $\mathcal{L} = \sum_{\mathbf{r}} \\\|\hat{C}_c(\mathbf{r}) - C(\mathbf{r})\\\|_2^2 + \\\|\hat{C}_f(\mathbf{r}) - C(\mathbf{r})\\\|_2^2$ — 단순 photometric L2. **Ray batch**: 4096 rays/iter, **Adam** lr 5e-4 → 5e-5 decay. **Overfitting on captured views**: 특정 scene 에 specialized — 100~300 image, 100k~500k iter, 1~2일 V100 GPU |
| [05. NeRF Variants — Mip-NeRF · Ref-NeRF · NeRF-W](./ch3-nerf/05-nerf-variants.md) | **Mip-NeRF (Barron 2021)**: pixel = cone, **IPE** $\gamma(\mu,\Sigma) = \mathbb{E}_{x\sim\mathcal{N}(\mu,\Sigma)}[\gamma(x)] = \exp(-\frac{1}{2}(2^l\pi)^2 \sigma^2) (\sin/\cos)$ — anti-aliasing $\square$. **Ref-NeRF (Verbin 2022)**: reflection vector parameterization, specular vs diffuse 분해. **NeRF-W (Martin-Brualla 2021)**: transient embedding 으로 photo-tourism 의 변동 처리 |
| [06. Instant-NGP — Multi-resolution Hash Encoding (Müller 2022)](./ch3-nerf/06-instant-ngp.md) | **Hash table**: $L$ levels (16개), each level $T = 2^{19}$ entries × $F = 2$ feature dim. **Lookup**: trilinear interp 후 $L$ level concat → tiny MLP (2 layers × 64). **Collision tolerance**: gradient 가 사용된 entry 에만 흐르므로 collision 이 noise 처럼 작용. **Memory** $O(LT)$ (~30MB) vs dense voxel $O(N^3)$ ($N=512$ 시 ~512MB), **1000× 학습 가속** (5초~5분) |

</details>

<br/>

### 🔹 Chapter 4: 3D Gaussian Splatting

> **핵심 질문:** Kerbl 2023 의 3D Gaussian $G(\mathbf{x}) = \exp(-\frac{1}{2}(\mathbf{x}-\mu)^\top\Sigma^{-1}(\mathbf{x}-\mu))$ 의 anisotropic covariance 가 왜 $\Sigma = RSS^\top R^\top$ 로 분해되어야 PSD 가 보장되는가? Spherical Harmonics 의 view-dependent color 표현이 왜 NeRF 의 view-direction MLP 보다 효율적인가? Zwicker 2001 의 EWA Jacobian $\Sigma' = JW\Sigma W^\top J^\top$ 가 왜 perspective projection 의 정확한 local linearization 인가? Tile-based rasterization 이 왜 NeRF 의 분 단위 렌더링을 100+ FPS 로 바꾸는가? Adaptive density control 의 clone · split · prune 이 어떻게 quasi-greedy mesh refinement 와 유사하게 작동하는가?

<details>
<summary><b>Anisotropic Gaussian 부터 Adaptive Density Control 까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. 3D Gaussian 의 Anisotropic Parameterization (Kerbl 2023)](./ch4-gaussian-splatting/01-anisotropic-gaussian.md) | **3D Gaussian**: $G(\mathbf{x}) = \exp(-\frac{1}{2}(\mathbf{x}-\mu)^\top\Sigma^{-1}(\mathbf{x}-\mu))$, parameter $\mu \in \mathbb{R}^3$, $\Sigma \in \mathbb{R}^{3\times3}$ PSD. **Factorization** $\Sigma = R(q) S S^\top R(q)^\top$, $R$ = quaternion-derived rotation, $S = \text{diag}(s_x, s_y, s_z)$. **정리**: 모든 PSD $\Sigma$ 가 이 형태로 표현 가능 + 학습 시 PSD 자동 보장 $\square$ |
| [02. Color · Opacity 와 Spherical Harmonics](./ch4-gaussian-splatting/02-spherical-harmonics-color.md) | **View-dependent color**: $c(\mathbf{d}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} Y_{lm}(\mathbf{d})$. **Spherical Harmonics** $Y_{lm}$: $L^2(S^2)$ 의 orthonormal basis (Legendre 다항식 + Fourier on $\phi$). $L=3$ 시 16 coefficient × 3 (RGB) = 48 / Gaussian. **NeRF MLP 대비**: closed-form evaluation, 학습 안정, view interpolation 자연스러움 |
| [03. EWA Splatting — Perspective Projection 의 Jacobian (Zwicker 2001)](./ch4-gaussian-splatting/03-ewa-projection.md) | **Perspective projection** $u = fx/z, v = fy/z$ — 비선형 → Gaussian 이 정확히 Gaussian 으로 안 가짐. **Local linearization**: $\mu$ 주변 1차 Taylor → Jacobian $J = \begin{psmallmatrix} f/z & 0 & -fx/z^2 \\\\ 0 & f/z & -fy/z^2 \\\\ 0 & 0 & 1 \end{psmallmatrix}$. **2D covariance**: $\Sigma' = JW\Sigma W^\top J^\top$ (top-left 2×2) $\square$. EWA 가 antialiasing 의 frequency-domain 해석 |
| [04. Tile-based Rasterization 과 Alpha-Compositing](./ch4-gaussian-splatting/04-tile-rasterization.md) | **Tile decomposition**: 16×16 pixel tile 단위로 관여 Gaussian 분류. **Per-tile sort**: depth-order sort (ascending). **Parallel alpha-composite**: $C = \sum_i \alpha_i T_i \mathbf{c}_i$, $T_i = \prod_{j<i}(1-\alpha_j)$, $\alpha_i = G_i'(p) \cdot \alpha_i^{\text{learn}}$. **CUDA kernel** 로 tile 별 thread block 병렬, 100+ FPS 가능. **미분 가능**: 각 단계가 differentiable, gradient 가 $\mu, \Sigma, c, \alpha$ 로 흐름 |
| [05. Adaptive Density Control — Clone · Split · Prune](./ch4-gaussian-splatting/05-adaptive-density-control.md) | **Densification**: view-space gradient $\\\|\nabla_{\mu_{\text{2D}}} L\\\|_2 > \tau_{\text{pos}}$ 인 Gaussian 을 ① **Clone** (under-reconstructed, $\\\|S\\\|$ 작음): 같은 위치에 복제. ② **Split** (over-reconstructed, $\\\|S\\\|$ 큼): scale $/1.6$ 로 두 개 분할. **Pruning**: $\alpha < \epsilon$ 또는 너무 큰 footprint Gaussian 제거. **Reset opacity**: 주기적 $\alpha$ 초기화로 floater 정리. Quasi-greedy mesh refinement 와 유사 |
| [06. 3DGS 학습 · 재현 · NeRF 와의 비교](./ch4-gaussian-splatting/06-3dgs-training-comparison.md) | **Loss**: $\mathcal{L} = (1-\lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$, $\lambda = 0.2$. **Init**: COLMAP point cloud 로 초기화 (또는 random). **재현**: Mip-NeRF 360 데이터셋에서 PSNR 27~28 (NeRF 와 비슷), 학습 30분 (V100), 100+ FPS rendering. **NeRF 와 비교**: training 빠름, rendering 압도적, mesh 추출 어려움, dynamic 어려움 |

</details>

<br/>

### 🔹 Chapter 5: Dynamic Scenes 와 4D

> **핵심 질문:** Dynamic NeRF (Nerfies, D-NeRF) 의 canonical space + deformation field $D(\mathbf{x},t) = \Delta\mathbf{x}$ 가 왜 시간 차원을 단순히 input 으로 추가하는 것보다 정확한가? HyperNeRF 의 ambient slicing 이 왜 입의 열림/닫힘 같은 topology 변화를 표현할 수 있는가? 4D Gaussian Splatting 의 per-Gaussian trajectory (polynomial vs HexPlane) 가 어떤 trade-off 를 가지는가? Monocular video 로부터 4D scene 을 복원할 때 camera trajectory estimation (COLMAP, MASt3R) 의 역할은?

<details>
<summary><b>Deformable NeRF 부터 Video-based 4D 까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Dynamic NeRF — Canonical Space 와 Deformation Field](./ch5-dynamic-4d/01-deformable-nerf.md) | **Architecture**: canonical NeRF $F_\theta(\mathbf{x}, \mathbf{d})$ + deformation $D_\psi: (\mathbf{x}, t) \to \Delta\mathbf{x}$. **Forward**: query at $\mathbf{x}^{\text{can}} = \mathbf{x} + D_\psi(\mathbf{x}, t)$. **Nerfies (Park 2021)**: SE(3) deformation, elastic regularizer로 rigid-as-possible. **D-NeRF (Pumarola 2021)**: time-conditional MLP. **이점**: temporal coherence (canonical 공유), monocular video 학습 가능 |
| [02. HyperNeRF — Topology Change 와 Ambient Slicing (Park 2021)](./ch5-dynamic-4d/02-hypernerf.md) | **문제**: Nerfies 는 fixed topology 가정 — 입 열림/닫힘 같은 위상 변화 처리 불가. **해법**: $\mathbb{R}^3$ canonical 을 $\mathbb{R}^{3+H}$ ambient space 로 확장, time 마다 slicing surface $\mathbf{w}(t) \in \mathbb{R}^H$. **이론**: $H = 2$ 차원 추가로 generic topology change 표현 가능 (continuous slicing). Ambient deformation field 와의 관계 |
| [03. 4D Gaussian Splatting (Wu 2024 / Yang 2024)](./ch5-dynamic-4d/03-4d-gaussian-splatting.md) | **두 접근**: ① **Per-Gaussian trajectory** (Yang 2024): $\mu(t) = \mu_0 + \sum_k a_k t^k$ polynomial, anisotropic Gaussian 의 시간 의존 parameter. ② **HexPlane decomposition** (Wu 2024): 6개 2D plane 으로 4D = 3D space × 1D time 분해. **Trade-off**: polynomial 은 long horizon 에 약함, HexPlane 은 memory 효율적이나 표현력 제한 |
| [04. Monocular Video → 4D Reconstruction · Camera Estimation](./ch5-dynamic-4d/04-video-reconstruction.md) | **Pipeline**: ① **Camera trajectory** (COLMAP SfM 또는 MASt3R feed-forward) ② **Multi-view consistency loss**: depth + flow + photometric. ③ **Regularizer**: as-rigid-as-possible, sparsity, smooth deformation. **Challenges**: dynamic objects 가 SfM 을 broken, partial observability. Recent: Dust3R-Dynamic, Shape-of-Motion (Wang 2024) |

</details>

<br/>

### 🔹 Chapter 6: Text-to-3D — Score Distillation

> **핵심 질문:** Text-to-3D 가 왜 2D diffusion prior 의 활용을 강제하는가 (3D paired data 부족)? Poole 2023 의 SDS gradient $\nabla_\theta\mathcal{L} = \mathbb{E}[w(t)(\hat{\epsilon}_\phi - \epsilon)\partial z/\partial\theta]$ 가 왜 U-Net Jacobian 을 명시적으로 계산하지 않고도 작동하는가? SDS 의 mode-seeking · over-saturation 이 KL divergence 의 어떤 asymmetry 에서 오는가? Wang 2023 의 VSD 가 어떻게 particle-specific score 로 mode-covering 을 회복하는가? Zero123 · MVDream · SV3D 같은 multi-view diffusion 이 왜 SDS 의 multi-view consistency 문제를 해결하는가?

<details>
<summary><b>Problem Setup 부터 Multi-View Diffusion 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Text-to-3D Problem Setup 과 2D Prior 활용 동기](./ch6-text-to-3d/01-problem-setup.md) | **문제**: text $y$ → 3D NeRF or mesh. **3D 데이터 부족**: Objaverse 800K 개 (vs LAION 5B 2D pair). **해결책 family**: ① **2D diffusion prior + differentiable renderer + score distillation** (DreamFusion, ProlificDreamer). ② **3D-native diffusion / regression** (Point-E, Shap-E, LRM). 각각의 trade-off (품질 vs 속도 vs diversity) |
| [02. Score Distillation Sampling (DreamFusion, Poole 2023)](./ch6-text-to-3d/02-sds-derivation.md) | **Setup**: 3D parameter $\theta$ (NeRF), random camera $\pi$, render $z = g(\theta, \pi)$. **SDS gradient**: $\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t,\epsilon,\pi}[w(t)(\hat{\epsilon}_\phi(z_t;y,t) - \epsilon)\partial z/\partial\theta]$, $z_t = \alpha_t z + \sigma_t \epsilon$. **유도**: KL$(q_\theta(z) \\\| p_\phi(z\\\|y))$ 의 gradient 에서 U-Net Jacobian $\partial\hat\epsilon/\partial z_t$ 를 identity 로 근사 → diffusion 은 frozen, gradient 는 renderer 만 통과 $\square$ |
| [03. SDS 의 Mode-Seeking 과 Over-Saturation](./ch6-text-to-3d/03-mode-seeking-saturation.md) | **현상**: cartoon-like, over-saturated, low diversity. **분석**: SDS 가 $\min_\theta \text{KL}(q_\theta \\\| p_\phi)$ 에 가까운 형태 — **forward KL (mode-covering)** 이 아닌 **reverse KL (mode-seeking)** 의 특성. **Diffusion score 의 mean-revert**: $\hat\epsilon_\phi$ 가 marginal 의 mean direction 으로 수렴 → $\theta$ 가 high-density mode 로 collapse. CFG (classifier-free guidance) scale 의 역할 |
| [04. Variational Score Distillation (ProlificDreamer, Wang 2023)](./ch6-text-to-3d/04-vsd-particle-score.md) | **VSD gradient**: $\nabla_\theta \mathcal{L}_{\text{VSD}} = \mathbb{E}[\hat\epsilon_\phi(z_t;y,t) - \hat\epsilon_\psi(z_t;y,t,\theta)]$. **Particle-specific score** $\hat\epsilon_\psi$ 를 LoRA 로 fine-tune (parameter $\theta$ 의 distribution 에 conditional). **이론**: Wasserstein gradient flow 에서 derive, mode-covering 회복 $\square$. CSD (Classifier Score Distillation, Yu 2024) 와의 관계 |
| [05. Multi-View Diffusion — Zero123 · MVDream · SV3D · Instant3D](./ch6-text-to-3d/05-multi-view-diffusion.md) | **Multi-view consistency 문제**: SDS 가 view 마다 독립 → Janus problem (multi-face). **Zero-1-to-3 (Liu 2023)**: relative camera pose 조건부 2D diffusion. **MVDream (Shi 2024)**: 4 view 동시 생성, cross-view attention. **SV3D (Voleti 2024)**: Stable Video 기반 video-as-3D. **Instant3D (Li 2024)**: 4-view + LRM-style regression — 수 초 내 3D |

</details>

<br/>

### 🔹 Chapter 7: 3D Foundation Models 과 응용

> **핵심 질문:** LRM (Hong 2024) 의 transformer-based 3D regression 이 왜 SDS 보다 빠르고 deterministic 한가 — 어떤 prior 를 large-scale data 에서 학습하는가? DUSt3R · MASt3R (Wang 2024) 의 dense pointmap 예측이 왜 uncalibrated stereo 의 SfM 을 대체하는가? Apple Vision Pro · 자율주행 · 로봇 시뮬레이션 같은 응용에서 NeRF · 3DGS 의 어떤 강점이 적합한가?

<details>
<summary><b>Large Reconstruction Models 부터 응용 까지 (3개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Large Reconstruction Models — LRM · GS-LRM · InstantMesh (Hong 2024)](./ch7-foundation/01-large-reconstruction-models.md) | **LRM (Hong 2024)**: image patches → transformer encoder → triplane NeRF decoder, Objaverse 에서 학습 후 single-image 3D 를 5초 내. **GS-LRM**: triplane 대신 3D Gaussian 직접 회귀. **InstantMesh (Xu 2024)**: multi-view → mesh 단일 stage. **이론적 동기**: 3D prior 를 generative 가 아닌 **regressive** 로 학습 → deterministic, fast, but limited diversity |
| [02. 3D Foundation Models — DUSt3R · MASt3R (Wang 2024)](./ch7-foundation/02-dust3r-mast3r.md) | **DUSt3R**: uncalibrated image pair → dense **pointmap** (3D coord per pixel) 직접 예측, ViT encoder + cross-attention. **MASt3R**: + matching head (sparse keypoint), 다중 image extension. **SfM 대체**: 전통 COLMAP 의 feature matching + bundle adjustment 가 feed-forward 로 환원, 수 초 내 결과. Robotics, AR 의 mapping 에 즉시 활용 |
| [03. 응용과 Frontier — Spatial Computing · Robotics · Autonomous Driving](./ch7-foundation/03-applications-frontier.md) | **VR / AR**: Apple Vision Pro spatial video, Quest 3 mixed reality, Gaussian Splatting content pipeline. **Robotics**: Habitat-Sim NeRF asset, GS-SLAM (Matsuki 2024). **Autonomous driving**: street-level NeRF (Block-NeRF, Tancik 2022), 4D LiDAR fusion. **Digital twin · scientific viz**: 산업용 reconstruction. **Frontier**: world model + 3D, Sora-like video → 4D scene |

</details>

---

> 🆕 **2026-04 최신 업데이트**: Ch2-04 의 Volume Rendering Integral 유도에 Chandrasekhar 1960 의 RTE 에서 emission-absorption-only 환원의 step-by-step 을 추가, Ch3-02 의 Spectral Bias 증명에 NTK eigendecomposition 의 명시적 수식을 정리, Ch4-03 의 EWA Jacobian 에 perspective projection 의 1차 Taylor 전개를 단계별로 분리, Ch6-02 의 SDS 유도에 KL divergence gradient 의 U-Net Jacobian 근사 정당화를 강화, Ch6-04 의 VSD 에 Wasserstein gradient flow 관점을 추가했습니다. **11-섹션 문서 골격이 전체 34개 문서에서 일관**됩니다.

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명** 또는 **원 논문 실험 재현** 을 제공하는 대표 결과 모음입니다. 각 챕터 문서에서 $\square$ 로 종결되는 엄밀한 증명 또는 `results/` 하의 학습 곡선·rendered novel-view 를 확인할 수 있습니다.

| 정리·결과 | 서술 | 출처 문서 |
|----------|------|----------|
| **Kajiya Rendering Equation** | $L_o = L_e + \int_\Omega f_r L_i \cos\theta\,d\omega$ — 모든 rendering 의 통합 framework | [Ch2-01](./ch2-rendering-physics/01-rendering-equation.md) |
| **Radiative Transfer Equation** | $dL/ds = -\sigma_t L + \sigma_s\int p L\,d\omega'$ — participating media 의 photon balance | [Ch2-02](./ch2-rendering-physics/02-radiative-transfer.md) |
| **Beer–Lambert → Transmittance** | Pure absorption 환원 → $T(t) = \exp(-\int\sigma\,ds)$ — NeRF 의 핵심 식 도출 | [Ch2-03](./ch2-rendering-physics/03-beer-lambert.md) |
| **Volume Rendering Integral (NeRF Form)** | $C(\mathbf{r}) = \int T(t)\sigma\mathbf{c}\,dt$ — RTE 의 emission-absorption 특수 형태 | [Ch2-04](./ch2-rendering-physics/04-volume-rendering-integral.md) |
| **Stratified Sampling 오차** | Discrete $\hat{C} = \sum T_i(1-e^{-\sigma_i\delta_i})c_i$ 의 $O(1/N)$ 수렴 | [Ch2-05](./ch2-rendering-physics/05-stratified-sampling.md) |
| **PointNet Universal Approx.** | $f(\\{x_i\\}) = g(\max_i h(x_i))$ 가 set function universal approximator | [Ch1-03](./ch1-representations/03-point-cloud-pointnet.md) |
| **Eikonal Equation $\\\|\nabla\phi\\\| = 1$** | True SDF 의 PDE 특성 → sphere tracing 의 Lipschitz 1 활용 | [Ch1-04](./ch1-representations/04-signed-distance-function.md) |
| **Spectral Bias of ReLU MLP** | NTK eigenvalue 가 low-freq dominant → high-freq target 학습 불가 | [Ch3-02](./ch3-nerf/02-positional-encoding-spectral-bias.md) |
| **Fourier Feature NTK Spectrum** | $\gamma(p)$ 가 NTK 의 spectrum 을 $L$ 단계로 균일 확장 — high-freq 학습 가능 | [Ch3-02](./ch3-nerf/02-positional-encoding-spectral-bias.md) |
| **Hierarchical Sampling Variance Reduction** | Importance sampling 으로 variance $\leq c \cdot \text{Var}_{\text{uniform}}$ | [Ch3-03](./ch3-nerf/03-hierarchical-sampling.md) |
| **Mip-NeRF Integrated PE** | $\gamma(\mu, \Sigma)$ 가 cone frustum 의 frequency attenuation — anti-aliasing | [Ch3-05](./ch3-nerf/05-nerf-variants.md) |
| **Instant-NGP Hash Encoding** | Multi-resolution hash table 의 collision tolerance — 1000× 학습 가속 | [Ch3-06](./ch3-nerf/06-instant-ngp.md) |
| **Anisotropic Σ = RSS^TR^T** | PSD 자동 보장하는 covariance factorization | [Ch4-01](./ch4-gaussian-splatting/01-anisotropic-gaussian.md) |
| **EWA Projection Jacobian** | $\Sigma' = JW\Sigma W^\top J^\top$ — perspective projection 의 local linearization | [Ch4-03](./ch4-gaussian-splatting/03-ewa-projection.md) |
| **Tile-based Alpha-Composite** | $C = \sum\alpha_i T_i c_i,\,T_i = \prod_{j<i}(1-\alpha_j)$ — 100+ FPS 의 수학 | [Ch4-04](./ch4-gaussian-splatting/04-tile-rasterization.md) |
| **Adaptive Density Control** | Clone · split · prune 의 quasi-greedy refinement 동작 | [Ch4-05](./ch4-gaussian-splatting/05-adaptive-density-control.md) |
| **HyperNeRF Ambient Slicing** | $\mathbb{R}^{3+H}$ canonical 으로 generic topology change 표현 | [Ch5-02](./ch5-dynamic-4d/02-hypernerf.md) |
| **SDS Gradient (DreamFusion)** | $\nabla_\theta\mathcal{L} = \mathbb{E}[w(\hat\epsilon_\phi - \epsilon)\partial z/\partial\theta]$ — U-Net Jacobian 근사 | [Ch6-02](./ch6-text-to-3d/02-sds-derivation.md) |
| **VSD Mode-Covering Recovery** | Particle-specific $\hat\epsilon_\psi$ 가 reverse-KL → forward-KL 보정 | [Ch6-04](./ch6-text-to-3d/04-vsd-particle-score.md) |
| **MVDream Cross-View Consistency** | 4 view 동시 생성 + cross-view attention 으로 Janus 해결 | [Ch6-05](./ch6-text-to-3d/05-multi-view-diffusion.md) |
| **LRM Triplane Regression** | Image patches → triplane NeRF 의 transformer prior — 5초 inference | [Ch7-01](./ch7-foundation/01-large-reconstruction-models.md) |
| **DUSt3R Pointmap Prediction** | Feed-forward dense pointmap 으로 SfM bundle adjustment 대체 | [Ch7-02](./ch7-foundation/02-dust3r-mast3r.md) |

> 💡 **챕터별 문서·정리/정의 수** (실측):
>
> | 챕터 | 문서 수 | 정리·정의 |
> |------|---------|------------|
> | Ch1 3D Representations | 5 | 38 |
> | Ch2 Rendering Physics | 5 | 41 |
> | Ch3 NeRF | 6 | 49 |
> | Ch4 3D Gaussian Splatting | 6 | 52 |
> | Ch5 Dynamic Scenes (4D) | 4 | 33 |
> | Ch6 Text-to-3D | 5 | 42 |
> | Ch7 Foundation Models | 3 | 28 |
> | **합계** | **34** | **283** |
>
> 추가로 **130+ 엄밀한 $\square$ 증명 + 102 연습문제 (모두 해설 포함) + 130+ NumPy/PyTorch/CUDA 실험 코드 (`### 실험 N` 형식)**.
>
> Ch5 (Dynamic 4D) 와 Ch7 (Foundation) 은 mature 한 주제만 다루기 위해 의도적으로 4·3 문서 (Ch1-Ch4 의 5~6 문서와 차이). Ch3 와 Ch4 는 NeRF · 3DGS 의 핵심 기여를 6 문서 로 풀어쓴다.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
torch==2.1.0
nerfstudio==1.0.0           # NeRF · Mip-NeRF · Instant-NGP 참조 구현 (Ch3)
gsplat==0.1.6               # 3D Gaussian Splatting CUDA kernel (Ch4)
threestudio==0.1.0          # Text-to-3D framework (Ch6 SDS · VSD)
open3d==0.18.0              # Point cloud · mesh · visualization (Ch1)
trimesh==4.0.0              # Mesh I/O · Marching Cubes (Ch1-05)
mitsuba==3.5.0              # Physically-based renderer (Ch2 검증)
diffusers==0.25.0           # Stable Diffusion · Zero123 · MVDream (Ch6)
matplotlib==3.8.0
seaborn==0.13.0
tqdm==4.66.0
jupyter==1.0.0
# 선택 사항
tensorboard==2.15.0         # 학습 곡선 로깅
wandb==0.16.0               # 실험 추적
einops==0.7.0               # tensor 재구성
tinycudann==1.7             # Instant-NGP fused MLP (Ch3-06, GPU)
```

```bash
# 환경 설치 (CUDA 11.8 기준, NeRF + 3DGS + Text-to-3D 포함)
pip install numpy==1.26.0 scipy==1.11.0 torch==2.1.0 \
            nerfstudio==1.0.0 gsplat==0.1.6 threestudio==0.1.0 \
            open3d==0.18.0 trimesh==4.0.0 mitsuba==3.5.0 \
            diffusers==0.25.0 matplotlib==3.8.0 seaborn==0.13.0 \
            tqdm==4.66.0 einops==0.7.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 ① — Volume Rendering 적분 + Spectral Bias 시연 (Ch2-05, Ch3-02)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Volume rendering (numerical integration, Beer–Lambert discrete form)
def volume_render(sigmas, colors, t_vals):
    """
    sigmas: [N_rays, N_samples] density σ
    colors: [N_rays, N_samples, 3] radiance c
    t_vals: [N_rays, N_samples] sample distances
    Discrete:  Ĉ = Σ T_i (1 - exp(-σ_i δ_i)) c_i,  T_i = exp(-Σ_{j<i} σ_j δ_j)
    """
    deltas = t_vals[..., 1:] - t_vals[..., :-1]
    deltas = torch.cat([deltas, torch.full_like(deltas[..., :1], 1e10)], dim=-1)

    alpha = 1 - torch.exp(-sigmas * deltas)                                     # [N, S]
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]),
                                 1 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = T * alpha                                                          # w_i = T_i α_i
    rgb     = (weights.unsqueeze(-1) * colors).sum(dim=-2)
    depth   = (weights * t_vals).sum(-1)
    return rgb, depth, weights


# Positional encoding γ(p) = (sin 2^l π p, cos 2^l π p)
def positional_encoding(x, L=10):
    freq_bands = 2.0 ** torch.arange(L) * np.pi
    pe = []
    for f in freq_bands:
        pe.append(torch.sin(f * x))
        pe.append(torch.cos(f * x))
    return torch.cat(pe, dim=-1)


# Spectral bias 시연 — high frequency 1D signal
def fit_signal_with_without_pe():
    x = torch.linspace(-1, 1, 1000).unsqueeze(-1)
    y_true = torch.sin(20 * np.pi * x).squeeze()        # high freq target

    model_no_pe = nn.Sequential(
        nn.Linear(1, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 1),
    )

    class MLPwithPE(nn.Module):
        def __init__(self, L=6):
            super().__init__()
            self.L = L
            self.net = nn.Sequential(
                nn.Linear(2 * L, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 1),
            )
        def forward(self, x):
            return self.net(positional_encoding(x, self.L))

    model_pe = MLPwithPE(L=6)

    for name, model in [("No PE", model_no_pe), ("With PE", model_pe)]:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for step in range(5000):
            pred = model(x).squeeze()
            loss = ((pred - y_true) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"{name}: final loss = {loss.item():.6f}")
    # With PE << No PE  →  spectral bias 극복 (Tancik 2020)


# 대표 실험 ② — 3D Gaussian Splatting EWA projection (Ch4-03)
def project_gaussian_3d_to_2d(mu_3d, Sigma_3d, W, focal):
    """
    mu_3d   : [3]    Gaussian center (world)
    Sigma_3d: [3, 3] anisotropic covariance
    W       : [3, 3] view rotation
    focal   : scalar camera focal length
    Returns mu_2d ([2]) and Sigma_2d ([2, 2]) screen footprint.
    """
    mu_view     = W @ mu_3d
    tx, ty, tz  = mu_view

    # Perspective Jacobian: u = f x/z, v = f y/z
    J = torch.zeros(3, 3)
    J[0, 0] = focal / tz;       J[0, 2] = -focal * tx / (tz ** 2)
    J[1, 1] = focal / tz;       J[1, 2] = -focal * ty / (tz ** 2)
    J[2, 2] = 1.0

    Sigma_cam = W @ Sigma_3d @ W.T
    Sigma_img = J @ Sigma_cam @ J.T
    Sigma_2d  = Sigma_img[:2, :2]                                # drop z

    mu_2d = torch.tensor([focal * tx / tz, focal * ty / tz])
    return mu_2d, Sigma_2d
# → ε = anisotropic ratio 별 2D footprint 시각화 (3D ellipsoid → 2D ellipse)


# 대표 실험 ③ — SDS Loss 의 토이 재현 (Ch6-02)
# 1D 또는 2D 에서 frozen "diffusion model" 로 단일 Gaussian mode-seek 관찰
# → over-saturation / mode collapse 직접 확인


# 대표 실험 ④ — Tile-based 3DGS rasterization 타이밍 (Ch4-04)
# 같은 scene 에서 NeRF (분 단위) vs 3DGS (실시간) FPS 비교
# gsplat CUDA kernel 활용
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격** 으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 ... 인가** | 해당 이론·알고리즘이 3D rendering 의 어떤 핵심 문제를 푸는지 |
| 3 | 📐 **수학적 선행 조건** | LA · Calc · Diffusion · CNN 레포의 어떤 정리를 전제하는지 |
| 4 | 📖 **직관적 이해** | Volume rendering · ray marching · Gaussian ellipsoid · SDS gradient 의 기하학적 직관 |
| 5 | ✏️ **엄밀한 정의** | Volume rendering integral · EWA Jacobian · SDS loss · soft Bellman 등 |
| 6 | 🔬 **정리와 증명** | Beer–Lambert 유도 · spectral bias 증명 · EWA derivation · SDS gradient 도출 |
| 7 | 💻 **PyTorch / CUDA 구현 검증** | 4 가지 실험 (`### 실험 1`~`### 실험 4`) — toy scene · NeRF / 3DGS 재현 · ablation · 시각화 |
| 8 | 🔗 **실전 활용** | 언제 NeRF · 언제 3DGS · 언제 SDS · 언제 LRM — 응용별 선택 가이드 |
| 9 | ⚖️ **가정과 한계** | 각 방식의 실패 모드 (조명 효과 제한 · dynamic · sparse view · ambiguity) |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 ($\boxed{}$ 핵심 수식 + 표) |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 기초 / 심화 / 논문 비평 의 3 문제, `<details>` 펼침 해설 |

> 📚 **연습문제 총 102개** (34 문서 × 3 문제): **기초 / 심화 / 논문 비평** 의 3-tier 구성, 모든 문제에 `<details>` 펼침 해설 포함. Beer–Lambert 손 유도부터 NTK eigenvalue 분석, EWA Jacobian 의 Taylor 전개, SDS gradient 의 KL 해석, VSD 의 Wasserstein flow, LRM 의 transformer prior 까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 다음 챕터 첫 문서로 자동 연결됩니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 약 500~700줄 (정의·증명·코드·연습문제 포함) 기준 **약 60분~2시간**. 전체 34문서는 약 **40~50시간** 상당 (증명 재구성·NeRF/3DGS 재현 포함 시 70시간+).

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "NeRF 는 써봤지만 volume rendering 의 물리학 토대를 이해하고 싶다" — 입문 투어 (1주, 약 12~14시간)</b></summary>

<br/>

```
Day 1  Ch2-01  Rendering Equation (Kajiya)
       Ch2-03  Beer–Lambert Law 와 Transmittance
Day 2  Ch2-04  Volume Rendering Integral (NeRF Form)
       Ch2-05  Stratified Sampling
Day 3  Ch3-01  NeRF 아키텍처
       Ch3-02  Positional Encoding 과 Spectral Bias
Day 4  Ch3-03  Hierarchical Sampling
       Ch3-04  Loss · Training
Day 5  Ch4-01  Anisotropic Gaussian
       Ch4-03  EWA Splatting
Day 6  Ch4-04  Tile-based Rasterization
       Ch4-06  3DGS 학습 · 비교
Day 7  Ch6-02  SDS Derivation (DreamFusion)
       Ch6-05  Multi-View Diffusion
```

</details>

<details>
<summary><b>🟡 "Volume Rendering 의 물리학 + NeRF · 3DGS 의 수학을 정복한다" — 이론 집중 (2주, 약 24~28시간)</b></summary>

<br/>

```
1주차 — Representations · Rendering Physics · NeRF
  Day 1    Ch1-01~02   Explicit/Implicit + Mesh Rasterization
  Day 2    Ch1-03~05   PointNet + SDF + Occupancy/Marching Cubes
  Day 3    Ch2-01~02   Rendering Equation + RTE
  Day 4    Ch2-03~05   Beer–Lambert + Volume Integral + Stratified
  Day 5    Ch3-01~02   NeRF 아키텍처 + PE/Spectral Bias
  Day 6    Ch3-03~04   Hierarchical Sampling + Loss
  Day 7    Ch3-05~06   Mip-NeRF + Instant-NGP

2주차 — 3DGS · Dynamic · Text-to-3D · Foundation
  Day 1    Ch4-01~02   Anisotropic Gaussian + SH Color
  Day 2    Ch4-03~04   EWA Projection + Tile Rasterization
  Day 3    Ch4-05~06   Density Control + 학습/비교
  Day 4    Ch5-01~02   Deformable NeRF + HyperNeRF
  Day 5    Ch5-03~04   4D GS + Video Reconstruction
  Day 6    Ch6-01~03   Problem Setup + SDS + Mode-Seeking
  Day 7    Ch6-04~05   VSD + Multi-View Diffusion
```

</details>

<details>
<summary><b>🔴 "3D & Neural Rendering 의 수학을 완전 정복한다" — 전체 정복 (8주, 약 40~50시간 + NeRF/3DGS 재현 18~24시간)</b></summary>

<br/>

```
1주차   Chapter 1 전체 — 3D Representations
         → Mesh / Point Cloud / Voxel / SDF / Occupancy 비교
         → PointNet 의 universal approximation 증명
         → DeepSDF · Occupancy Networks 직접 학습 + Marching Cubes mesh

2주차   Chapter 2 전체 — Physics of Rendering
         → Kajiya rendering equation 으로부터 RTE 유도
         → Beer–Lambert 의 미분 형태에서 NeRF transmittance 도출
         → Stratified vs Importance sampling 의 variance 비교

3주차   Chapter 3 전체 — NeRF
         → Spectral bias 의 NTK eigenvalue 손 분석
         → Positional Encoding L=4, 8, 10 ablation + NTK spectrum 측정
         → Lego scene 에서 NeRF 바닥부터 학습 → novel view 검증
         → Mip-NeRF 의 IPE · Instant-NGP 의 hash encoding 직접 구현

4주차   Chapter 4 전체 — 3D Gaussian Splatting
         → Σ = RSS^TR^T 의 PSD 보장 손 증명
         → EWA Jacobian 의 1차 Taylor 전개 손 유도
         → gsplat 으로 3DGS 작은 scene (Lego) 학습 + tile rasterization 타이밍
         → Adaptive density control 의 clone/split/prune 효과 ablation

5주차   Chapter 5 전체 — Dynamic 4D
         → Nerfies 의 deformation field 학습 (toy 4D scene)
         → HyperNeRF 의 ambient slicing 의 topology change 시각화
         → 4D GS (HexPlane vs polynomial) 의 trajectory 비교
         → Monocular video → 4D 재구성 (DUSt3R 활용)

6주차   Chapter 6 전체 — Text-to-3D · SDS
         → SDS gradient 의 U-Net Jacobian 근사 정당화 손 유도
         → 2D toy 에서 mode-seeking · over-saturation 재현
         → DreamFusion → ProlificDreamer (VSD) particle score 학습
         → MVDream / Zero123 의 multi-view consistency 측정

7주차   Chapter 7 (1~3) — Foundation Models · 응용
         → LRM 의 transformer prior 분석 + InstantMesh 재현
         → DUSt3R / MASt3R 의 pointmap inference 시연
         → Apple Vision Pro · Gaussian Splatting SLAM 의 응용 사례

8주차   종합 토론 + 추가 주제
         → "NeRF vs 3DGS" / "SDS vs 3D-native diffusion" 비교 논쟁
         → 4D dynamic scene 의 representation 표준화 토론
         → Sora-like video → 4D scene 의 frontier 정리
         → 자신만의 small-scale 3D 프로젝트 1개 완성
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | Eigendecomposition · Projection · Jacobian | **Ch1-04** (SDF gradient), **Ch4-01** (Σ = RSS^TR^T), **Ch4-03** (EWA Jacobian) |
| [calculus-deep-dive](https://github.com/iq-ai-lab/calculus-deep-dive) | Riemann integral · Chain rule · Gradient | **Ch2 전체** (volume integral), **Ch3-04** (rendering chain rule) |
| [diffusion-model-deep-dive](https://github.com/iq-ai-lab/diffusion-model-deep-dive) | Score matching · DDPM · Classifier-free guidance | **Ch6 전체** (SDS · VSD · multi-view diffusion) |
| [cnn-deep-dive](https://github.com/iq-ai-lab/cnn-deep-dive) | U-Net · Feature hierarchy · Receptive field | **Ch3-06** (feature grid), **Ch6** (diffusion U-Net) |
| [generative-model-deep-dive](https://github.com/iq-ai-lab/generative-model-deep-dive) | VAE · Mode collapse · Energy-based | **Ch6-03** (mode-seeking), **Ch7-01** (LRM 의 generative vs regressive) |
| [transformer-deep-dive](https://github.com/iq-ai-lab/transformer-deep-dive) | Self-attention · Cross-attention · Vision Transformer | **Ch7-01** (LRM transformer), **Ch7-02** (DUSt3R ViT) |
| [advanced-rl-deep-dive](https://github.com/iq-ai-lab/advanced-rl-deep-dive) | TRPO · PPO · SAC · TD3 | **Ch7-03** (robotics simulation 의 RL agent) |

> 💡 이 레포는 **"NeRF · 3DGS · DreamFusion 이 모두 volume rendering equation 의 다른 discretization 이고, positional encoding · EWA Jacobian · SDS gradient 가 왜 각각의 이론적 동기를 갖는가"** 에 집중합니다. Linear Algebra 에서 eigendecomposition 과 Jacobian 을, Calculus 에서 Riemann integral 과 chain rule 을, Diffusion Model 에서 score matching 과 classifier-free guidance 를, CNN 에서 U-Net 을 익힌 후 오면 Chapter 2 (rendering physics) 와 Chapter 6 (SDS 유도) 의 증명이 훨씬 자연스럽습니다. **Transformer Deep Dive** 와 함께 보면 Ch7 의 LRM · DUSt3R 같은 3D foundation model 이 왜 transformer prior 에 의존하는지 맥락이 선명해집니다.

---

## 📖 Reference

### 🏛️ Physics of Rendering
- **Radiative Transfer** (Chandrasekhar, 1960) — **고전 교과서, RTE 효시**
- **The Rendering Equation** (Kajiya, 1986) — **물리학 기반 rendering 의 효시**
- **EWA Splatting** (Zwicker et al., 2001) — **3D Gaussian splatting 의 projection 이론**
- **Physically Based Rendering: From Theory to Implementation** (Pharr, Jakob, Humphreys, 4th ed.)

### 🎨 3D Representations
- **PointNet: Deep Learning on Point Sets** (Qi et al., 2017)
- **PointNet++** (Qi et al., 2017) — Hierarchical feature learning
- **DeepSDF: Learning Continuous Signed Distance Functions** (Park et al., 2019)
- **Occupancy Networks: Learning 3D Reconstruction in Function Space** (Mescheder et al., 2019)
- **Marching Cubes** (Lorensen & Cline, 1987)
- **Convolutional Occupancy Networks** (Peng et al., 2020)

### 🌌 NeRF · Neural Volume Rendering
- **NeRF: Representing Scenes as Neural Radiance Fields** (Mildenhall et al., 2020) — **NeRF 효시**
- **On the Spectral Bias of Neural Networks** (Rahaman et al., 2019)
- **Fourier Features Let Networks Learn High Frequency Functions** (Tancik et al., 2020) — NTK 해석
- **Mip-NeRF: A Multiscale Representation for Anti-Aliasing NeRF** (Barron et al., 2021)
- **Mip-NeRF 360** (Barron et al., 2022) — Unbounded scenes
- **Ref-NeRF** (Verbin et al., 2022) — Reflection decomposition
- **NeRF in the Wild** (Martin-Brualla et al., 2021) — Photo tourism
- **Instant Neural Graphics Primitives** (Müller et al., 2022) — **Instant-NGP**
- **Plenoxels** (Fridovich-Keil et al., 2022) — Neural-free voxel
- **TensoRF** (Chen et al., 2022) — Tensor decomposition

### ✨ 3D Gaussian Splatting
- **3D Gaussian Splatting for Real-Time Radiance Field Rendering** (Kerbl et al., 2023) — **3DGS 효시**
- **Mip-Splatting** (Yu et al., 2024) — Anti-aliasing for 3DGS
- **2D Gaussian Splatting** (Huang et al., 2024) — Surface reconstruction
- **Scaffold-GS** (Lu et al., 2024) — Anchor-based densification
- **GS-SLAM** (Matsuki et al., 2024) — Real-time SLAM with 3DGS

### ⏳ Dynamic Scenes · 4D
- **Nerfies: Deformable Neural Radiance Fields** (Park et al., 2021)
- **HyperNeRF** (Park et al., 2021) — Topology change
- **D-NeRF** (Pumarola et al., 2021)
- **NeRFlow** (Du et al., 2021)
- **4D Gaussian Splatting** (Wu et al., 2024)
- **Real-time Photorealistic Dynamic Scene Representation** (Yang et al., 2024) — 4DGS
- **HexPlane** (Cao & Johnson, 2023)
- **Dynamic 3D Gaussians** (Luiten et al., 2024)
- **Shape of Motion** (Wang et al., 2024) — Monocular 4D

### 🎯 Text-to-3D · Score Distillation
- **DreamFusion: Text-to-3D using 2D Diffusion** (Poole et al., 2023) — **SDS**
- **Magic3D** (Lin et al., 2023) — Coarse-to-fine + mesh refinement
- **Score Jacobian Chaining** (Wang et al., 2023)
- **ProlificDreamer** (Wang et al., 2023) — **VSD**
- **Classifier Score Distillation** (Yu et al., 2024) — CSD
- **Zero-1-to-3** (Liu et al., 2023)
- **MVDream** (Shi et al., 2024)
- **SV3D** (Voleti et al., 2024)
- **Instant3D** (Li et al., 2024)
- **Wonder3D** (Long et al., 2024) — Cross-domain diffusion

### 🚀 3D Foundation Models · Feed-Forward
- **LRM: Large Reconstruction Model** (Hong et al., 2024)
- **GS-LRM** (Zhang et al., 2024) — LRM with Gaussian outputs
- **InstantMesh** (Xu et al., 2024)
- **CRM: Convolutional Reconstruction Model** (Wang et al., 2024)
- **DUSt3R: Geometric 3D Vision Made Easy** (Wang et al., 2024)
- **MASt3R: Grounding Image Matching in 3D** (Wang et al., 2024)
- **Splatt3R** (Smart et al., 2024) — Feed-forward 3DGS

### 🛠️ Implementation · Libraries
- **nerfstudio** (Tancik et al., 2023) — NeRF framework
- **gsplat** (Ye et al., 2024) — Differentiable 3DGS CUDA kernel
- **threestudio** (Guo et al., 2023) — Text-to-3D framework
- **Mitsuba 3** (Jakob et al., 2022) — Physically-based renderer
- **Open3D** (Zhou et al., 2018)
- **PyTorch3D** (Ravi et al., 2020)
- **Kaolin** (Jatavallabhula et al., 2019)

---

<div align="center">

**⭐️ 도움이 되셨다면 Star 를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"NeRF 를 호출하는 것과 — Mildenhall 2020 으로 volume rendering integral $C(\mathbf{r}) = \int T(t)\sigma\mathbf{c}\,dt$ 가 Kajiya 1986 의 rendering equation, Chandrasekhar 1960 의 RTE 의 emission-absorption 환원임을 한 줄씩 증명 · Tancik 2020 으로 positional encoding 이 NTK spectrum 의 균일 확장으로 spectral bias 를 해결함을 유도 · Kerbl 2023 으로 3D Gaussian 의 anisotropic covariance · EWA Jacobian · tile rasterization 이 어떻게 100+ FPS 를 가능하게 하는지 분석 · Poole 2023 으로 SDS gradient 가 어떻게 U-Net Jacobian 없이 3D parameter 를 update 하는지 도출 · Wang 2023 으로 VSD 의 particle-specific score 가 SDS 의 mode-seeking 을 어떻게 보정하는지 유도 · Hong 2024 / Wang 2024 로 LRM · DUSt3R 의 transformer prior 가 generative 가 아닌 regressive 3D foundation 인 이유를 정리 — 이 모든 '왜' 를 직접 유도할 수 있는 것은 다르다"*

</div>
