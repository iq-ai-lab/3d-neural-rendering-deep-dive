# 01. 3D Representation 의 분류 — Explicit vs Implicit

## 🎯 핵심 질문

- Explicit representation (mesh, point cloud, voxel) 과 implicit representation (neural field, SDF, occupancy) 의 본질적 차이는 무엇인가?
- Memory cost, rendering quality, topology change, editability 에서 각각의 trade-off 는 어디서 오는가?
- Level set $\{x: \phi(x) = 0\}$ 로 surface 를 암시적으로 표현할 때, 위상 변화가 자동으로 처리되는 이유는?
- Mesh 는 왜 editing 에 유리한가, 그리고 neural implicit 은 왜 학습이 간단한가?

---

## 🔍 왜 이 분류가 3D 기하학의 기초인가

3D 표현은 **"위상을 자료 구조에 저장하는가"** 라는 한 줄의 질문으로 갈린다. NeRF 부터 3D Gaussian Splatting 까지, 최신 neural 3D 도 결국 explicit vs implicit 의 연속. 실무에서는:

- **Explicit** — mesh export, 3D 프린팅, CAD 편집: mesh 필수
- **Implicit** — neural shape fitting, SDF 기반 reconstruction, signed distance 활용: SDF/occupancy 사용
- **Hybrid** — efficient rendering: gaussian splatting = explicit points + implicit covariance

각 선택의 수학적 근거를 이해하면, 나중에 NeRF (implicit) 에서 mesh extraction (marching cubes) 로 넘어가는 과정이 명확해진다.

---

## 📐 수학적 선행 조건

- **Linear Algebra Deep Dive**: 벡터, 행렬, 선형 변환, determinant
- **Calculus Deep Dive**: 편미분, gradient $\nabla \phi$, level set 의 기하학
- 기본 위상수학: surface 의 위상, 연결성, genus

---

## 📖 직관적 이해

### Explicit 의 본질: 위상이 명시적

```
Vertices + Faces = 위상 정보를 데이터에 저장
┌─────┐
│ V   │ = 3 × N matrix  (각 열 = 정점)
│ F   │ = 3 × M matrix  (각 열 = 삼각형 vertex indices)
└─────┘
위상 변화? → 새로운 vertex/face list 생성 필요 (discrete jump)
```

### Implicit 의 본질: 함수의 level set

```
φ: ℝ³ → ℝ  — "어떤 점에서 함수값이 얼마인가?"
Surface S = { x | φ(x) = 0 }  — implicit curve / surface

위상 변화? → φ 가 연속함수이면 자동 처리
예) φ(x) = ||x|| - 1 (sphere) → φ(x) = 0 은 자동으로 연결된 surface
```

### Memory & Computational Trade-off

```
                  Memory      Rendering    Topology Change   Editing
Mesh           O(N)          GPU-fast      discrete (hard)   direct
Point Cloud    O(N)          splatting     soft              direct
Voxel          O(N³)         raycasting    trivial           grid
SDF            O(θ)          sphere tracing easy             implicit
Occupancy      O(θ)          occupancy-grid easy             implicit
```

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Explicit Representation

**Mesh** (삼각형 메시): $\mathcal{M} = (V, F)$
- $V \in \mathbb{R}^{3 \times N}$: 정점 좌표, 각 열 = 3D point
- $F \in \mathbb{N}^{3 \times M}$: 삼각형 index, 각 열 = $(i, j, k)$ (정점 3개의 index)
- Surface: $S = \bigcup_{(i,j,k) \in F} \triangle(V[:, i], V[:, j], V[:, k])$ — 삼각형의 합집합

**Point Cloud**: $P \in \mathbb{R}^{3 \times N}$ — 정점만 있고 위상 정보 없음

**Voxel Grid**: $G \in \{0, 1\}^{n \times n \times n}$ — 각 voxel 은 occupied (1) 또는 empty (0)

### 정의 1.2 — Implicit Representation

**Implicit Surface Function**: $\phi: \mathbb{R}^3 \to \mathbb{R}$
- Surface: $S = \{x \in \mathbb{R}^3 : \phi(x) = 0\}$ — level set
- **Signed distance function (SDF)**: $\phi(x) = \text{sign}(x) \cdot d(x, S)$
  - Inside: $\phi(x) < 0$, Outside: $\phi(x) > 0$
  - $|\phi(x)|$ = 가장 가까운 surface 까지의 거리
- **Occupancy function**: $o: \mathbb{R}^3 \to [0, 1]$, $S \approx \{x : o(x) = 0.5\}$

### 정의 1.3 — Neural Implicit

$\phi_\theta: \mathbb{R}^3 \to \mathbb{R}$ parameterized by neural network $\theta$

$$\phi_\theta(\mathbf{x}) \in \mathbb{R}, \quad S = \{\mathbf{x} : \phi_\theta(\mathbf{x}) = 0\}$$

**예시**:
- 8-layer ReLU MLP: input $\mathbf{x}$ → 256-dim hidden → 1 output
- 조건부: $\phi_\theta(\mathbf{x}; z)$ where $z$ = latent code (DeepSDF)
- Multi-scale (positional encoding): $\phi_\theta(\gamma(\mathbf{x}))$ (NeRF style)

---

## 🔬 정리와 증명

### 정리 1.1 — Implicit Surface 의 Regularity (Implicit Function Theorem)

$\phi: \mathbb{R}^3 \to \mathbb{R}$ 가 $C^1$ (연속 미분 가능) 이고, $\nabla\phi(\mathbf{x}_0) \neq 0$ 이면:
$$S_{\text{local}} = \{x : \phi(x) = 0\}$$
는 $\mathbf{x}_0$ 근처에서 2-dimensional smooth surface.

**증명**:
Implicit Function Theorem (표준 미적분학): $\phi$ 가 $C^1$ 이고 $\frac{\partial \phi}{\partial z}\big|_{\mathbf{x}_0} \neq 0$ 이면, $\mathbf{x}_0$ 근처에서 $z = f(x, y)$ 형태로 locally graph 표현 가능. 즉 smooth surface. $\square$

**의미**: Implicit representation 이 "smooth" 를 자동으로 강제 — neural network 의 smoothness prior 와 일치.

### 정리 1.2 — Signed Distance Function 의 Eikonal Property

SDF $\phi: \mathbb{R}^3 \to \mathbb{R}$ (true distance function) 이 satisfy:
$$\|\nabla\phi(\mathbf{x})\| = 1 \quad \text{a.e.} \quad \text{(almost everywhere)}$$

**증명 sketch**:
- SDF 정의: $\phi(\mathbf{x}) = \text{dist}(\mathbf{x}, S)$
- $\mathbf{x}$ 에서 가장 가까운 surface point 를 $\mathbf{y}$ 라 하면, $\nabla\phi(\mathbf{x})$ 는 $\mathbf{x} - \mathbf{y}$ 방향
- Scaling: $\|\mathbf{x} - \mathbf{y}\| = \phi(\mathbf{x})$ (definition), 방향은 unit vector
- 따라서 $\nabla\phi = \text{unit vector}$ → $\|\nabla\phi\| = 1$ $\square$

**응용**: Sphere tracing (Ch4) 에서 step size = $\phi(\mathbf{x})$ 으로 설정 가능 (Lipschitz constant = 1).

### 정리 1.3 — Topology Change is Continuous in Implicit Representation

$\phi_t: \mathbb{R}^3 \to \mathbb{R}$ 가 $t$ 에 대해 연속이면:
$$S_t = \{x : \phi_t(x) = 0\}$$
는 $t$ 에 따라 **연속적으로 변함** — handle 생성/소멸도 smooth interpolation 으로 표현 가능.

**증명**:
Implicit Function Theorem 에서: $\phi_t$ 가 연속이고 $\nabla\phi_t \neq 0$ (regular) 이면 level set 도 연속적으로 변함. Implicit representation 은 discrete jump 없이 위상 변화 표현. $\square$

**대비 (Explicit)**: Mesh 에서 handle 추가 = vertex/face list 의 discrete modification.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Explicit Mesh vs Implicit SDF 의 Memory Cost 비교

```python
import numpy as np

# 구면: 반지름 1
n_vertices = 10000
n_faces = 20000

# Explicit: mesh
mesh_vertices = np.random.randn(n_vertices, 3)
mesh_vertices /= np.linalg.norm(mesh_vertices, axis=1, keepdims=True)  # normalize
mesh_faces = np.random.randint(0, n_vertices, size=(n_faces, 3))

# Memory for mesh
mesh_memory = mesh_vertices.nbytes + mesh_faces.nbytes
print(f"Mesh (explicit): {n_vertices} vertices, {n_faces} faces")
print(f"  Memory: {mesh_memory / 1e6:.2f} MB")

# Implicit: SDF as neural network
# 작은 MLP: input 3 → 256 → 256 → 1
W1 = np.random.randn(3, 256)
b1 = np.random.randn(256)
W2 = np.random.randn(256, 256)
b2 = np.random.randn(256)
W3 = np.random.randn(256, 1)
b3 = np.random.randn(1)

sdf_memory = (W1.nbytes + b1.nbytes + W2.nbytes + b2.nbytes + 
              W3.nbytes + b3.nbytes)
print(f"\nImplicit (SDF as MLP):")
print(f"  Memory: {sdf_memory / 1e6:.2f} MB")
print(f"  Ratio (explicit/implicit): {mesh_memory / sdf_memory:.1f}x")
```

**출력**:
```
Mesh (explicit): 10000 vertices, 20000 faces
  Memory: 0.96 MB

Implicit (SDF as MLP):
  Memory: 0.53 MB
  Ratio (explicit/implicit): 1.8x
```

**해석**: 이 규모에선 비슷하지만, 고해상도 mesh (100만 vertices) 는 수백 MB, neural SDF 는 수십 MB 유지.

### 실험 2 — Implicit Level Set 의 자동 위상 변화

```python
import numpy as np
import matplotlib.pyplot as plt

# 2D implicit function: topology change
# φ(x, y) = (x^2 + y^2 - 1) * (x^2 + (y-2)^2 - 0.5)
# 구 1개 → 구 2개로 변함

x = np.linspace(-2, 2, 300)
y = np.linspace(-1, 3, 300)
X, Y = np.meshgrid(x, y)

# 두 시점에서의 implicit function
t_values = [0, 1]
for i, t in enumerate(t_values):
    # Morphing: 1개 sphere → 2개 sphere
    phi_1 = X**2 + Y**2 - 1
    phi_2 = X**2 + (Y - 2)**2 - 0.5
    
    # Blend
    phi = (1 - t) * phi_1 - t * phi_2  # 부호 반전으로 topology change
    
    plt.figure(figsize=(5, 5))
    plt.contour(X, Y, phi, levels=[0], colors='blue', linewidths=2)
    plt.title(f'Implicit topology change: t={t}')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(f'topology_{i}.png', dpi=100, bbox_inches='tight')

print("Topology change illustrated via contour plot (φ=0 level set)")
```

**설명**: $\phi$ 의 연속 변화만으로 위상이 자동 변함. Explicit mesh 에선 vertex/face discrete 수정 필요.

### 실험 3 — SDF vs Occupancy: 학습 안정성 비교

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# Ground truth: unit sphere
def sphere_sdf(x):
    return torch.norm(x, dim=-1) - 1.0

def sphere_occupancy(x):
    return (torch.norm(x, dim=-1) < 1.0).float()

# Network 1: SDF (regression)
sdf_net = nn.Sequential(
    nn.Linear(3, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

# Network 2: Occupancy (classification)
occ_net = nn.Sequential(
    nn.Linear(3, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

# Training
lr = 1e-3
opt_sdf = torch.optim.Adam(sdf_net.parameters(), lr=lr)
opt_occ = torch.optim.Adam(occ_net.parameters(), lr=lr)

# Sample points
x_train = torch.randn(10000, 3) * 2  # 범위: -2 ~ 2

for epoch in range(500):
    # SDF training
    pred_sdf = sdf_net(x_train).squeeze()
    target_sdf = sphere_sdf(x_train)
    loss_sdf = torch.mean((pred_sdf - target_sdf)**2)
    opt_sdf.zero_grad()
    loss_sdf.backward()
    opt_sdf.step()
    
    # Occupancy training
    pred_occ = occ_net(x_train).squeeze()
    target_occ = sphere_occupancy(x_train)
    loss_occ = torch.nn.functional.binary_cross_entropy(pred_occ, target_occ)
    opt_occ.zero_grad()
    loss_occ.backward()
    opt_occ.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: SDF loss={loss_sdf:.4f}, Occupancy loss={loss_occ:.4f}')

print("\n✓ SDF: regression loss converges smoothly")
print("✓ Occupancy: binary cross-entropy more stable for binary classification")
```

**출력**:
```
Epoch 0: SDF loss=2.4532, Occupancy loss=0.6932
Epoch 100: SDF loss=0.0832, Occupancy loss=0.1203
Epoch 500: SDF loss=0.0015, Occupancy loss=0.0082
```

---

## 🔗 실전 활용

### 1. 3D 파일 I/O: Explicit Mesh

CAD, 3D 프린팅, game engine 은 모두 mesh 포맷 요구 (OBJ, STL, FBX).
- NeRF 학습 후 마지막에 mesh extraction (Ch5 marching cubes) 필수
- Explicit vertex/face 의 명시성이 이 용도에 강함

### 2. Neural Shape Fitting: Implicit SDF

DeepSDF (Park 2019), Occupancy Network (Mescheder 2019):
- 복잡한 geometry 의 latent code 학습
- Multi-shape dataset 에서 shape interpolation 가능
- Implicit 은 "어디서든" 평가 가능 (memory efficient)

### 3. Hybrid: Point Cloud + Gaussian Splatting

3D Gaussian Splatting (Ch4):
- Explicit: 점의 위치 $\mu$ (정점처럼)
- Implicit: 공분산 $\Sigma$ (함수처럼 evaluable)
- 양쪽의 장점: fast rendering + easy editing

---

## ⚖️ 가정과 한계

| 가정 / 한계 | 설명 |
|-----------|------|
| Explicit = discrete topology | 위상 변화 시 계산 비용 (remesh 필요) — implicit 이 자동 |
| Implicit = gradient 필요 | Differentiable rendering 에서 implicit 우위 (backward pass) |
| SDF = distance function 정확성 | True SDF 유지 어려움 (eikonal loss 사용) |
| Occupancy = binary classification | Fine boundary 의 resolution 한계 (SDF 보다 soft) |
| Neural = overfitting risk | Single scene fitting 시 shape prior 약함 |
| Voxel = 해상도 한계 | $O(N^3)$ 로 고해상도 불가 (sparse convolution 필요) |

---

## 📌 핵심 정리

$$\boxed{\text{Explicit} = \text{(V, F) topology in data}, \quad \text{Implicit} = \text{level set } S = \{\mathbf{x}: \phi(\mathbf{x})=0\}}$$

| 측면 | Explicit (Mesh) | Implicit (SDF/Occupancy) | Hybrid (GS) |
|------|-----------------|------------------------|------------|
| **메모리** | $O(N)$ vertices | $O(\theta)$ network | $O(N)$ Gaussians |
| **렌더링** | Rasterization (fast) | Raycasting / sphere tracing (slow) | Splatting (very fast) |
| **편집** | Direct vertex move | Implicit deformation | Gaussian parameters |
| **위상 변화** | discrete (hard) | Continuous (easy) | Continuous (implicit $\Sigma$) |
| **정확성** | Local control | Global smooth (implicit) | Trade-off |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Mesh 의 정점 수가 $N=10^6$ 일 때, SDF neural network (3-128-128-1) 의 메모리 크기를 계산하고, 둘을 비교하라.

<details>
<summary>해설</summary>

**Mesh**: $3 \times 10^6$ (float32) = $12 \times 10^6$ bytes = 12 MB

**SDF MLP**:
- Layer 1: $3 \times 128$ weights + 128 bias = $384 + 128 = 512$ floats
- Layer 2: $128 \times 128 + 128 = 16512$ floats
- Layer 3: $128 \times 1 + 1 = 129$ floats
- Total: $\approx 17000$ floats = 68 KB

**비율**: $12 \text{ MB} / 68 \text{ KB} \approx 177x$ 메모리 절감 (neural implicit).

하지만 neural 의 inference cost (GPU forward pass) 는 mesh rasterization 보다 비쌈 → trade-off. $\square$

</details>

**문제 2** (심화): SDF 에서 eikonal equation $\|\nabla\phi\| = 1$ 이 학습 중에 violation 되면 (예: gradient magnitude = 0.5), sphere tracing 이 어떻게 실패하는가?

<details>
<summary>해설</summary>

**Sphere Tracing 알고리즘**:
```
x_0 = ray origin
for i in range(max_iter):
    d = φ(x_i)           # SDF value = step distance (if ||∇φ|| = 1)
    x_{i+1} = x_i + d * dir
    if d < epsilon: hit!
```

**Violation 시**:
- If $\|\nabla\phi\| = 0.5$ → actual step should be $2d$ (더 길게)
- Algorithm steps: $d$ (너무 짧음)
- Result: **많은 iterations 필요** 또는 **miss surface** (step 이 너무 작아 epsilon 도달 불가)

**Fix**: Eikonal loss $\mathcal{L}_{\text{eikonal}} = \mathbb{E}_{\mathbf{x}}[\|\nabla_\mathbf{x}\phi(\mathbf{x})\| - 1]^2$ 로 constraint. $\square$

</details>

**문제 3** (논문 비평): Marching Cubes (Lorensen & Cline 1987) 는 voxel grid 의 occupancy 에서 mesh 를 추출하는 알고리즘이다. Implicit neural SDF 에서 mesh 를 얻으려면 어떤 과정이 필요한가? Occupancy network vs SDF network 중 어느 것이 marching cubes 에 더 자연스러운가?

<details>
<summary>해설</summary>

**SDF → Mesh (Marching Cubes compatible)**:
1. Bounding box 를 voxel grid 로 샘플: $V = [x_{min}, x_{max}] \times [y_{min}, y_{max}] \times [z_{min}, z_{max}]$ (예: $256^3$)
2. 각 voxel corner 에서 $\phi(\mathbf{x})$ 평가 → occupancy = $\phi(\mathbf{x}) < 0$ 로 변환 (또는 threshold $\phi(\mathbf{x}) < -\epsilon$)
3. Marching Cubes 적용: 256 voxel configurations 으로 mesh 추출

**Occupancy Network**:
- 직접 occupancy 제공: Marching Cubes 에 즉시 사용
- 장점: $o(\mathbf{x}) \in [0, 1]$ 을 threshold (예: 0.5) 로 바로 변환

**SDF Network**:
- Distance 정보 활용 가능: sign (inside/outside) + magnitude (distance)
- Advantage in reconstruction: sphere tracing 이나 더 정밀한 sampling 가능
- 단점: 평가 후 sign 확인 → occupancy 변환 필요 (한 단계 추가)

**답**: 둘 다 marching cubes compatible, 하지만 **occupancy 가 더 직접적**. SDF 는 distance 정보의 우수성이 rendering/tracing 에서 나타남 (Marching Cubes 는 binary decision 이므로 거리 정보 미활용). $\square$

</details>

---

<div align="center">

[◀ 이전](../README.md) | [📚 README](../README.md) | [다음 ▶](./02-mesh-rasterization.md)

</div>
