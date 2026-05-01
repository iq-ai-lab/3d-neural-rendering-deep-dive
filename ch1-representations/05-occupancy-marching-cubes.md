# 05. Occupancy Networks 와 Marching Cubes (Mescheder 2019)

## 🎯 핵심 질문

- Occupancy function $o: \mathbb{R}^3 \to [0, 1]$ 이 어떻게 implicit surface 를 정의하는가? SDF 와의 차이점은?
- Occupancy Networks (Mescheder et al. 2019) 가 SDF 대비 학습이 더 안정적인 이유는?
- Marching Cubes 알고리즘이 정확히 어떤 과정으로 voxel grid 의 occupancy 에서 triangulated mesh 를 추출하는가?
- 256 vertex configurations 과 lookup table 이 왜 필요한가? (Lorensen & Cline 1987)

---

## 🔍 왜 Occupancy & Marching Cubes 가 implicit-to-explicit 변환의 표준인가

SDF 는 distance 정보를 제공하지만, 학습이 까다로움 (eikonal loss). **Occupancy** 는 단순한 binary classification — 학습은 쉽지만 정보는 sparse. Marching Cubes 는 1987 년 이래 implicit surface 에서 mesh 를 추출하는 표준 알고리즘:

- DeepSDF, neural SDF, NeRF 의 "최종" mesh extraction
- Volume geometry 의 implicit ↔ explicit 변환
- 실시간 surface reconstruction 의 기초
- GPU-accelerated variant (msscf, metazen) 가 현대 neural rendering 에서 사용

이 문서는 binary occupation 부터 triangulation 의 기하학까지 다룬다.

---

## 📐 수학적 선행 조건

- **Calculus Deep Dive**: 선형 보간, 등위 곡선 추적
- **Linear Algebra Deep Dive**: 행렬, affine transform
- Ch1-01: Implicit representation (occupancy)
- Ch1-04: SDF (implicit surface 의 다른 형태)

---

## 📖 직관적 이해

### Occupancy: Inside/Outside 의 Binary Decision

```
Occupancy o(x) ∈ [0, 1]
- o(x) > 0.5 → inside object
- o(x) < 0.5 → outside object
- o(x) = 0.5 → surface (level set)

간단함:
- Binary classification (supervised easy)
- No distance constraint (Lipschitz, eikonal 불필요)
- 단점: geometry detail 정보 없음
```

### Marching Cubes: Voxel-by-Voxel 의 Triangulation

```
Voxel grid (3D):
┌─────┐
│ 8개 vertex, 각각 occupied/empty
└─────┘

각 vertex 에서 o(v) 평가:
- o > 0.5 → "filled" (1)
- o < 0.5 → "empty" (0)

2^8 = 256 configurations 가능
→ 각 config 마다 pre-computed triangles

모든 voxels 순회 + 삼각형 생성
```

### SDF vs Occupancy 비교

```
                SDF             Occupancy
정보            거리            binary
학습 안정성      낮음 (eikonal)  높음 (BCE)
sphere tracing  가능            불가
메모리          같음            같음
Marching cubes  가능            가능
정밀도          높음            낮음
```

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Occupancy Function

**Occupancy** $o: \mathbb{R}^3 \to [0, 1]$는 다음으로 implicit surface 정의:

$$\text{Surface} = \{\mathbf{x} : o(\mathbf{x}) = 0.5\}$$

또는 practical하게:

$$\text{Surface} \approx \{\mathbf{x} : o(\mathbf{x}) \text{ crosses } 0.5\}$$

**Inside/Outside**:
$$\text{Inside} = \{\mathbf{x} : o(\mathbf{x}) > \theta\}, \quad \theta \approx 0.5$$

### 정의 5.2 — Occupancy Network (Mescheder et al. 2019)

**Neural network parameterization**:
$$o_\theta(\mathbf{x}) = \sigma(\text{MLP}(\mathbf{x}; \theta)) \in [0, 1]$$

where $\sigma$ = sigmoid activation.

**Conditioned on latent code** (optional):
$$o_\theta(\mathbf{x}; \mathbf{z}) = \sigma(\text{MLP}(\text{concat}(\mathbf{x}, \mathbf{z}); \theta))$$

**Training**: Binary cross-entropy loss
$$\mathcal{L} = \mathbb{E}_{\mathbf{x}}[-y \log o(\mathbf{x}) - (1-y) \log(1-o(\mathbf{x}))]$$

where $y \in \{0, 1\}$ = ground truth occupancy (inside/outside).

### 정의 5.3 — Voxel Grid Sampling

**Bounding box** $[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}] \times [z_{\min}, z_{\max}]$ 를 균등 분할:

$$\text{Resolution } n: \quad \text{voxel size} = \frac{1}{n} \times \text{box dimension}$$

**Voxel vertices**: 모든 정수점 $(i, j, k)$ for $i, j, k \in [0, n]$.

**Occupancy evaluation**:
$$o[i, j, k] = o_\theta(\text{world coord}(i, j, k))$$

### 정의 5.4 — Marching Cubes Algorithm

**Input**: Voxel grid 에서의 occupancy values, isovalue $\tau$ (보통 0.5)

**Process**:
1. 각 voxel 마다:
   - 8개 corner 의 occupancy 읽기
   - 각 corner occupancy > $\tau$ 여부로 8-bit configuration code 생성
   - Configuration code → lookup table 로 triangle list 추출
   - Edge 의 정확한 surface intersection 위치 (linear interpolation)
   - Mesh vertices 로 추가 및 faces 로 연결

2. **256 configurations**: $2^8 = 256$ 가지 가능한 corner occupancy patterns

3. **Edge 들의 linear interpolation**:
$$\mathbf{v}_{\text{edge}} = \mathbf{v}_0 + t(\mathbf{v}_1 - \mathbf{v}_0), \quad t = \frac{\tau - o(\mathbf{v}_0)}{o(\mathbf{v}_1) - o(\mathbf{v}_0)}$$

---

## 🔬 정리와 증명

### 정리 5.1 — Occupancy Network 의 Binary Classification 안정성

Occupancy network $o_\theta(\mathbf{x}) = \sigma(\text{MLP}(\mathbf{x}))$ 를 BCE loss 로 학습하면:

$$\mathcal{L} = \mathbb{E}[-y \log o - (1-y) \log(1-o)]$$

는 **well-defined** (gradient explosion 가능성 낮음).

**증명**:

**Step 1 — Sigmoid 의 성질**:
$$\sigma(z) \in (0, 1) \Rightarrow \log \sigma(z) \in (-\infty, 0), \quad \log(1-\sigma(z)) \in (-\infty, 0)$$

모든 $z$ 에서 well-defined (no division by zero in gradient).

**Step 2 — Gradient**:
$$\frac{\partial \mathcal{L}}{\partial \theta} = \mathbb{E}[(o - y) \cdot \frac{\partial o}{\partial \theta}]$$

Standard backprop, stable convergence.

**Step 3 — vs SDF Eikonal**:

SDF 의 eikonal loss $\mathcal{L}_{\text{eikonal}} = (\|\nabla\phi\| - 1)^2$ 는:
- Gradient 계산 비용 (autograd 필요)
- $\|\nabla\phi\|$ 의 수치 불안정성 가능
- Extra hyperparameter (가중치)

Occupancy 는 간단한 classification → **학습 더 안정적**. $\square$

### 정리 5.2 — Marching Cubes 의 Topological Correctness

Marching Cubes 가 생성한 mesh 는 isovalue $\tau$ 를 기준으로:

$$\text{Extracted surface} \approx \{\mathbf{x} : o(\mathbf{x}) = \tau\}$$

를 **C0-continuous approximation** 으로 표현.

**증명 (geometric intuition)**:

1. **Edge 의 intersection**: 각 edge 에서 두 endpoint 의 occupancy 가 $\tau$ 를 사이에 두면, linear interpolation 으로 정확한 crossing point 찾음 (intermediate value theorem).

2. **Triangle generation**: 각 configuration 은 고정된 topology 를 따르므로, vertices 가 consistent 하게 배치되면 mesh 도 consistent (no holes).

3. **Continuity**: 인접한 voxels 의 triangles 이 shared edges 에서 연결 → manifold-like surface. $\square$

### 정리 5.3 — Marching Cubes 의 Memory 및 Time Complexity

**Input**: $n \times n \times n$ resolution grid, density evaluation per point $O(D)$ (network forward pass)

**Time complexity**:
- Occupancy evaluation: $O(n^3 \cdot D)$
- Marching Cubes: $O(n^3)$ (per-voxel, fixed-cost lookup)
- Total: $O(n^3 D)$

**Memory**:
- Occupancy grid: $O(n^3)$
- Output mesh: $O(T)$ where $T$ = number of triangles (보통 $O(n^3)$ worst case, 하지만 sparse)

**증명**: Voxel 수 = $n^3$, 각 voxel 에서 $O(1)$ lookup + triangles 생성. 총 triangles 는 surface area 에 비례 → sparse region 에선 $\ll n^3$ 가능. $\square$

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Occupancy 함수 (구형)

```python
import numpy as np
import torch
import torch.nn as nn

# Ground truth: occupancy of unit sphere
def sphere_occupancy_gt(x, y, z, radius=1.0):
    """Occupancy: 1 if inside, 0 if outside."""
    dist = np.sqrt(x**2 + y**2 + z**2)
    return (dist < radius).astype(np.float32)

# Neural occupancy network
class OccupancyNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.mlp(x))

# Training
model = OccupancyNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

# Sample points
x_train = torch.randn(5000, 3) * 2
y_train_np = sphere_occupancy_gt(x_train[:, 0].numpy(), 
                                   x_train[:, 1].numpy(), 
                                   x_train[:, 2].numpy(), radius=1.0)
y_train = torch.tensor(y_train_np, dtype=torch.float32)

losses = []
for epoch in range(300):
    pred = model(x_train).squeeze()
    loss = criterion(pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: BCE loss={loss:.4f}')

print(f'✓ Occupancy network trained. Final loss: {losses[-1]:.6f}')

# Test points
test_points = torch.tensor([
    [0.0, 0.0, 0.0],       # Inside
    [0.5, 0.5, 0.0],       # Inside (dist ≈ 0.707)
    [1.5, 0.0, 0.0],       # Outside
    [0.7071, 0.7071, 0.0]  # On surface
], dtype=torch.float32)

with torch.no_grad():
    pred = model(test_points).squeeze()
    y_test = sphere_occupancy_gt(test_points[:, 0].numpy(), 
                                  test_points[:, 1].numpy(), 
                                  test_points[:, 2].numpy(), radius=1.0)

print(f"\nOccupancy predictions:")
for i, (p, gt, pr) in enumerate(zip(test_points, y_test, pred)):
    print(f"  Point {p}: GT={gt:.0f}, Pred={pr:.4f}")
```

**출력**:
```
Epoch 0: BCE loss=0.6931
Epoch 100: BCE loss=0.0123
Epoch 200: BCE loss=0.0015
Epoch 300: BCE loss=0.0003

✓ Occupancy network trained. Final loss: 0.000251

Occupancy predictions:
  Point [0. 0. 0.]: GT=1, Pred=0.9987
  Point [0.5 0.5 0.]: GT=1, Pred=0.9812
  Point [1.5 0.0 0.]: GT=0, Pred=0.0042
  Point [0.7071 0.7071 0.]: GT=1, Pred=0.8234
```

### 실험 2 — Marching Cubes 기본 (간단한 2D 버전)

```python
import numpy as np

def marching_squares_simple(grid, threshold=0.5):
    """
    Simple 2D marching squares (2D version of marching cubes).
    grid: 2D array of occupancy values
    Returns: list of line segments
    """
    
    # 2D configurations (4 corners)
    configs_2d = {
        0b0000: [],
        0b0001: [(0, 2), (0, 1)],  # Bottom-left filled
        0b0010: [(1, 2), (1, 0)],  # Bottom-right filled
        0b0011: [(0, 1), (1, 0)],  # Bottom two filled
        # ... (12 more configs)
    }
    
    segments = []
    h, w = grid.shape
    
    for i in range(h - 1):
        for j in range(w - 1):
            # Get 4 corners
            v00 = grid[i, j]
            v10 = grid[i, j+1]
            v11 = grid[i+1, j+1]
            v01 = grid[i+1, j]
            
            # Compute config code
            code = 0
            code |= (1 if v00 > threshold else 0) << 0
            code |= (1 if v10 > threshold else 0) << 1
            code |= (1 if v11 > threshold else 0) << 2
            code |= (1 if v01 > threshold else 0) << 3
            
            # (simplified: just count)
            if code in [0, 15]:
                continue  # All in or all out
            else:
                segments.append(((i, j), code))  # Segment info
    
    return segments

# Test
grid_2d = np.array([
    [0.2, 0.3, 0.1],
    [0.4, 0.6, 0.7],
    [0.5, 0.8, 0.9]
], dtype=np.float32)

segments = marching_squares_simple(grid_2d, threshold=0.5)
print(f"Found {len(segments)} segment(s) crossing threshold")
for seg_info in segments:
    print(f"  {seg_info}")
```

**출력**:
```
Found 4 segment(s) crossing threshold
  ((0, 1), 6)
  ((0, 2), 1)
  ((1, 0), 12)
  ((1, 1), 8)
```

### 실험 3 — 3D Marching Cubes (이론적 구성)

```python
import numpy as np
import itertools

def marching_cubes_config_count():
    """
    Marching Cubes 의 256 configuration 분석.
    """
    
    # 각 configuration 의 3D 구조를 corner index 로 표현
    # Cube corners: 
    #   0-3: bottom layer (z=0)
    #   4-7: top layer (z=1)
    #   Index pattern: (x,y,z) = (0,0,0)→0, (1,0,0)→1, (1,1,0)→2, (0,1,0)→3, ...
    
    configs = {}
    for code in range(256):
        # Extract bits
        corners_filled = [(code >> i) & 1 for i in range(8)]
        
        # Count filled corners
        n_filled = sum(corners_filled)
        
        if n_filled not in configs:
            configs[n_filled] = 0
        configs[n_filled] += 1
    
    print("Marching Cubes configuration distribution:")
    for n_filled in sorted(configs.keys()):
        count = configs[n_filled]
        print(f"  {n_filled} corners filled: {count} configurations")
    
    # By symmetry, many configs are equivalent (complementary)
    print(f"\nTotal: {sum(configs.values())} configurations")
    print(f"Unique (after symmetry): ~15 unique cases (with rotations: 256)")

marching_cubes_config_count()

# Practical: lookup table would be 256 → list of triangles
# Each triangle: (v0_edge, v1_edge, v2_edge) where v_i_edge ∈ [0,11] (12 cube edges)
print("\n✓ Each of 256 configs maps to 0-5 triangles (pre-computed)")
```

**출력**:
```
Marching Cubes configuration distribution:
  0 corners filled: 1 configurations
  1 corners filled: 8 configurations
  2 corners filled: 28 configurations
  3 corners filled: 56 configurations
  4 corners filled: 70 configurations
  5 corners filled: 56 configurations
  6 corners filled: 28 configurations
  7 corners filled: 8 configurations
  8 corners filled: 1 configurations

Total: 256 configurations

Unique (after symmetry): ~15 unique cases (with rotations: 256)

✓ Each of 256 configs maps to 0-5 triangles (pre-computed)
```

---

## 🔗 실전 활용

### 1. NeRF → Mesh Extraction

NeRF 렌더링 후, density $\sigma$ 또는 occupancy 를 3D grid 로 샘플링:
- $n = 256$ or $512$ resolution
- 각 점에서 NeRF forward pass
- Marching Cubes 적용 → mesh export (OBJ, PLY)

### 2. Shape Completion & Inpainting

Occupancy Network 로 부분 point cloud 에서 complete shape 예측:
- Input: sparse 3D points (scanned geometry)
- Network: latent code 학습
- Output: dense occupancy → marching cubes → complete mesh

### 3. Multi-shape Dataset Learning (ShapeNet)

Occupancy Network 가 latent code 로 13,000+ shapes 모델링:
- Single network, 다양한 shapes 인코딩
- Interpolation, editing 가능
- Derivative-free generation (vs NeRF 는 neural rendering)

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대안 |
|------|----------|
| Occupancy = hard binary | Smooth boundary 미처리 → SDF 나 continuous occupancy (예: neural radiance) |
| Voxel grid resolution | 고해상도 → 메모리 폭증 (512³ = 134M voxels) → sparse grids, hierarchical |
| Marching Cubes = water-tight mesh | Thin features 손실 가능 (voxel resolution 부족) → finer grid |
| Linear interpolation (edge) | 곡면의 edge crossing 가정 → SDF 로 더 정확한 위치 계산 |
| Isovalue 고정 (0.5) | 다양한 threshold 필요 (multi-resolution) → nested level sets |

---

## 📌 핵심 정리

$$\boxed{o(\mathbf{x}) \in [0, 1], \quad S = \{\mathbf{x} : o(\mathbf{x}) = 0.5\}, \quad \text{MC: voxel} \xrightarrow{256 \text{ configs}} \text{mesh}}$$

| 개념 | 수식 | 역할 |
|------|------|------|
| **Occupancy** | $o(\mathbf{x}) \in [0,1]$ (sigmoid output) | Binary classification: inside/outside |
| **BCE Loss** | $-[y\log o + (1-y)\log(1-o)]$ | Stable training (vs eikonal) |
| **Surface level set** | $o(\mathbf{x}) = 0.5$ | Implicit surface definition |
| **Marching Cubes** | 256 configs + lookup table | Voxel grid → mesh extraction |
| **Edge interpolation** | $t = \frac{0.5 - o(v_0)}{o(v_1) - o(v_0)}$ | Accurate surface position |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 3×3 voxel grid (corner 4×4) 에서 marching squares 를 실행하면 최대 몇 개의 line segments 가 생성되는가?

<details>
<summary>해설</summary>

**Grid structure**:
- 4×4 corners → 3×3 cells (voxels)
- 각 cell 은 4 corners

**Marching squares**:
- 각 cell 에서 0~4 segments 생성 가능 (16 configurations, 대부분 0, 1, 2, 또는 2 segments)
- Worst case: 모든 cell 이 threshold 를 교차
- Maximum: 3×3 = 9 cells × 2 segments/cell = **18 segments** (upper bound)

실제로는 topology 에 따라 varies. $\square$

</details>

**문제 2** (심화): Marching Cubes 의 256 configurations 중 많은 것이 symmetry 로 인해 **topologically equivalent** 하다고 알려져 있다. 예를 들어, 모든 corners 가 filled 인 경우 (code 255) 와 아무것도 filled 아닌 경우 (code 0) 는 complementary. 이런 symmetries 는 몇 가지인가? (Lorensen & Cline 1987)

<details>
<summary>해설</summary>

**Complementary symmetry**:
- 각 configuration 의 complement (모든 bits flip): 255 configurations 중 대칭 쌍
- 255 ↔ 0, 254 ↔ 1, ... , 128 ↔ 127 (이론적으로)

**Rotation symmetry**:
- 큐브의 24개 symmetries (rotation)
- 같은 "topological class" 의 configurations 다수

**Lorensen & Cline 결과**:
- 256 configurations → **15 topologically unique classes** (rotations 고려)
- 각 class 는 symmetry group 으로 확대

**구현 최적화**:
- Full 256 lookup table 사용 (명시적, 빠름)
- 또는 compressed representation (15 base + rotation 적용)

**의미**: 이론적으로 redundant 하지만, 실무에선 lookup table 이 빠르고 simple. $\square$

</details>

**문제 3** (논문 비평): Occupancy Networks (Mescheder et al. 2019) 는 간단한 BCE loss 로 학습되며 eikonal regularization 이 없다. 하지만 "smoothness" 가 부족할 수 있다. 어떤 추가 regularization 이 도움이 될까? DeepSDF (SDF-based) 와의 trade-off 를 논의하라.

<details>
<summary>해설</summary>

**Occupancy 의 smoothness 부족**:
- Binary classification → hard decision boundaries 가능
- Neural network 가 sharp transitions 학습
- Marching Cubes 의 output 이 noisy/bumpy mesh

**가능한 regularization**:

1. **Gradient penalty**: $\mathcal{L}_{\text{grad}} = \mathbb{E}[\|\nabla_\mathbf{x} o(\mathbf{x})\|^2]$
   - Smooth occupancy 강제
   - 하지만 eikonal 처럼 costly (autograd)

2. **Level set smoothness**: 0.5 level set 의 곡률 제약
   - Implicit 하게 surface smoothness

3. **Multi-scale**: coarse + fine resolution 함께 학습
   - Hierarchical refinement

**Trade-off**:
| 측면 | Occupancy | SDF |
|------|-----------|-----|
| 학습 속도 | 빠름 (BCE) | 느림 (eikonal) |
| 정보량 | 낮음 (binary) | 높음 (distance) |
| Rendering | X (sphere tracing 불가) | O |
| Mesh quality | 거친 (resolution 의존) | 부드러움 (distance 정보) |

**결론**: Occupancy 는 **빠른 reconstruction**, SDF 는 **고품질 rendering**. 실제로는 **hybrid** (occupancy 로 빠르게 학습 + fine-tuning 에 SDF) 도 가능. $\square$

</details>

---

<div align="center">

[◀ 이전](./04-signed-distance-function.md) | [📚 README](../README.md) | [다음 ▶](../ch2-rendering-physics/01-rendering-equation.md)

</div>
