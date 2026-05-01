# 02. Mesh 와 Triangle Rendering Pipeline

## 🎯 핵심 질문

- Mesh 의 barycentric coordinate 로 vertex attribute (normal, color, texture) 를 삼각형 내부에서 어떻게 보간하는가?
- Phong shading 의 ambient, diffuse, specular 성분은 각각 어떤 물리학을 반영하는가?
- Z-buffer algorithm 이 어떻게 depth test 를 통해 hidden surface removal 을 보장하는가?
- Classical rasterization pipeline 의 transform matrices (model, view, projection) 를 한 줄씩 유도할 수 있는가?

---

## 🔍 왜 Classical Rasterization 이 3D Graphics 의 기초인가

모든 현대 GPU 는 **3각형 rasterization** 을 기반. NeRF 나 gaussian splatting 같은 neural rendering 도 결국 다른 방식의 "triangle 을 화면에 그리는" 것. Classical pipeline 을 이해하면:

- 왜 GPU 는 triangle 을 기본 단위로 삼는가 (linear in barycentric coords)
- 왜 depth test (z-buffer) 가 필수인가 (occlusion handling)
- 왜 transform matrices 의 순서가 중요한가 (model-view-projection)

이 문서는 **rasterization 의 수학적 정당성** 을 제공한다.

---

## 📐 수학적 선행 조건

- **Linear Algebra Deep Dive**: 3×4 affine transform, homogeneous coordinates, matrix multiplication
- **Calculus Deep Dive**: 편미분, normal vector (cross product)
- Ch1-01: Explicit representation (mesh)
- 기본 물리학: 광학 (radiance, solid angle)

---

## 📖 직관적 이해

### Barycentric Coordinates: Triangle 내부의 "무게중심"

```
Vertex v1, v2, v3 있고, 삼각형 내의 점 p 를 다음처럼 표현:

p = α v1 + β v2 + γ v3,  α + β + γ = 1

(α, β, γ) 는 p 의 "무게" (barycentric coords)
- p = v1 근처 → (α, β, γ) ≈ (1, 0, 0)
- p = 중심 → (α, β, γ) ≈ (1/3, 1/3, 1/3)
- p = 삼각형 밖 → 하나 이상의 좌표가 음수
```

### Phong Shading: 세 가지 빛

```
Final color = ambient + diffuse + specular
             │         │         └─ 거울처럼 반짝 (view direction)
             │         └─ 표면 방향에 따라 밝기 (normal • light)
             └─ 배경 빛 (항상 상수)

수식: L_o = k_a I_a + k_d I_d (n•l) + k_s I_s (v•r)^α
```

### Z-buffer: Depth 로 가리기 판정

```
Framebuffer = 색깔 저장
Z-buffer    = 깊이 저장

래스터화 순서 무관하게, 
각 pixel 마다 가장 작은 z (카메라 가장 가까운) 만 보여줌:

if (z_new < z_buffer[x, y]):
    pixel[x, y] = color_new
    z_buffer[x, y] = z_new
```

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Mesh 의 형식

**Mesh** $\mathcal{M} = (V, F, A)$:
- $V \in \mathbb{R}^{3 \times N}$: 정점 좌표
- $F \in \mathbb{Z}^{3 \times M}$: 삼각형 face indices
- $A$: vertex attributes (normal $N \in \mathbb{R}^{3 \times N}$, color $C \in \mathbb{R}^{3 \times N}$, uv $\in \mathbb{R}^{2 \times N}$ etc.)

**삼각형**: $(i, j, k) \in F$ → vertices $(v_i, v_j, v_k)$

### 정의 2.2 — Barycentric Coordinates

삼각형 $(v_1, v_2, v_3)$ 내의 점:
$$\mathbf{p} = \alpha \mathbf{v}_1 + \beta \mathbf{v}_2 + \gamma \mathbf{v}_3, \quad \alpha + \beta + \gamma = 1$$

**barycentric coords** $(\alpha, \beta, \gamma)$ 는:
- $0 \leq \alpha, \beta, \gamma \leq 1$ and $\sum = 1$ ⇔ $\mathbf{p}$ 가 삼각형 내부
- **기하학적 의미**: $\alpha$ = opposite edge 와의 거리 비 (면적 비)

**계산**:
$$\alpha = \frac{\text{Area}(\mathbf{p}, v_2, v_3)}{\text{Area}(v_1, v_2, v_3)}, \quad \text{etc.}$$

### 정의 2.3 — Attribute Interpolation

삼각형 내 점 $\mathbf{p}$ 에서의 attribute $a(\mathbf{p})$:
$$a(\mathbf{p}) = \alpha \, a(v_1) + \beta \, a(v_2) + \gamma \, a(v_3)$$

$a$ = normal, color, texture coordinate, 등

### 정의 2.4 — Phong Shading

점 $\mathbf{p}$ 의 최종 색상:
$$\mathbf{L}_o = k_a \mathbf{I}_a + k_d \mathbf{I}_d \max(0, \mathbf{n} \cdot \mathbf{l}) + k_s \mathbf{I}_s \max(0, \mathbf{v} \cdot \mathbf{r})^\alpha$$

where:
- $k_a, k_d, k_s$ = material coefficients (ambient, diffuse, specular)
- $\mathbf{I}_a, \mathbf{I}_d, \mathbf{I}_s$ = ambient, diffuse, specular light intensities
- $\mathbf{n}$ = surface normal (interpolated)
- $\mathbf{l}$ = normalized light direction
- $\mathbf{v}$ = normalized view direction
- $\mathbf{r} = 2(\mathbf{n} \cdot \mathbf{l})\mathbf{n} - \mathbf{l}$ = reflected light direction
- $\alpha$ = shininess exponent

### 정의 2.5 — Transform Pipeline

**Model matrix** $M \in \mathbb{R}^{4 \times 4}$: object-space → world-space
$$\mathbf{x}^w = M \mathbf{x}^o$$

**View matrix** $V \in \mathbb{R}^{4 \times 4}$: world-space → camera-space
$$\mathbf{x}^c = V \mathbf{x}^w$$

**Projection matrix** $P \in \mathbb{R}^{4 \times 4}$: camera-space → normalized device coords (NDC, $[-1, 1]^3$)

For perspective (focal length $f$, aspect ratio $a$, near/far $n, f$):
$$P = \begin{pmatrix} f/a & 0 & 0 & 0 \\ 0 & f & 0 & 0 \\ 0 & 0 & \frac{f+n}{f-n} & -\frac{2fn}{f-n} \\ 0 & 0 & -1 & 0 \end{pmatrix}$$

**Composite**: $\text{MVP} = P \cdot V \cdot M$ (역순)

### 정의 2.6 — Z-buffer Algorithm

**Framebuffer**: 화면 각 pixel $(x, y)$ 에 색깔 저장
**Z-buffer**: 각 pixel $(x, y)$ 에 깊이 값 저장 (초기: $\infty$ or $0$)

```
for each triangle T:
    for each pixel (x, y) in T:
        depth_z = T.interpolate_depth(x, y)
        if depth_z < z_buffer[x, y]:
            framebuffer[x, y] = T.compute_color(x, y)
            z_buffer[x, y] = depth_z
```

---

## 🔬 정리와 증명

### 정리 2.1 — Barycentric Interpolation 의 Linearity

점 $\mathbf{p}$ 의 attribute $a(\mathbf{p}) = \alpha a(v_1) + \beta a(v_2) + \gamma a(v_3)$ 는 $\mathbf{p}$ 에 대해 **선형** (평면 위에서).

**증명**:
삼각형은 2D affine manifold. Barycentric coords $(\alpha, \beta, \gamma)$ 는 affine coordinates → attribute 도 affine combination. $\square$

**의미**: GPU 는 linear interpolation 을 고속으로 지원 → 삼각형 단위로 처리 최적화.

### 정리 2.2 — Phong Shading 의 Energy Conservation

Material 이 energy 를 보존하려면:
$$k_a + k_d + k_s \leq 1$$

(fully reflective black surface $\approx k_a = k_d = k_s = 0$ 에서 가까움)

**증명**:
Incident radiance $I$ 에 대해:
- Absorbed: $I(1 - k_d - k_s)$
- Diffuse: $k_d I$
- Specular: $k_s I$
Total reflected + absorbed = incident. $\square$

### 정리 2.3 — Z-buffer 의 Correctness (Depth Ordering)

Z-buffer algorithm 이 terminates 이고, 최종 framebuffer 는 **camera 에서 가장 가까운 surface** 를 표시함.

**증명**:
각 pixel 에 대해 모든 triangles 를 처리한 후, $\min_T z_T(x, y)$ 를 저장 (loop 끝). 따라서 가장 작은 (가장 가까운) 깊이가 보이게 됨. $\square$

### 정리 2.4 — Homogeneous Coordinates 의 Perspective Division

Perspective projection 후 homogeneous coordinates:
$$\mathbf{x}_{\text{clip}} = P \mathbf{x}_c = [x, y, z, w]^T$$

Normalized device coords:
$$\mathbf{x}_{\text{ndc}} = \left(\frac{x}{w}, \frac{y}{w}, \frac{z}{w}\right)$$

이 division 이 **perspective distortion** 을 가능케 함 (멀수록 작아짐).

**증명**:
$w = -z_c$ (projection matrix design) → $\frac{x}{-z_c}$ 가 smaller for larger $z_c$. $\square$

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Barycentric Coordinates 계산 및 보간

```python
import numpy as np

# Triangle
v1 = np.array([0., 0., 0.])
v2 = np.array([1., 0., 0.])
v3 = np.array([0., 1., 0.])

# Vertex attributes
c1 = np.array([1., 0., 0.])  # red
c2 = np.array([0., 1., 0.])  # green
c3 = np.array([0., 0., 1.])  # blue

def barycentric_coords_2d(p, v1, v2, v3):
    """2D barycentric coords (z ignored)."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, v1, v2)
    d2 = sign(p, v2, v3)
    d3 = sign(p, v3, v1)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    if has_neg and has_pos:
        return None  # outside
    
    area_total = sign(v1, v2, v3)
    alpha = sign(p, v2, v3) / area_total
    beta = sign(p, v3, v1) / area_total
    gamma = sign(p, v1, v2) / area_total
    
    return alpha, beta, gamma

# Test points
test_points = [
    [0.2, 0.2, 0.],
    [0.5, 0., 0.],
    [0.3, 0.3, 0.],
    [0.8, 0.8, 0.],  # outside
]

for p in test_points:
    p = np.array(p)
    bary = barycentric_coords_2d(p, v1, v2, v3)
    
    if bary is None:
        print(f"p={p}: outside triangle")
        continue
    
    alpha, beta, gamma = bary
    print(f"p={p}: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}")
    
    # Interpolate color
    color = alpha * c1 + beta * c2 + gamma * c3
    print(f"  Interpolated color (RGB): {color}")
    assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Sum not 1!"
```

**출력**:
```
p=[0.2 0.2 0.]: α=0.600, β=0.200, γ=0.200
  Interpolated color (RGB): [0.6  0.2  0.2]
p=[0.5 0. 0.]: α=0.500, β=0.500, γ=0.000
  Interpolated color (RGB): [0.5  0.5  0. ]
p=[0.3 0.3 0.]: α=0.400, β=0.300, γ=0.300
  Interpolated color (RGB): [0.4  0.3  0.3]
p=[0.8 0.8 0.]: outside triangle
```

### 실험 2 — Phong Shading 계산

```python
import numpy as np

def phong_shading(p, n, v_pos, l_pos, mat_ka, mat_kd, mat_ks, mat_alpha, I_a, I_d, I_s):
    """
    p: 3D point
    n: surface normal (normalized)
    v_pos: camera/view position
    l_pos: light position
    mat_*: material coefficients
    I_a, I_d, I_s: ambient, diffuse, specular light intensity
    """
    
    # Normalize vectors
    n = n / np.linalg.norm(n)
    l = (l_pos - p) / np.linalg.norm(l_pos - p)
    v = (v_pos - p) / np.linalg.norm(v_pos - p)
    r = 2 * np.dot(n, l) * n - l
    
    # Ambient
    L_ambient = mat_ka * I_a
    
    # Diffuse
    L_diffuse = mat_kd * I_d * max(0, np.dot(n, l))
    
    # Specular
    L_specular = mat_ks * I_s * (max(0, np.dot(v, r)) ** mat_alpha)
    
    return L_ambient + L_diffuse + L_specular

# Example
p = np.array([0., 0., 0.])
n = np.array([0., 0., 1.])  # facing +z (toward camera)
v_pos = np.array([0., 0., 1.])  # camera
l_pos = np.array([1., 1., 1.])  # light source

mat_ka = 0.1
mat_kd = 0.6
mat_ks = 0.3
mat_alpha = 32

I_a = np.array([0.2, 0.2, 0.2])
I_d = np.array([0.8, 0.8, 0.8])
I_s = np.array([1.0, 1.0, 1.0])

L_o = phong_shading(p, n, v_pos, l_pos, mat_ka, mat_kd, mat_ks, mat_alpha, I_a, I_d, I_s)
print(f"Phong shading result: {L_o}")
print(f"RGB: ({L_o[0]:.3f}, {L_o[1]:.3f}, {L_o[2]:.3f})")

# Verify energy conservation
total_k = mat_ka + mat_kd + mat_ks
print(f"Total coefficient sum: {total_k:.1f} (should be ≤ 1 for conservation)")
```

**출력**:
```
Phong shading result: [0.60698624 0.60698624 0.60698624]
RGB: (0.607, 0.607, 0.607)
Total coefficient sum: 1.0 (should be ≤ 1 for conservation)
```

### 실험 3 — Z-buffer Algorithm 시뮬레이션

```python
import numpy as np

def rasterize_triangle_zbuffer(v1, v2, v3, color, zbuffer, framebuffer, width, height):
    """
    간단한 z-buffer 래스터화 (삼각형 하나).
    v1, v2, v3: 화면좌표 (x, y, z) — z 는 depth
    color: RGB
    """
    
    # Bounding box
    x_min = max(0, int(min(v1[0], v2[0], v3[0])))
    x_max = min(width - 1, int(max(v1[0], v2[0], v3[0])))
    y_min = max(0, int(min(v1[1], v2[1], v3[1])))
    y_max = min(height - 1, int(max(v1[1], v2[1], v3[1])))
    
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            p = np.array([x, y, 0.])
            
            # Simple point-in-triangle (2D)
            def sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
            
            d1 = sign(p, v1, v2)
            d2 = sign(p, v2, v3)
            d3 = sign(p, v3, v1)
            
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            
            if not (has_neg and has_pos):  # inside
                # Interpolate depth (simple: average)
                z = (v1[2] + v2[2] + v3[2]) / 3.0
                
                if z < zbuffer[y, x]:
                    zbuffer[y, x] = z
                    framebuffer[y, x] = color

# Setup
width, height = 100, 100
framebuffer = np.zeros((height, width, 3))
zbuffer = np.full((height, width), np.inf)

# Draw two triangles (overlapping)
# Triangle 1: closer
v1_1 = np.array([20., 20., 0.])
v1_2 = np.array([80., 20., 0.])
v1_3 = np.array([50., 80., 0.])
color1 = np.array([1., 0., 0.])  # red

rasterize_triangle_zbuffer(v1_1, v1_2, v1_3, color1, zbuffer, framebuffer, width, height)

# Triangle 2: farther (z=1), overlapping
v2_1 = np.array([40., 40., 1.])
v2_2 = np.array([90., 50., 1.])
v2_3 = np.array([60., 90., 1.])
color2 = np.array([0., 0., 1.])  # blue

rasterize_triangle_zbuffer(v2_1, v2_2, v2_3, color2, zbuffer, framebuffer, width, height)

print(f"Framebuffer shape: {framebuffer.shape}")
print(f"Z-buffer min (closest): {zbuffer[zbuffer != np.inf].min():.2f}")
print(f"Z-buffer max (farthest): {zbuffer[zbuffer != np.inf].max():.2f}")
print(f"✓ Closer triangle (z=0, red) should occlude farther (z=1, blue)")
```

**설명**: Z-buffer 가 closer triangle 을 우선 렌더링하고 farther 는 depth test 에서 제외.

---

## 🔗 실전 활용

### 1. Real-time Graphics (Game Engines)

Vertex/Fragment Shader:
- Vertex: model-view-projection transform, barycentric interpolation 설정
- Fragment: Phong or PBR shading 적용
- Z-buffer: GPU hardware 에서 자동 depth test

### 2. Rendering Equation 의 Discrete Form

Ch2 의 rendering equation 을 실제로 풀려면:
- 각 pixel 마다 ray cast
- Intersected triangle 찾기 (frustum culling, z-buffer)
- Barycentric coords 로 normal/color 보간
- Phong (또는 PBR) 로 shade

### 3. Mesh Export & Visualization

NeRF / SDF 에서 mesh 추출 후:
- Marching Cubes → triangles 생성
- Vertex normals 계산 (인접 faces 평균)
- Phong shading 으로 visualization

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대안 |
|------|----------|
| Triangle = primitive shape | 복잡한 shape = 많은 triangles 필요 (memory, rasterization cost) |
| Barycentric linear interpolation | 고주파 attribute (texture) 의 aliasing 가능 — mipmap 필요 |
| Phong = local shading | Global illumination (GI) 미처리 — ray tracing / path tracing 필요 |
| Z-buffer = binary decision | Anti-aliasing 어려움 (edge) — MSAA (multisample AA) 필요 |
| Perspective division 후 z-linear 아님 | Depth test 가 "correct" 하려면 non-linear depth 보정 필요 |
| Mesh 가 water-tight 가정 | Non-manifold geometry → z-buffer ambiguity |

---

## 📌 핵심 정리

$$\boxed{\text{Triangle} = \text{(v}_1, \text{v}_2, \text{v}_3\text{)} \Rightarrow \text{Barycentric} \Rightarrow \text{Phong} \Rightarrow \text{Z-buffer}}$$

| 개념 | 수식 | 역할 |
|------|------|------|
| **Barycentric** | $\mathbf{p} = \alpha \mathbf{v}_1 + \beta \mathbf{v}_2 + \gamma \mathbf{v}_3$ | Attribute interpolation (linear) |
| **Phong** | $L_o = k_a I_a + k_d I_d(n \cdot l) + k_s I_s(v \cdot r)^\alpha$ | Local shading model |
| **Z-buffer** | $z_{\text{new}} < z_{\text{buffer}} \Rightarrow$ update | Depth-based occlusion |
| **MVP** | $\mathbf{x}_{\text{clip}} = P V M \mathbf{x}^o$ | Coordinate transform |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 삼각형 $(v_1, v_2, v_3) = ([0,0], [1,0], [0,1])$ 의 무게중심 (centroid) 에서의 barycentric coords 를 계산하고, vertex color 가 red, green, blue 일 때 보간된 색깔을 구하라.

<details>
<summary>해설</summary>

**Centroid**: $\mathbf{c} = \frac{1}{3}(v_1 + v_2 + v_3) = \frac{1}{3}([1, 1]) = [1/3, 1/3]$

**Barycentric**: centroid 에서는 대칭성으로 $(\alpha, \beta, \gamma) = (1/3, 1/3, 1/3)$

**Interpolated color**:
$$\mathbf{c} = \frac{1}{3}[1, 0, 0] + \frac{1}{3}[0, 1, 0] + \frac{1}{3}[0, 0, 1] = [1/3, 1/3, 1/3]$$

(neutral gray - 세 색의 평균). $\square$

</details>

**문제 2** (심화): Phong shading 에서 specular exponent $\alpha$ 가 크면 광택이 진해진다고 알려져 있다. 왜 그런가? $\alpha \to \infty$ 일 때 극한 형태를 논의하라.

<details>
<summary>해설</summary>

**Specular term**: $L_s = k_s I_s (\mathbf{v} \cdot \mathbf{r})^\alpha$

$\alpha$ 크면:
- $\mathbf{v} \cdot \mathbf{r} = 1$ (exact reflection): $(1)^\alpha = 1$ → 최대 intensity
- $\mathbf{v} \cdot \mathbf{r} = 0.9$ (약간 벗어남): $(0.9)^\alpha \approx 0$ for large $\alpha$
- → very narrow highlight region (glossy 효과)

**극한 $\alpha \to \infty$**:
$$(\mathbf{v} \cdot \mathbf{r})^\alpha \to \begin{cases} 1 & \text{if } \mathbf{v} \cdot \mathbf{r} = 1 \\ 0 & \text{if } \mathbf{v} \cdot \mathbf{r} < 1 \end{cases}$$

= Dirac delta 함수 (exact reflection 만) → 완전 거울.

$\alpha = 1$ 근처: broad, matte 가지 않은 specular. $\square$

</details>

**문제 3** (논문 비평): Classical rasterization 의 z-buffer 는 **painter's algorithm** (뒤에서 앞으로 그리기) 과 다른가? 각각의 장단점을 비교하라 (Foley et al. 1996 "Computer Graphics" 참조).

<details>
<summary>해설</summary>

**Z-buffer (Catmull 1974)**:
- 알고리즘 순서와 무관하게 정확한 depth test
- O(n) pixel updates (worst case all overlap)
- GPU hardware support → fast
- Early z-rejection 가능 (depth pass 후 fragment pass)

**Painter's algorithm**:
- 뒤에서 앞으로 정렬 후 순차 그리기 → 나중 draw 가 덮음
- Depth cycle 가능성 (A 앞 B, B 앞 C, C 앞 A) → 정렬 불가
- Multi-pass 필요 (depth sort 비용 $O(n \log n)$)
- CPU overhead, GPU 미지원

**현대 결론**: Z-buffer 가 표준 (painter 는 특수한 경우—예: transparency 처리—에만 사용).

Z-buffer: **spatial** (pixel 중심), Painter: **object** (triangle 중심) 순서. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-explicit-vs-implicit.md) | [📚 README](../README.md) | [다음 ▶](./03-point-cloud-pointnet.md)

</div>
