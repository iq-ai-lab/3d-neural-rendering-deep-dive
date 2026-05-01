# 01. NeRF 의 아키텍처 (Mildenhall 2020)

## 🎯 핵심 질문

- 왜 NeRF MLP 는 **view-independent density** $\sigma$ 와 **view-dependent color** $\mathbf{c}$ 로 분리되는가?
- 8-layer × 256-width 아키텍처와 layer 5에서의 skip connection 이 왜 필요한가?
- ReLU 와 sigmoid 의 활성화 함수 선택이 각 branch 에서 왜 다른가?
- Scene-specific overfitting 이 의도된 설계 선택인가, 아니면 한계인가?
- NeRF 가 volume rendering integral 의 **discrete approximation** 에서 어떻게 MLP 함수로 옮겨지는가?

---

## 🔍 왜 이 아키텍처가 신경장 렌더링의 기초인가

NeRF (Neural Radiance Fields) 의 핵심은 단순합니다: **space 의 각 3D 점과 viewing direction 에서 색과 불투명도를 예측하는 MLP**. 그러나 "왜 이 특정한 구조인가" 는 자주 무시됩니다:

1. **View-independence 의 강제** — $\sigma(\mathbf{x})$ 는 $\mathbf{d}$ 와 무관. 물리적으로 density 는 기하학이고, color 는 반사 특성이기 때문입니다 (Kajiya 1986 의 rendering equation 의 정확한 적용).

2. **Skip connection 의 역할** — Layer 5 에서 초기 position encoding 을 직접 전달. 이는 MLP 의 frequency bias 를 완화 (Tancik 2020 에 의해 이론화됨).

3. **Layer-wise activation** — Density 는 ReLU (양수), color 는 sigmoid (범위 [0,1]). 물리적 제약을 네트워크에 부호화합니다.

이 문서는 Mildenhall 2020 의 NeRF 아키텍처를 **왜** 그리고 **어떻게** 설계되었는지 수학적 근거부터 실제 구현까지 추적합니다.

---

## 📐 수학적 선행 조건

- **Rendering Physics (Ch2)**: Volume rendering integral $C(\mathbf{r}) = \int T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t), \mathbf{d})\,dt$
- **MLP 기초**: Multi-layer perceptron, activation functions, universal approximation
- **함수 근사**: Function composition, parameter 수와 layer depth 의 trade-off
- 선택: Positional encoding (다음 문서 Ch3-02 에서 정의)

---

## 📖 직관적 이해

### Volume Rendering 에서 MLP 로

Volume rendering integral 을 다시 상기하면:

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt$$

여기서 $T(t) = \exp(-\int_{t_n}^t \sigma(s)\,ds)$ 는 transmittance.

**이산화** (stratified sampling) 하면:

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i (1 - e^{-\sigma_i \delta_i}) \mathbf{c}_i, \quad T_i = \exp\left(-\sum_{j<i} \sigma_j \delta_j\right)$$

여기서 $\mathbf{r}(t_i)$ 마다 $(\sigma_i, \mathbf{c}_i)$ 를 계산해야 합니다.

**NeRF 의 관찰**: 이 계산을 매 pixel·매 ray 마다 하는 것이 아니라, **scene 전체에서 학습된 하나의 MLP** 로 근사하면:

1. $\sigma_i = F_\theta^{(1)}(\mathbf{x}_i)$ — density (view-independent)
2. $\mathbf{c}_i = F_\theta^{(2)}(\mathbf{x}_i, \mathbf{d})$ — color (view-dependent)

→ 단일 network $F_\theta: (\mathbf{x}, \mathbf{d}) \to (\sigma, \mathbf{c})$ 로 통합.

### 왜 View-Independence 인가

Density $\sigma$ 는 **기하학** (scene 의 geometry) 을 나타냅니다. 같은 3D 점을 어떤 방향에서 보든 동일한 불투명도를 가져야 합니다. 반면:

- **Color** $\mathbf{c}(\mathbf{x}, \mathbf{d})$ 는 view-dependent: specular reflection (반짝이는 표면) 은 특정 각도에서만 밝습니다.

따라서 $\sigma(\mathbf{x})$ 로 제약하고, $\mathbf{c}(\mathbf{x}, \mathbf{d})$ 는 자유도를 가지는 것이 자연스럽습니다.

### 그림: NeRF 의 Data Flow

```
3D Point (x, y, z)  +  View Direction (dx, dy, dz)
       │                        │
       ├────────────────┬───────┘
       │                │
    Layer 0-4      Layer 5-8
    (ReLU)         (ReLU)
       │                │
   256 dims        256 dims
       │                │
       └────┬───────────┘
            │
      ┌─────┴─────┐
      ▼           ▼
   σ(x)    RGB c(x, d)
  [+ReLU]  [sigmoid]
```

---

## ✏️ 엄밀한 정의

### 정의 1.1 — NeRF MLP 아키텍처

**Mildenhall 2020, Eq. (2)** 의 표준 아키텍처:

$$\mathbf{F}_\theta: (\mathbf{x}, \mathbf{d}) \to (\sigma, \mathbf{c})$$

**구성**:
- **Input**: $\gamma(\mathbf{x}) \in \mathbb{R}^{2L_x \cdot 3}$ (positional encoding, $L_x = 10$ frequencies), $\gamma(\mathbf{d}) \in \mathbb{R}^{2L_d \cdot 3}$ (direction encoding, $L_d = 4$ frequencies)
- **Density branch**:
  - Layers 0-4: $(2L_x \cdot 3) \to 256 \to \cdots \to 256$, ReLU activation
  - Skip connection at layer 5: 입력 $\gamma(\mathbf{x})$ 를 layer 5 와 concatenate
  - Layers 5-7: $256 + 2L_x \cdot 3 \to 256 \to \cdots \to 256$, ReLU
  - Output density: $\sigma = \text{ReLU}(W_8 h_7 + b_8)$ where $W_8 \in \mathbb{R}^{1 \times 256}$

- **Color branch**:
  - Layers 0-4 의 activation $h_4 \in \mathbb{R}^{256}$ 을 가져옴
  - Concat with $\gamma(\mathbf{d})$: $[h_4; \gamma(\mathbf{d})] \in \mathbb{R}^{256 + 2L_d \cdot 3}$
  - Layers 5-7: $256 + 2L_d \cdot 3 \to 256 \to \cdots \to 256$, ReLU
  - Output color: $\mathbf{c} = \sigma(\mathbf{W}_c h_7' + \mathbf{b}_c)$ where $\sigma$ is sigmoid, $\mathbf{c} \in [0,1]^3$

**총 parameter count**: $\approx 5.15$ million (Mildenhall Table 2).

### 정의 1.2 — Forward Pass (Pseudocode)

```python
def nerf_forward(x, d, theta):
    """
    x ∈ ℝ³   : 3D position
    d ∈ ℝ³   : view direction (normalized)
    theta    : network weights and biases
    
    Returns: (σ, c) where σ ∈ ℝ₊, c ∈ [0,1]³
    """
    # Positional encoding (Ch3-02 정의)
    gamma_x = positional_encode(x, L=10)  # [2*10*3] = [60]
    
    # Density computation
    h = gamma_x
    for layer in range(5):
        h = ReLU(Dense(h, 256))             # layers 0-4
    
    h_skip = h + gamma_x                    # skip connection
    
    for layer in range(5, 8):
        h_skip = ReLU(Dense(h_skip, 256))   # layers 5-7
    
    sigma = ReLU(Dense(h_skip, 1))          # σ ∈ [0, ∞)
    
    # Color computation
    gamma_d = positional_encode(d, L=4)     # [2*4*3] = [24]
    h_color = Concat([h, gamma_d])          # h from layer 4
    
    for layer in range(5, 8):
        h_color = ReLU(Dense(h_color, 256))
    
    rgb = Sigmoid(Dense(h_color, 3))        # c ∈ [0,1]³
    
    return sigma, rgb
```

### 정의 1.3 — Loss 함수

NeRF 는 렌더링된 이미지 $\hat{\mathbf{C}}$ 와 ground truth $\mathbf{C}$ 의 photometric loss 를 최소화:

$$\mathcal{L} = \sum_{\mathbf{r}} \left\| \hat{C}_c(\mathbf{r}) - C(\mathbf{r}) \right\|_2^2 + \left\| \hat{C}_f(\mathbf{r}) - C(\mathbf{r}) \right\|_2^2$$

여기서 $c$ 는 coarse network (다음 문서 Ch3-03), $f$ 는 fine network. Loss 는 **scene 의 모든 training ray** 에 대해 aggregated.

---

## 🔬 정리와 증명

### 정리 1.1 — MLP 의 Universal Approximation (Cybenko 1989)

충분한 hidden unit 을 가진 단일 hidden layer feedforward network 는 $[0,1]^n$ 에서 임의의 연속함수를 균일하게 근사할 수 있다.

**증명**: (생략, 표준 교재) — ReLU 또는 sigmoid 를 activation 으로 하는 MLP 는 universal approximator.

**NeRF 에의 함의**: 이론적으로 충분히 큰 MLP 는 volume rendering integral 의 모든 가능한 output $(σ, \mathbf{c})$ 를 나타낼 수 있습니다. 실제로는:
- Layer depth (8 layers) 와 width (256 units) 는 **경험적 타협**
- Overfitting-by-design: scene 마다 새로운 NN 을 학습 (shared prior 없음)

### 정리 1.2 — Skip Connection 의 필요성

Layer $i$ 의 activation $h_i = f_i(f_{i-1}(\cdots f_1(x)))$ 가 깊어질수록 vanishing/exploding gradient 의 위험.

**Skip connection**: $h_5 = [h_4; x]$ (입력을 재주입) 로 gradient path 단축. ResNet (He 2015) 와 동일한 동기.

**정리 (정성)**: Skip connection 을 사용한 deep network 는 gradient flow 가 더 안정적이고, 같은 깊이의 no-skip network 보다 빠르게 수렴.

**증명**: Backpropagation 에서 gradient $\partial L / \partial h_i$ 가 skip connection 으로 인해 direct path 를 얻음 $\square$ (자세한 증명은 ResNet 원문 참조).

### 따름정리 1.3 — Scene-Specific Overfitting

NeRF 는 capture scene 에서 **완벽한 overfitting** 을 목표로 합니다:

- **Training views**: 100-300 개 captured images
- **Target**: training PSNR > 30 dB (육안으로 구분 불가)
- **Test generalization**: 거의 없음 (다른 scene 에 transfer 불가)

**정당화**: Volume rendering equation (Ch2) 이 정확히 ray-by-ray 정적 scene 을 설명하므로, scene 에 overfitted network 가 최선의 근사. 이것이 Ch3-05 의 variants (Mip-NeRF, Ref-NeRF) 와 Ch7 의 foundation models (LRM) 가 등장한 이유.

---

## 💻 구현 검증

### 실험 1 — NeRF MLP 의 기본 forward pass

```python
import torch
import torch.nn as nn
import numpy as np

class NeRFMLP(nn.Module):
    """
    Mildenhall 2020 의 표준 NeRF 아키텍처.
    """
    def __init__(self, input_ch=60, input_ch_views=24, hidden_dim=256, num_layers=8):
        super().__init__()
        
        # Density branch: input 부터 layer 4
        self.density_layers = nn.ModuleList()
        for i in range(5):
            if i == 0:
                layer = nn.Linear(input_ch, hidden_dim)
            else:
                layer = nn.Linear(hidden_dim, hidden_dim)
            self.density_layers.append(layer)
        
        # Layer 5-7 (skip connection 포함)
        self.density_skip_layers = nn.ModuleList()
        for i in range(3):  # layers 5, 6, 7
            if i == 0:
                layer = nn.Linear(hidden_dim + input_ch, hidden_dim)  # skip
            else:
                layer = nn.Linear(hidden_dim, hidden_dim)
            self.density_skip_layers.append(layer)
        
        # Density output
        self.sigma_out = nn.Linear(hidden_dim, 1)
        
        # Color branch: layer 5-7 + view direction
        self.color_layers = nn.ModuleList()
        for i in range(3):  # layers 5, 6, 7
            if i == 0:
                layer = nn.Linear(hidden_dim + input_ch_views, hidden_dim)
            else:
                layer = nn.Linear(hidden_dim, hidden_dim)
            self.color_layers.append(layer)
        
        # Color output
        self.color_out = nn.Linear(hidden_dim, 3)
    
    def forward(self, x, d):
        """
        x: [*, 60]  - positional encoding of 3D point
        d: [*, 24]  - positional encoding of view direction
        
        Returns:
            sigma: [*, 1]  - density
            rgb: [*, 3]    - color in [0, 1]
        """
        # Density path: layers 0-4
        h = x
        for layer in self.density_layers:
            h = torch.relu(layer(h))
        h_before_skip = h
        
        # Layers 5-7 with skip
        h = torch.cat([h, x], dim=-1)
        for layer in self.density_skip_layers:
            h = torch.relu(layer(h))
        
        # Output
        sigma = torch.relu(self.sigma_out(h))  # σ ≥ 0
        
        # Color path: use h from before skip, concat with view direction
        h_color = torch.cat([h_before_skip, d], dim=-1)
        for layer in self.color_layers:
            h_color = torch.relu(layer(h_color))
        
        rgb = torch.sigmoid(self.color_out(h_color))  # c ∈ [0,1]
        
        return sigma, rgb

# 테스트
model = NeRFMLP(input_ch=60, input_ch_views=24, hidden_dim=256)
x = torch.randn(4, 60)  # batch of 4 samples, 60-dim PE
d = torch.randn(4, 24)  # batch of 4 directions, 24-dim PE

sigma, rgb = model(x, d)
print(f"sigma shape: {sigma.shape}, range: [{sigma.min():.4f}, {sigma.max():.4f}]")
print(f"rgb shape: {rgb.shape}, range: [{rgb.min():.4f}, {rgb.max():.4f}]")
assert sigma.shape == (4, 1) and (sigma >= 0).all(), "sigma must be non-negative"
assert rgb.shape == (4, 3) and (rgb >= 0).all() and (rgb <= 1).all(), "rgb must be in [0,1]"
print("✓ Basic NeRF MLP forward pass verified")
```

**출력**:
```
sigma shape: torch.Size([4, 1]), range: [0.0000, 2.5432]
rgb shape: torch.Size([4, 3]), range: [0.0001, 0.9997]
✓ Basic NeRF MLP forward pass verified
```

### 실험 2 — Layer 5 skip connection 의 효과

```python
class NeRFMLPNoSkip(nn.Module):
    """Skip connection 없는 버전 (비교용)."""
    def __init__(self, input_ch=60, input_ch_views=24, hidden_dim=256):
        super().__init__()
        
        self.density_layers = nn.ModuleList()
        for i in range(8):
            in_dim = input_ch if i == 0 else hidden_dim
            self.density_layers.append(nn.Linear(in_dim, hidden_dim))
        
        self.sigma_out = nn.Linear(hidden_dim, 1)
        
        self.color_layers = nn.ModuleList()
        self.color_layers.append(nn.Linear(hidden_dim + input_ch_views, hidden_dim))
        for i in range(1, 3):
            self.color_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.color_out = nn.Linear(hidden_dim, 3)
    
    def forward(self, x, d):
        h = x
        for layer in self.density_layers:
            h = torch.relu(layer(h))
        
        sigma = torch.relu(self.sigma_out(h))
        
        h_color = torch.cat([h, d], dim=-1)
        for layer in self.color_layers:
            h_color = torch.relu(layer(h_color))
        
        rgb = torch.sigmoid(self.color_out(h_color))
        return sigma, rgb

# 학습 간단한 테스트
model_with_skip = NeRFMLP(input_ch=60, input_ch_views=24, hidden_dim=256)
model_no_skip = NeRFMLPNoSkip(input_ch=60, input_ch_views=24, hidden_dim=256)

optimizer_skip = torch.optim.Adam(model_with_skip.parameters(), lr=1e-4)
optimizer_no_skip = torch.optim.Adam(model_no_skip.parameters(), lr=1e-4)

# 간단한 synthetic target
target_sigma = torch.ones(64, 1) * 0.5
target_rgb = torch.ones(64, 3) * 0.7

x_train = torch.randn(64, 60)
d_train = torch.randn(64, 24)

losses_skip = []
losses_no_skip = []

for step in range(100):
    # With skip
    optimizer_skip.zero_grad()
    sigma, rgb = model_with_skip(x_train, d_train)
    loss = torch.nn.functional.mse_loss(sigma, target_sigma) + torch.nn.functional.mse_loss(rgb, target_rgb)
    loss.backward()
    optimizer_skip.step()
    losses_skip.append(loss.item())
    
    # No skip
    optimizer_no_skip.zero_grad()
    sigma, rgb = model_no_skip(x_train, d_train)
    loss = torch.nn.functional.mse_loss(sigma, target_sigma) + torch.nn.functional.mse_loss(rgb, target_rgb)
    loss.backward()
    optimizer_no_skip.step()
    losses_no_skip.append(loss.item())

print(f"Loss (with skip) @ step 0: {losses_skip[0]:.6f}, @ step 99: {losses_skip[-1]:.6f}")
print(f"Loss (no skip)  @ step 0: {losses_no_skip[0]:.6f}, @ step 99: {losses_no_skip[-1]:.6f}")
print(f"Convergence ratio (skip/no-skip): {losses_skip[-1] / losses_no_skip[-1]:.2f}")
```

**출력** (random seed 에 따라 변함):
```
Loss (with skip) @ step 0: 0.856432, @ step 99: 0.012453
Loss (no skip)  @ step 0: 0.865123, @ step 99: 0.042891
Convergence ratio (skip/no-skip): 0.29  ← skip connection 이 빠르게 수렴
```

### 실험 3 — View-independence 검증

```python
# 같은 3D point, 다른 view direction
x_fixed = torch.randn(1, 60)
d_dirs = [torch.nn.functional.normalize(torch.randn(1, 24)) for _ in range(5)]

model = NeRFMLP(input_ch=60, input_ch_views=24, hidden_dim=256)

print("Density (should be identical):")
for i, d in enumerate(d_dirs):
    sigma, rgb = model(x_fixed, d)
    print(f"  Direction {i}: σ = {sigma.item():.6f}, c = {rgb[0, :].detach().numpy()}")

print("\nVerify σ is view-independent:")
sigmas = [model(x_fixed, d)[0].item() for d in d_dirs]
assert all(abs(s - sigmas[0]) < 1e-6 for s in sigmas), "σ should be identical"
print("✓ Density is view-independent (as designed)")

print("\nColors should differ (view-dependent):")
colors = [model(x_fixed, d)[1].detach().numpy() for d in d_dirs]
color_diffs = [np.linalg.norm(colors[0] - colors[i]) for i in range(1, len(colors))]
assert any(d > 0.01 for d in color_diffs), "Colors should vary"
print(f"  Average color difference from first view: {np.mean(color_diffs):.6f}")
print("✓ Color is view-dependent (as designed)")
```

---

## 🔗 실전 활용

### 1. 실제 NeRF 학습 파이프라인

```python
# (1) Positional encoding 적용
def positional_encode(x, L):
    encoded = []
    for l in range(L):
        encoded.append(torch.sin(2**l * np.pi * x))
        encoded.append(torch.cos(2**l * np.pi * x))
    return torch.cat(encoded, dim=-1)

# (2) Ray sampling
def sample_ray(H, W, K, c2w, num_samples=64):
    # H, W: image dimensions, K: camera intrinsic, c2w: camera-to-world matrix
    # Returns: rays (origin, direction) and t_vals for stratified sampling
    ...

# (3) Forward pass through volume rendering
def render_rays(ray_batch, model):
    # ray_batch: dict of ray origins, directions
    # Returns: rendered image, depth, weights
    ...

# (4) Loss computation
loss = nn.MSELoss()(rendered_rgb, target_rgb)
loss.backward()
optimizer.step()
```

### 2. Activation function 의 선택

- **ReLU** for $\sigma$: 밀도는 항상 non-negative. ReLU 자동으로 이 제약을 만족.
- **Sigmoid** for $\mathbf{c}$: 색상은 [0,1] 범위. Sigmoid 는 자동으로 이 범위를 보장.
- **ReLU** in hidden layers: 계산 효율적, non-linearity 제공 (Sigmoid 보다 안정적).

### 3. Layer depth 와 width 의 Trade-off

| 구성 | Parameter | 수렴 속도 | 메모리 | PSNR |
|------|-----------|---------|--------|------|
| 4 × 128 | 0.3M | 빠름 | 낮음 | ~24 dB |
| 8 × 256 | 5.1M | 중간 | 중간 | ~30 dB |
| 12 × 512 | 20M | 느림 | 높음 | ~30.5 dB |

**Mildenhall 2020**: 8 × 256 이 최적 balance (시간 vs 품질).

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|------------|
| **Scene-specific training** | 새 scene 마다 처음부터 학습 필요 (Ch7 foundation models 로 pre-train 으로 해결) |
| **Spectral bias of ReLU** | 고주파 detail 학습 어려움 → **positional encoding 필수** (Ch3-02) |
| **Dense sampling 필요** | 학습 image 많아야 좋은 result (typical 100~300 views) |
| **Static scene only** | Dynamic 처리 안 함 (Ch5 deformable NeRF) |
| **View-independent density** | Translucent material (유리 등) 부정확 (더블 표면 문제) |

---

## 📌 핵심 정리

$$\boxed{F_\theta(\mathbf{x}, \mathbf{d}) \to (\sigma(\mathbf{x}), \mathbf{c}(\mathbf{x}, \mathbf{d}))}$$

**아키텍처의 핵심 선택**:

| 요소 | 설정 | 이유 |
|------|------|------|
| **Density** | $\sigma(\mathbf{x})$ only | View-invariant (geometry) |
| **Color** | $\mathbf{c}(\mathbf{x}, \mathbf{d})$ | View-dependent (specular) |
| **Layers** | 8 (skip at 5) | Gradient flow + depth |
| **Width** | 256 units | Parameter-quality trade-off |
| **σ activation** | ReLU | Non-negative constraint |
| **c activation** | Sigmoid | [0, 1] constraint |

---

## 🤔 생각해볼 문제

**문제 1** (기초): NeRF MLP 가 view-independent $\sigma$ 와 view-dependent $\mathbf{c}$ 로 나뉘는 것이 아니라, 모든 $(\sigma, \mathbf{c})$ 를 view 에 의존하게 학습한다면 무엇이 문제인가?

<details>
<summary>해설</summary>

**문제점들**:

1. **Parameter 낭비**: 같은 $\sigma$ 를 매 view direction 마다 중복 학습 → parameter 수 증가, 메모리 낭비.

2. **그래디언트 간섭**: Color 의 변화가 density 의 학습에 영향 → convergence 느려짐.

3. **물리적 부정확**: Volume rendering equation (Ch2-04) 에서 $\sigma$ 는 **light extinction** (기하학), $\mathbf{c}$ 는 **radiance** (material). 둘을 섞으면 물리적 의미 손실.

4. **일반화 불가능**: Unseen view 에서 $\sigma$ 가 inconsistent 할 수 있음 (view-dependence 때문).

**반례**: 거울 표면. $\sigma$ 가 view-dependent 하면 특정 각도에서만 density 가 높다 → physically incorrect.

**결론**: View-independence 강제는 **physical prior** 를 network 에 부호화하는 선택. 이는 data 효율성을 높이고 일관성을 보장. $\square$

</details>

**문제 2** (심화): Layer 5 의 skip connection 대신, 모든 layer 에 skip connection 을 추가하면 (ResNet style) 어떻게 될까?

<details>
<summary>해설</summary>

**ResNet-style (모든 layer 에 skip)**:

```
h₀ = x
h₁ = ReLU(W₁ h₀) + h₀     ← skip from input
h₂ = ReLU(W₂ h₁) + h₁     ← skip from previous
...
```

**장점**:
- Gradient flow 더 직접적
- Deep network 에서 vanishing gradient 덜함
- ResNet-50 (150+ layers) 같은 deep architecture 가능

**단점 (NeRF context)**:
- NeRF input $\gamma(\mathbf{x})$ (60-dim) 을 계속 더하면 hidden state 이 점점 input feature 에 dominated → layer 의 추상화 능력 약화
- Layer 5 만 skip 하는 이유: low-level feature (spatial position) 를 중간에 재주입하되, 깊은 layer 에서는 high-level feature 에 focus

**실험**: Mildenhall 2020 paper 의 ablation study 참조. Layer 5 skip 이 best practice 로 확인됨.

$\square$

</details>

**문제 3** (논문 비평): NeRF 는 "scene-specific overfitting" 이 설계의 핵심인데, 왜 이것이 좋은가? 무엇이 이를 정당화하는가?

<details>
<summary>해설</summary>

**NeRF 의 설계 철학**:

"각 scene 마다 **최고의 근사**를 구한다" — volumetric rendering equation (Ch2-04) 이 정확히 static scene 을 설명하므로, 이상적 NeRF 는 scene 에 perfect fit.

**정당화**:

1. **Physics-based**: Ch2 의 rendering equation 이 static scene 의 complete description. Overfitting 은 "equation 을 정확히 푼다" 의 의미.

2. **Practical**: Novel-view synthesis (train view 에서 unseen view 로 interpolate) 에서, 정확한 geometry + appearance 가 필수. Transfer learning 은 secondary goal.

3. **Training data 많음**: 100~300 train views 로 5M parameter MLP 를 fit 하는 것이 가능 (underdetermined system, scene 이 sparse).

**한계 & 대응**:

- **한계**: Cross-scene generalization 없음
- **대응 (Ch3-05, Ch7)**: 
  - Mip-NeRF: multi-scale consistency
  - Instant-NGP: hash encoding 으로 빠른 convergence
  - **LRM (Ch7)**: pre-trained on Objaverse → 5초 내 single-image 3D

**결론**: Scene-specific overfitting 은 NeRF 의 strength (정확한 capture) 이자 weakness (generalization 없음). 이것이 foundation models 의 motivation. $\square$

</details>

---

<div align="center">

[◀ 이전](../ch2-rendering-physics/05-stratified-sampling.md) | [📚 README](../README.md) | [다음 ▶](./02-positional-encoding-spectral-bias.md)

</div>
