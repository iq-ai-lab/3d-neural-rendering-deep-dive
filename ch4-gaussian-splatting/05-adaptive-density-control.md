# 05. Adaptive Density Control — Clone · Split · Prune

## 🎯 핵심 질문

- 3DGS 학습 중 Gaussian 의 개수는 왜 **고정되지 않고** 동적으로 증가/감소하는가?
- View-space gradient $\|\nabla_{\mu_{2D}} \mathcal{L}\|$ 가 크면 뭔가 빠졌다는 신호인데, 이를 **Clone** (복제) 과 **Split** (분할) 으로 어떻게 대응하는가?
- **Pruning** (제거) 은 opacity α < ε 인 Gaussian 을 언제, 왜 제거하는가?
- 주기적인 **opacity reset** 은 무엇인가?

---

## 🔍 왜 Adaptive Density Control 이 필수인가

NeRF 와 달리 3DGS 는 **명시적 점 집합** 으로 표현합니다. 따라서:

**초기 상태의 문제**:
- COLMAP point cloud 로 시작 → 원래부터 sparse (coverage 부족)
- 첫 몇 iteration 에서는 어떤 영역이 "underfit" 인지 모름

**학습 중 현상**:
- 높은 gradient 영역: detail 표현 부족 → Gaussian 추가 필요
- 낮은 opacity 영역: 기여도 거의 없음 → 정리 필요
- Overfitting 경향: 매우 크고 불투명한 Gaussian → 너무 specific

**3DGS 의 솔루션: Adaptive Density Control**
1. **Clone**: gradient 크고 작은 Gaussian → 복제
2. **Split**: gradient 크고 큰 Gaussian → 반으로 분할 (더 정교한 표현)
3. **Prune**: opacity 매우 낮음 → 제거
4. **Opacity Reset**: 주기적으로 α 를 중간값으로 초기화 (re-balance)

결과: **최적의 점 개수**, **균형잡힌 fitting**.

---

## 📐 수학적 선행 조건

- 최적화: gradient 크기, Hessian
- 선형대수: covariance 의 크기 (norm, trace, determinant)
- 확률론: Gaussian mixture 의 해석
- 문서 01, 04 참조: Gaussian parameterization, opacity

---

## 📖 직관적 이해

### Gradient 를 기준으로 한 "부족함" 판단

Loss $\mathcal{L}$ 에서 각 Gaussian i 의 2D 위치에 대한 gradient:
$$\mathbf{g}_i = \nabla_{\mu_{2D,i}} \mathcal{L}$$

**의미**:
- $\|\mathbf{g}_i\|$ 크다 = pixel 값이 민감함 = **세부 표현 부족** (더 많은 Gaussian 필요)
- $\|\mathbf{g}_i\|$ 작다 = pixel 변화 무관 = 이 Gaussian 무의미

**Threshold**: τ (hyperparameter, e.g., 0.0002) 를 설정.
- $\|\mathbf{g}_i\| > \tau$ : clone 또는 split 의 candidate

### Clone vs Split

두 전략 모두 "Gaussian 을 추가" 하지만:

**Clone** (작은 Gaussian):
- 현재 Gaussian 을 **그대로 복제**
- 같은 위치, 같은 크기 (s 동일)
- **목적**: 공간적 해상도 높이기 (locally dense representation)

**Split** (큰 Gaussian):
- 현재 Gaussian 을 **축소 & 분할**
  - 원본: 크기 s
  - 분할된 2개: 각각 크기 s/1.6 (약 63%)
  - 원본 제거, 분할된 2개 추가
- **목적**: covariance 가 이미 커서 더 정교한 분해 필요

**판단 기준**:
$$\|\mathbf{s}_i\| = \sqrt{s_x^2 + s_y^2 + s_z^2}$$

- $\|\mathbf{s}_i\|$ **작음** (e.g., < 0.1): Clone (세밀한 디테일)
- $\|\mathbf{s}_i\|$ **큼** (e.g., > 0.5): Split (큰 영역을 세분화)

### Pruning: Opacity 기준

매 N iteration 마다:
- $\alpha_i < \epsilon$ 인 Gaussian 제거 (ε = 0.005 등)
- **이유**: 거의 투명한 Gaussian 은 렌더링에 기여 무시 → memory 낭비

### Opacity Reset

매 N iteration 마다:
- 모든 Gaussian 의 opacity 를 중간값 (e.g., 0.5) 으로 초기화
- **이유**: 일부 Gaussian 이 α = 1 (완전 불투명) 에 stuck 될 수 있음
  - Reset 없으면 → 해당 Gaussian 뒤의 점들은 영원히 gradient 못 받음
  - Reset 하면 → 모든 Gaussian 이 "재경쟁" 할 기회

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Clone Condition

Iteration $t$ 에서, Gaussian $i$ 를 clone 하는 조건:

$$\|\nabla_{\boldsymbol{\mu}_{2D,i}} \mathcal{L}\| > \tau \quad \text{AND} \quad \|\mathbf{s}_i\| < s_{\text{threshold}}$$

**후처리**:
- 새 Gaussian $i'$ 을 생성: $\boldsymbol{\mu}_{i'} = \boldsymbol{\mu}_i$, $\mathbf{q}_{i'} = \mathbf{q}_i$, $\mathbf{s}_{i'} = \mathbf{s}_i$
- 모든 속성 복제 (색, opacity 포함)

### 정의 1.2 — Split Condition

$$\|\nabla_{\boldsymbol{\mu}_{2D,i}} \mathcal{L}\| > \tau \quad \text{AND} \quad \|\mathbf{s}_i\| > s_{\text{threshold}}$$

**후처리**:
- 기존 Gaussian $i$ 제거
- 새 Gaussian $i_1, i_2$ 생성:
  - $\boldsymbol{\mu}_{i_1}, \boldsymbol{\mu}_{i_2}$ = $\boldsymbol{\mu}_i$ 근처 (작은 offset)
  - $\mathbf{s}_{i_1} = \mathbf{s}_{i_2} = \mathbf{s}_i / 1.6$ (축소된 크기)
  - $\mathbf{q}_{i_1} = \mathbf{q}_{i_2} = \mathbf{q}_i$ (회전 유지)
  - 색, opacity 각각 약간의 noise 추가 (gradient signal 다양화)

### 정의 1.3 — Pruning Condition

매 $N$ iteration 마다:
$$\text{remove } i \quad \text{if} \quad \alpha_i < \epsilon_{\text{prune}}$$

(e.g., $\epsilon_{\text{prune}} = 0.005$)

### 정의 1.4 — Opacity Reset

매 $N$ iteration 마다 (e.g., N=3000):
$$\alpha_i \leftarrow 0.5 \quad \text{for all } i$$

---

## 🔬 정리와 증명

### 정리 1.1 — Gradient Magnitude 의 역할

**명제**: Clone/Split 의 threshold τ 를 적절히 설정하면, Gaussian density 는 Loss landscape 의 변화에 자동으로 적응.

**증명 아이디어** (엄밀하지 않음, 직관):

Loss $\mathcal{L}$ 를 Taylor 전개하면:
$$\mathcal{L}(\boldsymbol{\mu} + \Delta\boldsymbol{\mu}) \approx \mathcal{L}(\boldsymbol{\mu}) + \nabla \mathcal{L} \cdot \Delta\boldsymbol{\mu} + \frac{1}{2} \Delta\boldsymbol{\mu}^T H \Delta\boldsymbol{\mu}$$

High $\|\nabla \mathcal{L}\|$ = first-order term 이 큼 = **현재 Gaussian 으로 fit 하기 어려운 영역** → Gaussian 추가 필요.

Adaptive 전략이 자동으로 이를 감지. ∎

### 정리 1.2 — Split 의 분산 보존

Split 후 분할된 두 Gaussian:
- Original: mean μ, cov Σ = R S S^T R^T
- Split: mean μ ± δ, cov Σ' = R (S/1.6)(S/1.6)^T R^T

**분산 비율**:
$$\|Σ'\| = \|Σ\| / 2.56 \approx \|Σ\| / 2.5$$

(각 축이 1.6배 줄어들면, 체적은 1.6³ ≈ 4 배 줄어드나, 우리는 2개 추가 → net effect ~1/2.5)

**의미**: 전체 "coverage" 는 유지, 하지만 더 정교한 분해.

### 정리 1.3 — Opacity Reset 의 정당성

Reset 전:
$$\alpha_1 = 1.0, \alpha_2 = 1.0, \alpha_3 = 0.1$$

Rendering: C = α₁ c₁ + (1-α₁) α₂ c₂ + ... = c₁ (α₂, α₃ 무시됨)

Reset 후:
$$\alpha_1 = 0.5, \alpha_2 = 0.5, \alpha_3 = 0.5$$

각 Gaussian 이 "공정하게" 25% 기여 → gradient 모두 학습 가능.

---

## 💻 실전 구현 (PyTorch)

### 실험 1 — Gradient 계산 및 Clone/Split 판단

```python
import torch
import torch.nn as nn

def evaluate_densification_candidates(
    positions_3d, scales, rotations, 
    gradients_2d, 
    s_threshold_clone=0.1, s_threshold_split=0.5,
    grad_threshold=0.0002
):
    """
    각 Gaussian 에 대해 Clone/Split/Keep 판단
    
    Args:
        positions_3d: (N, 3) 3D positions
        scales: (N, 3) scaling vectors
        rotations: (N, 4) quaternions
        gradients_2d: (N, 2) 2D position gradients
        s_threshold_clone, s_threshold_split: float
        grad_threshold: float
    
    Return:
        actions: (N,) tensor, 0=keep, 1=clone, 2=split
    """
    N = positions_3d.shape[0]
    
    # Gradient magnitude
    grad_norm = torch.norm(gradients_2d, dim=1)  # (N,)
    
    # Scale magnitude
    scale_norm = torch.norm(scales, dim=1)  # (N,)
    
    actions = torch.zeros(N, dtype=torch.long)
    
    # Clone condition
    clone_mask = (grad_norm > grad_threshold) & (scale_norm < s_threshold_clone)
    actions[clone_mask] = 1
    
    # Split condition (only if not clone)
    split_mask = (grad_norm > grad_threshold) & (scale_norm > s_threshold_split) & (actions == 0)
    actions[split_mask] = 2
    
    return actions

# Test
N = 10
pos_3d = torch.randn(N, 3)
pos_3d[:, 2] = torch.abs(pos_3d[:, 2]) + 2.0

scales = torch.abs(torch.randn(N, 3)) + 0.01
rotations = torch.randn(N, 4)
rotations = rotations / torch.norm(rotations, dim=1, keepdim=True)

grad_2d = torch.randn(N, 2)

actions = evaluate_densification_candidates(
    pos_3d, scales, rotations, grad_2d,
    s_threshold_clone=0.1, s_threshold_split=0.5,
    grad_threshold=0.0001
)

print(f"Actions: {actions}")
print(f"Clone candidates: {(actions == 1).sum().item()}")
print(f"Split candidates: {(actions == 2).sum().item()}")
```

### 실험 2 — Clone & Split 실행

```python
def execute_densification(
    positions_3d, scales, rotations, colors, opacities,
    actions
):
    """
    Clone/Split 실행
    
    Args:
        positions_3d, scales, rotations, colors, opacities: Gaussian parameters
        actions: (N,) tensor, 0=keep, 1=clone, 2=split
    
    Return:
        new_positions, new_scales, new_rotations, new_colors, new_opacities
    """
    N = positions_3d.shape[0]
    
    # Collect new Gaussians
    new_pos, new_scale, new_rot, new_color, new_opacity = [], [], [], [], []
    
    for i in range(N):
        if actions[i] == 0:
            # Keep
            new_pos.append(positions_3d[i])
            new_scale.append(scales[i])
            new_rot.append(rotations[i])
            new_color.append(colors[i])
            new_opacity.append(opacities[i])
        
        elif actions[i] == 1:
            # Clone: add a duplicate
            new_pos.append(positions_3d[i])
            new_scale.append(scales[i])
            new_rot.append(rotations[i])
            new_color.append(colors[i])
            new_opacity.append(opacities[i])
            
            # Duplicate
            new_pos.append(positions_3d[i])
            new_scale.append(scales[i])
            new_rot.append(rotations[i])
            new_color.append(colors[i])
            new_opacity.append(opacities[i])
        
        elif actions[i] == 2:
            # Split: replace with 2 smaller Gaussians
            offset = 0.05  # Small spatial offset
            delta = torch.randn(3) * 0.01
            
            scale_split = scales[i] / 1.6
            
            # First split
            new_pos.append(positions_3d[i] + delta)
            new_scale.append(scale_split)
            new_rot.append(rotations[i])
            new_color.append(colors[i] + torch.randn(colors[i].shape) * 0.01)
            new_opacity.append(opacities[i])
            
            # Second split
            new_pos.append(positions_3d[i] - delta)
            new_scale.append(scale_split)
            new_rot.append(rotations[i])
            new_color.append(colors[i] + torch.randn(colors[i].shape) * 0.01)
            new_opacity.append(opacities[i])
    
    return (
        torch.stack(new_pos) if new_pos else torch.empty((0, 3)),
        torch.stack(new_scale) if new_scale else torch.empty((0, 3)),
        torch.stack(new_rot) if new_rot else torch.empty((0, 4)),
        torch.stack(new_color) if new_color else torch.empty((0, 3)),
        torch.stack(new_opacity) if new_opacity else torch.empty((0,)),
    )

# Test
actions = torch.tensor([0, 1, 2, 0, 1])
colors = torch.randn(5, 3)
opacities = torch.rand(5)

new_pos, new_scale, new_rot, new_color, new_opacity = execute_densification(
    pos_3d[:5], scales[:5], rotations[:5], colors, opacities,
    actions
)

print(f"Original: {pos_3d[:5].shape[0]} Gaussians")
print(f"After densification: {new_pos.shape[0]} Gaussians")
print(f"Expected: 5 (keep) + 1 clone (2) + 1 split (2) = 9")
```

### 실험 3 — Pruning & Opacity Reset

```python
def prune_and_reset_opacity(
    positions_3d, scales, rotations, colors, opacities,
    opacities_threshold=0.005,
    reset=True
):
    """
    Low-opacity Gaussian 제거 및 선택적 opacity reset
    
    Args:
        ...: Gaussian parameters
        opacities_threshold: opacity 이 이 값 이하면 제거
        reset: True 면 opacity 를 0.5 로 초기화
    
    Return:
        pruned and reset Gaussians
    """
    # Mask: keep if opacity > threshold
    keep_mask = opacities > opacities_threshold
    
    new_pos = positions_3d[keep_mask]
    new_scale = scales[keep_mask]
    new_rot = rotations[keep_mask]
    new_color = colors[keep_mask]
    new_opacity = opacities[keep_mask]
    
    # Opacity reset
    if reset:
        new_opacity = torch.full_like(new_opacity, 0.5)
    
    return new_pos, new_scale, new_rot, new_color, new_opacity

# Test
opacities_test = torch.tensor([0.9, 0.003, 0.5, 0.001, 0.7])
pos_test = torch.randn(5, 3)
scale_test = torch.randn(5, 3)
rot_test = torch.randn(5, 4)
color_test = torch.randn(5, 3)

print(f"Before prune: {len(opacities_test)} Gaussians, opacities: {opacities_test.tolist()}")

new_pos, new_scale, new_rot, new_color, new_opacity = prune_and_reset_opacity(
    pos_test, scale_test, rot_test, color_test, opacities_test,
    opacities_threshold=0.005, reset=True
)

print(f"After prune & reset: {len(new_opacity)} Gaussians")
print(f"New opacities: {new_opacity.tolist()}")
```

---

## 🔗 실전 활용

### 1. Iteration 스케줄

```python
# 대략적 3DGS 학습 루프
for iteration in range(num_iterations):
    # Render
    rendered_image = render(gaussians, camera)
    
    # Loss
    loss = compute_loss(rendered_image, gt_image)
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    # Adaptive densification (e.g., every 100 iterations)
    if iteration % 100 == 0 and iteration > 500:
        actions = evaluate_densification_candidates(...)
        execute_densification(...)
    
    # Pruning & opacity reset (e.g., every 3000 iterations)
    if iteration % 3000 == 0 and iteration > 1000:
        prune_and_reset_opacity(...)
```

### 2. Hyperparameter 권장값

- **Clone threshold**: s < 0.1 (작은 Gaussian)
- **Split threshold**: s > 0.5 (큰 Gaussian)
- **Gradient threshold**: τ ~ 0.0002 (scene 에 따라 조정)
- **Opacity threshold**: ε ~ 0.005
- **Reset interval**: 3000 iteration

### 3. 초기화 전략

- COLMAP points 로 시작 (sparse)
- 초기 α = 0.1 (매우 투명, 점차 학습)
- Clone/Split 으로 점 수 증가: 초기 ~10k → 최종 ~100k+

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|------------|
| Gradient 신호가 유효 | 초기 몇 iteration 에서 gradient 노이즈. 충분한 warmup 후 densification 시작 |
| Clone/Split threshold 고정 | Scene 마다 최적값 다름. Adaptive threshold 도 가능 |
| Pruning threshold 고정 | 매우 복잡한 scene 에서 α<0.005 인 유용한 점도 있을 수 있음 |
| Opacity reset 모든 점에 적용 | 수렴하는 점은 α=1 근처 → reset 하면 다시 진동. Selective reset 도 가능 |
| Split 위치 무작위 | Deterministic offset (e.g., PCA 주축) 을 사용하는 것이 더 효율적일 수 있음 |

---

## 📌 핵심 정리

| 조건 | 작용 | 효과 |
|------|------|------|
| $\|\nabla \mathcal{L}\| > \tau$ AND $\|\mathbf{s}\| < s_c$ | **Clone** | +1 Gaussian (같은 위치, 같은 크기) |
| $\|\nabla \mathcal{L}\| > \tau$ AND $\|\mathbf{s}\| > s_s$ | **Split** | -1 + 2 Gaussians (축소 & 분산) |
| $\alpha < \epsilon$ | **Prune** | -1 Gaussian (제거) |
| Every N iter | **Reset α** | α ← 0.5 (모든 점) |

**핵심 성질**:
1. ✓ Gradient 기반 자동 adaptation
2. ✓ Clone: 세밀한 디테일, Split: 큰 영역 해석
3. ✓ Pruning: 메모리 & 속도 최적화
4. ✓ Opacity reset: 학습 안정성 (모든 점이 기여 기회)

---

## 🤔 생각해볼 문제

**문제 1** (기초): Clone 과 Split 의 차이를 "정보 이론" 으로 설명하라. 어느 것이 더 많은 파라미터 추가인가?

<details>
<summary>해설</summary>

**Clone**: Gaussian 개수 +1. Parameterization 공간 +8 (position 3, quaternion 4, scale 3, color 48, opacity 1 → 총 64).

**Split**: Gaussian 개수 +1 (총 +2 - 1). Parameterization 공간 +64.

**Information**: Split 이 원본 Gaussian 을 "분해"하므로, information-theoretically 더 정교한 fitting → 이론상 Split 이 더 효율적.

하지만 Clone 도 위치에 약간의 노이즈를 추가하면 다른 detail 을 잡을 수 있음.

</details>

**문제 2** (심화): Opacity reset 없이 학습하면 무엇이 문제인가? 극단적 예시를 들어라.

<details>
<summary>해설</summary>

예: α₁=1.0 (완전 불투명), α₂=0.1, α₃=0.1 로 수렴.

Rendering: C ≈ α₁c₁ (α₂, α₃ 는 완전히 가려짐)

만약 충분한 training view 가 있어서 α₁ 만으로는 모든 color 를 표현 불가능하면, loss 가 stuck.

Reset 후: α₁=0.5, α₂=0.5, α₃=0.5 → 각각 재경쟁, gradient 흐름 복구.

</details>

**问题 3** (論文批評): Densification 조건이 "gradient magnitude" 인데, 다른 기준 (e.g., Hessian, 2nd-order curvature) 을 사용할 수는 없는가?

<details>
<summary>해설</summary>

**1차 gradient**: 계산 빠름, 직관적 (변화율)

**Hessian**: 곡률 정보, "얼마나 가파른가" → 더 정교한 densification 가능. 하지만:
- 계산 비용 높음 (O(N²) vs O(N))
- 메모리 많음 (대칭 행렬)
- 실무에선 1차 gradient 로도 충분

일부 연구: "Hessian-guided" 를 시도했으나, 속도/효과 대비 복잡도 증가 → 3DGS 원논문은 1차 유지.

</details>

---

<div align="center">

[◀ 이전](./04-tile-rasterization.md) | [📚 README](../README.md) | [다음 ▶](./06-3dgs-training-comparison.md)

</div>
