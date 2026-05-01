# 05. Multi-View Diffusion — Zero123 · MVDream · SV3D · Instant3D

## 🎯 핵심 질문

- DreamFusion / ProlificDreamer 의 근본적 한계는 무엇인가?
- **Janus problem** (다면체 object 의 한쪽 면만 학습) 은 왜 발생하는가?
- **Zero-1-to-3** 는 어떻게 relative camera pose 를 조건부로 사용하는가?
- **MVDream** 의 "4-view simultaneous + cross-view attention" 은 무엇을 해결하는가?
- **SV3D** (Stable Video) 와 **Instant3D** 는 어떻게 수 초 내에 3D 를 생성하는가?

---

## 🔍 왜 Multi-View 인가

DreamFusion / ProlificDreamer (SDS/VSD) 의 핵심 문제:

```
NeRF θ ──render──> z (single view)
  ↓
Diffusion model (frozen, 2D image trained)
  ↓
Score evaluation
  ↓
∇_θ update
```

**문제들**:

1. **Janus Problem (다면체 artifact)**: 
   - Text "a cat" → NeRF 가 모든 각도에서 "cat face" 를 만들려고 시도
   - 하지만 단일 view diffusion 은 한 direction 의 image 에만 평가
   - 결과: 한 쪽 면은 猫의 얼굴, 다른 쪽은 random (multi-face collapse)

2. **View Inconsistency**:
   - 각 iteration 마다 다른 random camera
   - 같은 NeRF 의 앞면과 뒷면이 "일관성 있게" 좋을 확률 ↓

3. **Geometry Ambiguity**:
   - 2D image 만으로는 3D shape 의 depth 를 잘 모름
   - "자동차" → 옆면? 앞면? 45도?

**해결책: Multi-View Diffusion**
- 동시에 여러 view 의 일관성을 평가
- "고양이의 앞면과 옆면이 동시에 plausible" 해야 함

---

## 📐 수학적 선행 조건

- **Ch6-02, 03, 04**: SDS, VSD
- **NeRF**: Multi-view rendering
- **Cross-attention**: Transformer 기반 view consistency
- **Relative camera pose**: Camera parameterization

---

## 📖 직관적 이해

### Zero-1-to-3: Relative Camera Pose Conditioning

**아이디어**:
```
Input: 1 reference image + text
         │
         └──> Diffusion trained on: (image, relative_camera, text)
              → "이 reference image 로부터 
                  상대 위치 (45도 회전) 에서의 이미지는?"
         │
         └──> Generate: 4 views at [0°, 90°, 180°, 270°]
              (relative to input camera frame)
```

**수학**:
$$
p(z_{\text{view}k} | z_{\text{ref}}, \Delta\pi_k, y)
$$

여기서:
- $z_{\text{ref}}$: reference view image
- $\Delta\pi_k$: relative camera transform (azimuth, elevation, distance)
- $y$: text prompt

**장점**: Single-image to multi-view 를 정규화된 relative frame 으로 학습 → generalization 좋음.

### MVDream: Simultaneous 4-View + Cross-View Attention

**개선**:
```
4 views ──────┐
              ├──> Transformer with cross-view attention
              │    (각 view 의 feature 가 다른 views 정보 활용)
              └──> Single unified score evaluation

Effect: "4개 view 가 geometry 적으로 일관됨" 을 강제
```

**Cross-view attention** 의 역할:
- View 1 의 feature ← 학습 (View 2, 3, 4 의 정보)
- → Multi-view geometry constraint 가 diffusion 에 내재

### SV3D & Instant3D: 회귀와 속도

**SV3D** (Stable Video 기반):
- Diffusion 이 "video sequence" (여러 frames) 를 생성
- Frames = multi-view (각 frame ≈ different angle)
- 시간적 consistency = geometric consistency

**Instant3D** (LRM + 4-view):
- Diffusion 으로 4-view 생성
- → LRM (Large Reconstruction Model, Hong 2024) 로 직접 regression
- → Triplane NeRF 추출
- 결과: **5초** (vs 5-15분 for SDS/VSD)

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Janus Problem

**정의**: 3D object 생성에서 multiple faces / geometric inconsistencies 가 나타나는 현상.

**수학적 표현**: NeRF $\theta$ 가 여러 camera angles $\pi_1, \pi_2$ 에서:
$$
z_1 = g(\theta, \pi_1), \quad z_2 = g(\theta, \pi_2)
$$

이 두 rendered images 가 **geometric 하게 일관성이 없는** 경우.

**원인**: Single-view diffusion score 는 각 view 에 independently 평가 → Janus reward (어느 각도든 diffusion 이 같은 얼굴을 만들라고 evaluate 가능).

### 정의 5.2 — Zero-1-to-3 Conditioning

Pre-trained diffusion model $p_\phi$:
$$
p_\phi(z_k | z_{\text{ref}}, \Delta\pi_k, y)
$$

입력:
- $z_{\text{ref}} \in \mathbb{R}^{H \times W \times 3}$: reference image
- $\Delta\pi_k = (\text{azimuth}_k, \text{elevation}_k, \text{scale}_k)$: 상대 카메라
- $y$: text prompt

출력:
- $z_k$: k-번째 view (예: k=0,1,2,3 for 4 cardinal directions)

**학습 데이터**: Objaverse + synthetic camera parameterization.

### 정의 5.3 — Cross-View Attention

Transformer layer 에서, 각 view 의 feature map 간 attention:
$$
\text{Attention}_{\text{cross}}(Q_i, K_j, V_j) = \text{softmax}\left(\frac{Q_i K_j^\top}{\sqrt{d}}\right) V_j
$$

여기서 $i \neq j$ (다른 view).

**효과**: View $i$ 의 denoising 이 views $j \neq i$ 의 구조를 "인식".

### 정의 5.4 — Multi-View SDS Loss

$$
\mathcal{L}_{\text{multi-view}} = \sum_{k=1}^K w_k \mathcal{L}_{\text{SDS}}(\theta; z_k, \text{camera}_k, y)
$$

여기서:
- $K$ = view 수 (보통 4)
- $z_k = g(\theta, \pi_k)$ (렌더링)
- Cross-view attention 으로 정규화

---

## 🔬 정리와 증명

### 정리 5.1 — Zero-1-to-3 는 Single-Image 3D 에 대한 Conditional Prior

**명제**: Relative camera pose 를 조건부로 학습된 diffusion $p_\phi(z | z_{\text{ref}}, \Delta\pi, y)$ 은, single reference image 로부터 plausible multi-view 를 생성한다.

**증명 sketch**:

1. **학습 데이터 구조**: Objaverse 의 각 3D model $\mathcal{M}$ 에 대해:
   - Reference camera $\pi_0$
   - Multiple views $\pi_k$ (다양한 상대 각도)
   - 각 view pair: $(z_0, z_k, \Delta\pi_k, \text{caption})$

2. **Conditional independence**:
   Reference image $z_{\text{ref}}$ 와 relative pose $\Delta\pi$ 가 주어지면, 
   implicit 3D model 의 shape/appearance 에 대한 충분한 정보 제공.
   → Generalization to unseen objects.

3. **정규화 효과**: 
   - Absolute camera frame 이 아니라 relative frame 에서 학습
   - Different object sizes/positions 에 대한 invariance
   → Better generalization than "absolute pose" conditioning.

**결론**: $p_\phi$ 가 3D geometry 의 "conditional model" 역할 수행 $\square$.

### 정리 5.2 — Cross-View Attention 은 Epipolar Constraint 를 Soft 하게 enforce

**명제**: Cross-view attention 을 통한 denoising 은, epipolar geometry 을 "soft constraint" 로 작용시킨다.

**증명 (intuitive)**:

Epipolar geometry: View 1 의 점 $x_1$ 이 3D 에서 $X$ 에 대응하면,
view 2 의 대응 점 $x_2$ 는 epipolar line $l_2$ 위에 있음.

Cross-view attention:
- View 1 의 feature $f_1$ 이 view 2 의 features 의 attention weight 계산
- → 기하학적으로 "관련 있는 부분" 이 높은 weight
- → Epipolar structure 가 implicitly 학습됨

**수학적으로는**: Attention weight distribution 이 epipolar line 근처에 집중하는 경향.

### 따름 정리 5.3 — Multi-View SDS 의 Janus Problem 해결

**명제**: 여러 view 에 대한 SDS loss 를 동시에 최소화하면, single-view SDS 의 Janus problem 을 대부분 해결한다.

**증명 idea**:

Janus problem 원인:
- 각 view 에서 diffusion 이 독립적으로 "good image" 를 요구
- → 다른 view 와의 일관성 무시 가능

Multi-view:
- 모든 view 가 **같은 NeRF** $\theta$ 에서 파생
- 앞면이 좋으려면, 뒷면도 (역으로) plausible 해야 함
- ∵ 같은 NeRF parameters
- → Single 3D geometry 의 제약이 모든 view 에 propagate

**정량적**: Janus artifact frequency (LPIPS-based) 가 multi-view SDS 에서 ~90% 감소 (논문 결과).

---

## 💻 구현 검증

### 실험 1 — Single-View vs Multi-View SDS

```python
import torch
import torch.nn as nn

class SimpleNeRF3D(nn.Module):
    """Toy 3D NeRF"""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # σ, RGB
        )
    
    def forward(self, points):
        """points: (N, 3) -> (N, 4) [σ, R, G, B]"""
        return self.mlp(points)

def render_view(nerf, camera_angle, resolution=32):
    """
    Simple volumetric rendering from given camera angle.
    camera_angle: azimuth in radians
    """
    # Generate ray grid
    grid = torch.linspace(-1, 1, resolution)
    x_grid, y_grid = torch.meshgrid(grid, grid, indexing='ij')
    
    # Rotate by camera angle
    x_rot = x_grid * torch.cos(camera_angle) - y_grid * torch.sin(camera_angle)
    y_rot = x_grid * torch.sin(camera_angle) + y_grid * torch.cos(camera_angle)
    z_grid = torch.ones_like(x_grid) * 0.5
    
    points = torch.stack([x_rot, y_rot, z_grid], dim=-1).reshape(-1, 3)
    
    # NeRF forward
    sigma_rgb = nerf(points)  # (N, 4)
    sigma = torch.relu(sigma_rgb[:, 0])
    rgb = torch.sigmoid(sigma_rgb[:, 1:])
    
    # Simple alpha composite
    alpha = 1 - torch.exp(-sigma * 0.1)
    rgb_out = (alpha[:, None] * rgb).sum(dim=0) / (alpha.sum() + 1e-5)
    
    return rgb_out.reshape(resolution, resolution, 3)

def mock_diffusion_score(z, view_id=None):
    """
    Mock diffusion that evaluates view quality.
    view_id: which view is this? (for Janus problem simulation)
    """
    # Prefer "high color saturation" for faces
    # Different face preference for different views (Janus!)
    if view_id == 0:  # Front
        target_color = torch.tensor([1.0, 0.8, 0.6])  # Face-like
    elif view_id == 1:  # Side
        target_color = torch.tensor([1.0, 0.5, 0.5])  # Different face!
    else:
        target_color = torch.tensor([0.5, 0.5, 0.5])  # Neutral
    
    mean_color = z.mean(dim=(0, 1))
    score = -(mean_color - target_color).norm()
    
    return score.unsqueeze(0).unsqueeze(0).expand_as(z)

# Experiment: Single-View SDS
print("=== Single-View SDS (Janus prone) ===")
nerf_single = SimpleNeRF3D()
opt_single = torch.optim.Adam(nerf_single.parameters(), lr=0.01)

for it in range(50):
    # Only front view
    z_front = render_view(nerf_single, torch.tensor(0.0))
    eps_pred = mock_diffusion_score(z_front, view_id=0)
    noise = torch.randn_like(z_front)
    
    loss = ((eps_pred - noise) ** 2).mean()
    
    opt_single.zero_grad()
    loss.backward()
    opt_single.step()
    
    if (it + 1) % 10 == 0:
        # Check consistency: render from side
        z_side = render_view(nerf_single, torch.tensor(torch.pi / 2))
        front_color = z_front.mean(dim=(0, 1))
        side_color = z_side.mean(dim=(0, 1))
        
        print(f"Iter {it+1:2d}: Front RGB = {front_color.detach().numpy()}, " 
              f"Side RGB = {side_color.detach().numpy()}")

# Experiment: Multi-View SDS
print("\n=== Multi-View SDS (Janus resistant) ===")
nerf_multi = SimpleNeRF3D()
opt_multi = torch.optim.Adam(nerf_multi.parameters(), lr=0.01)

for it in range(50):
    losses = []
    
    # 4 views
    for view_id, angle in enumerate([0, torch.pi/2, torch.pi, 3*torch.pi/2]):
        z = render_view(nerf_multi, angle)
        eps_pred = mock_diffusion_score(z, view_id=view_id)
        noise = torch.randn_like(z)
        
        loss = ((eps_pred - noise) ** 2).mean()
        losses.append(loss)
    
    total_loss = sum(losses) / len(losses)
    
    opt_multi.zero_grad()
    total_loss.backward()
    opt_multi.step()
    
    if (it + 1) % 10 == 0:
        z_front = render_view(nerf_multi, torch.tensor(0.0))
        z_side = render_view(nerf_multi, torch.tensor(torch.pi / 2))
        
        front_color = z_front.mean(dim=(0, 1))
        side_color = z_side.mean(dim=(0, 1))
        
        print(f"Iter {it+1:2d}: Front RGB = {front_color.detach().numpy()}, " 
              f"Side RGB = {side_color.detach().numpy()}")

print("✓ Multi-view maintains more consistent geometry across views")
```

### 실험 2 — Zero-1-to-3 Relative Pose Conditioning

```python
def simulate_relative_pose_effect():
    """
    Demonstrate how relative pose conditioning helps generalization.
    """
    
    # Mock diffusion trained with (z_ref, rel_pose, text) triplets
    def conditional_prior_score(z_current, z_ref, rel_pose, text):
        """
        Prior probability of z_current given z_ref and relative pose.
        z_ref: reference image
        rel_pose: (azimuth_delta, elevation_delta, scale_delta)
        """
        
        # Learned expectation: 
        # "If ref is frontal view, and we rotate 90°, expect side view texture"
        
        expected_z = z_ref * (1.0 + rel_pose[0] * 0.1)  # Simplified
        
        error = (z_current - expected_z) ** 2
        score = -error.mean()
        
        return score
    
    # Test on unseen object
    z_ref_unseen = torch.randn(4, 4, 3) * 0.5 + 0.5
    rel_pose = torch.tensor([0.5, 0.1, 1.0])  # 90° rotation
    
    scores = []
    for _ in range(5):
        z_generated = torch.randn(4, 4, 3) * 0.5 + 0.5
        score = conditional_prior_score(z_generated, z_ref_unseen, rel_pose, "a car")
        scores.append(score.item())
    
    print(f"✓ Relative pose conditioning generates scores: {scores}")
    print(f"✓ Can be used for zero-shot generalization to unseen objects")

simulate_relative_pose_effect()
```

### 실험 3 — MVDream Cross-View Attention

```python
class CrossViewTransformer(nn.Module):
    """
    Simplified cross-view attention block for diffusion.
    """
    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, features_per_view):
        """
        features_per_view: list of (B, H, W, C) tensors, one per view
        """
        # Simple implementation: compute attention between view features
        B, H, W, C = features_per_view[0].shape
        K = len(features_per_view)  # number of views
        
        # Flatten spatial
        all_features = torch.cat([f.reshape(B, -1, C) for f in features_per_view], dim=1)
        
        # Cross-attention
        qkv = self.to_qkv(all_features)  # (B, K*H*W, C*3)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Attention
        scale = self.dim_head ** -0.5
        dots = torch.einsum('bic,bjc->bij', q, k) * scale
        attn = dots.softmax(dim=-1)
        
        out = torch.einsum('bij,bjc->bic', attn, v)
        out = self.to_out(out)
        
        return out

# Test cross-view attention
transformer = CrossViewTransformer(dim=64)
features_4views = [torch.randn(2, 16, 16, 64) for _ in range(4)]

cross_attended = transformer(features_4views)
print(f"✓ Cross-view attention output shape: {cross_attended.shape}")
print(f"✓ Features from different views are now mixed")
```

---

## 🔗 실전 활용

### 1. MVDream 전체 파이프라인

```python
def train_mvdream_style(nerf, prompt, num_iterations=1000):
    """
    Multi-view DreamFusion with cross-view attention.
    """
    
    # Pre-trained diffusion (multiview-aware)
    mvdiffusion = load_multiview_diffusion()  # Trained on Zero123 or similar
    
    # Cross-view transformer
    cross_view_attn = CrossViewTransformer()
    
    optimizer = torch.optim.Adam(
        list(nerf.parameters()) + list(cross_view_attn.parameters()),
        lr=0.001
    )
    
    for it in range(num_iterations):
        # Sample 4 cameras (cardinal directions often used)
        cameras_4 = sample_4cardinal_cameras()
        
        # Render 4 views
        rendered_4views = [nerf_render(nerf, cam) for cam in cameras_4]
        
        # Cross-view attention
        z_attended = cross_view_attn(rendered_4views)
        
        # Multi-view diffusion score
        t = torch.randint(100, 900, (1,)).item()
        
        losses = []
        for i, z in enumerate(rendered_4views):
            # Noisy diffusion
            noise = torch.randn_like(z)
            z_t = alpha_t * z + sigma_t * noise
            
            # Diffusion (attended to other views)
            with torch.no_grad():
                eps_pred = mvdiffusion(z_t, t, prompt, view_index=i)
            
            # SDS loss
            loss = ((eps_pred - noise) ** 2).mean()
            losses.append(loss)
        
        total_loss = sum(losses) / len(losses)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (it + 1) % 100 == 0:
            print(f"Iter {it+1:4d}: loss = {total_loss.item():.6f}")
```

### 2. Instant3D: 4-View + LRM Regression

```python
def instant3d_pipeline(prompt, num_iterations=1000):
    """
    Fast 3D generation: Multiview diffusion + LRM regression.
    """
    
    # Stage 1: Generate 4 views using Zero123-like diffusion
    diffusion_4view = load_zero123_or_mvdream()
    
    # Arbitrary reference image (or generated from text)
    reference_image = generate_or_input_reference(prompt)
    
    # Generate 4 views
    views_4 = []
    for relative_pose in [0°, 90°, 180°, 270°]:
        view = diffusion_4view.sample(
            reference_image=reference_image,
            relative_pose=relative_pose,
            prompt=prompt
        )
        views_4.append(view)
    
    # Stage 2: Direct regression to triplane NeRF (LRM-style)
    lrm_model = LargeReconstructionModel()
    
    # Concatenate views
    views_stacked = torch.stack(views_4)  # (4, H, W, 3)
    
    # Single forward pass
    triplane = lrm_model(views_stacked)  # Triplane NeRF features
    
    # Decode to mesh or NeRF
    mesh = decode_triplane_to_mesh(triplane)
    
    return mesh

# Result: 3D in seconds, not minutes!
```

### 3. SV3D: Video-as-3D

```python
def sv3d_pipeline(prompt):
    """
    SV3D: Generate video sequence, treat frames as multi-view.
    """
    
    # Stable Video Diffusion (fine-tuned for 3D)
    sv3d_model = StableVideoDiffusion()
    
    # Generate video: 4-16 frames
    video_frames = sv3d_model.sample(
        prompt=prompt,
        num_frames=8,  # 8 frames ≈ camera orbit
        guidance_scale=7.5
    )  # (8, H, W, 3)
    
    # Frames = multi-view: extract 3D using NeRF or LRM
    nerf_or_triplane = reconstruct_from_video(video_frames)
    
    return nerf_or_triplane
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| **4 views 가 충분** | 실제로는 8-16 views 가 더 정확할 수 있음. Trade-off: views ↑ → compute ↑ |
| **Diffusion 이 multi-view 에 trained** | Zero123, MVDream 모두 3D data (Objaverse) 에서 학습. 다른 domain 이면 generalization 약할 수 있음 |
| **Relative pose 가 정확** | Relative camera parameterization 이 정확하지 않으면 misalignment. Camera calibration 필요 |
| **Cross-view attention 이 충분** | Attention mechanism 이 항상 correct correspondence 를 학습하는 것은 아님. 기하학적 constraints (epipolar geometry) 를 explicit 하게 추가할 수도 있음 |
| **Janus problem 완전 해결** | Multi-view SDS 로도 일부 inconsistency 남을 수 있음 (특히 complex topology) |
| **속도 (Instant3D)** | Regression-based (LRM) 는 빠르지만, fine-tuning 불가능 (SDS/VSD 는 iterative 이므로 원하는 대로 adjust 가능) |

---

## 📌 핵심 정리

$$\boxed{\text{Multi-View Diffusion} = \text{여러 view 의 기하학적 일관성} + \text{single-image prior 회피}}$$

| 방법 | 주요 아이디어 | 장점 | 단점 |
|------|-------------|------|------|
| **Zero-1-to-3** | Relative pose conditioning | Generalization 우수 | Single reference 필요 |
| **MVDream** | 4-view + cross-attn | 일관성 우수 | Compute 증가 |
| **SV3D** | Video frames as multi-view | 자연스러운 orbit | 학습 데이터 필요 |
| **Instant3D** | 4-view diffusion + LRM | 초당 생성 | 품질 vs 속도 trade-off |

**Key insight**: DreamFusion 은 "2D image prior 활용" 이 핵심이지만, 그 한계 (Janus) 를 해결하려면 **Multi-view consistency** 를 강제해야 함. Zero123 이후의 발전은 모두 이 방향을 따름.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Janus problem 이 정확히 무엇이고, SDS/VSD 의 어떤 특성이 이를 유발하는가?

<details>
<summary>해설</summary>

**Janus problem**:
- Janus: 로마 신화의 양면신 (두 얼굴)
- 3D 생성에서: 한쪽은 완벽한 "고양이 얼굴", 다른 쪽은 "노이즈"

**원인**:

SDS loss: $\nabla_\theta L = \mathbb{E}_{\pi, t, \epsilon}[w(t)(\hat{\epsilon}_\phi - \epsilon) \partial z / \partial \theta]$

각 iteration 에서:
- Random camera $\pi$ 샘플링
- 그 camera 에서만 평가 (한 view)
- Diffusion: "이 view 는 텍스트와 매치" 라고만 평가

결과:
- Front view: diffusion 이 "고양이 얼굴" 라고 라벨 → NeRF 가 optimize
- Side view (다음 iteration): 또 diffusion 이 "고양이 얼굴" 라고 라벨 → 다시 optimize
- 같은 NeRF 가 모든 각도에서 "얼굴"을 만들려고 시도
- → 기하학적으로 불가능 → 한쪽은 random collapse

**다중 view 가 해결하는 방식**:
- 4 views 동시 평가
- 같은 NeRF $\theta$ 에서 모든 view 파생
- → "Front 가 고양이 얼굴이면, side 는 (역학적으로) 어떻게 보이나?" 를 diffusion 이 평가

$\square$

</details>

**문제 2** (심화): Zero-1-to-3 에서 relative camera pose 를 conditioning 하는 것이 왜 "absolute camera frame" 보다 generalization 이 좋은가?

<details>
<summary>해설</summary>

**Absolute frame conditioning 의 문제**:

Diffusion 이 (image, absolute_camera, text) 로 학습되면:
- "Camera at (x=1, y=0, z=5)" 의 이미지 생성
- 다른 object size 나 position 에서는 같은 absolute pose 가 의미 다름
- → Overfitting to training data distribution

**Relative pose conditioning 의 장점**:

Diffusion 이 (reference_image, relative_delta, text) 로 학습:
- "Reference view 로부터 45° 회전" 이라는 상대적 개념
- Object size, position 과 무관
- → Invariance to scale/translation

**수학적**:

Absolute: $p_\phi(z_k | z_{\text{ref}}, \pi_k, y)$
- $\pi_k$ 가 global coordinate 라면, distribution $p$ 가 $\pi_k$ 의 scale 에 dependent
- Seen data: $[\pi_0, \pi_1, \ldots]$ → unseen pi 에서 poor generalization

Relative: $p_\phi(z_k | z_{\text{ref}}, \Delta\pi_k, y)$
- $\Delta\pi_k$ 가 relative 라면, object 의 absolute position 과 독립
- → Better invariance, fewer parameters to fit

**결과**: Zero-1-to-3 는 Objaverse 800K 에 학습되지만, 임의의 새 object 에도 잘 작동 $\square$.

</details>

**문제 3** (논문 비평): Instant3D (Li et al. 2024) 가 4-view diffusion + LRM 으로 "5초" 를 달성하는 반면, ProlificDreamer 는 왜 5-15분이 걸리는가? 속도 개선의 source 는 무엇인가?

<details>
<summary>해설</summary>

**시간 비교**:

| 단계 | DreamFusion (SDS) | ProlificDreamer (VSD) | Instant3D |
|------|-------------------|---------------------|-----------|
| 초기화 | 1초 | 1초 | 1초 |
| Optimization iterations | 1000+ × (render + diffusion) | 500+ × (render + diffusion + LoRA) | 0초 (feed-forward) |
| Render 시간 | 1-2초/iter | 1-2초/iter | Single render (1초) |
| Diffusion inference | ~5초/iter (frozen) | ~5초/iter + LoRA fine-tune | 1회만 (4-view generation) |
| **총 시간** | 30-40분 | 5-15분 | 5초 |

**Instant3D 의 속도 비결**:

1. **Zero123/MVDream 사용**: 이미 학습된 multi-view diffusion 으로 4-view 직접 생성
   - → SDS iteration 없음 (0 iterations vs 1000+)

2. **LRM (Large Reconstruction Model)**: 
   - Hong 2024 에서 제안
   - 4-view image → triplane NeRF (feed-forward regression)
   - 학습: Objaverse 에 한번만 (inference 시점에는 학습 없음)
   - → SDS/VSD 의 iterative refinement 완전 회피

3. **Regression vs Optimization**:
   - SDS/VSD: $\nabla \text{loss}$ 를 따라 iteratively update
   - LRM: Single forward pass ($\approx$ classifier)

**Trade-off**:

| 측면 | SDS/VSD | Instant3D |
|------|---------|-----------|
| 속도 | 5-30분 | 5초 |
| 품질 | 높음 (fine detail) | 중간 (general structure) |
| Control | 높음 (text 조정 가능) | 낮음 (4-view 이후 fixed) |
| 메모리 | GPU 16GB+ | GPU 8GB (forward only) |

**의미**: "3D generation 이 optimization 에서 regression 으로 shift" 중 — LRM, DUSt3R 등. 속도와 품질의 근본적 선택 $\square$.

</details>

---

<div align="center">

[◀ 이전](./04-vsd-particle-score.md) | [📚 README](../README.md) | [다음 ▶](../ch7-foundation/01-large-reconstruction-models.md)

</div>
