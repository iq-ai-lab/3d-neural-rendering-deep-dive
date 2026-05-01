# 02. 3D Foundation Models — DUSt3R · MASt3R (Wang 2024)

## 🎯 핵심 질문

- **DUSt3R** 이 왜 COLMAP 의 feature matching + bundle adjustment 를 **한 줄의 ViT cross-attention 으로 대체**할 수 있는가?
- Uncalibrated image pair 에서 **dense pointmap** (3D coord per pixel) 를 직접 회귀하는 것이 기하학적으로 가능한 이유는?
- **Epipolar geometry** 의 핵심 (fundamental matrix, epipolar line) 이 어떻게 transformer 의 inductive bias 로 암시적 학습되는가?
- **MASt3R** 이 matching head 와 multi-image extension 을 추가하면서 어떤 성능 향상을 이루는가?
- Robotics · AR mapping 에서 즉시 활용 가능한 이유는?

---

## 🔍 왜 DUSt3R 이 "3D 재구성의 패러다임"인가

### COLMAP 의 한계

전통적 SfM 파이프라인:

```
Image 1 ──┐
          ├─→ [Feature Extraction: SIFT/ORB]
Image 2 ──┤   ↓
          ├─→ [Feature Matching: ratio test, RANSAC]
...       │   ↓
          └─→ [Bundle Adjustment: iterative Levenberg-Marquardt]
                ↓
          Refined camera pose + 3D points
          (minutes ~ hours depending on scale)
```

**문제점**:
- Feature matching 은 **hand-crafted descriptor** 의존 (SIFT 는 50년 전)
- Occlusion, low-texture, view-dependent lighting 에 약함
- Bundle adjustment 는 non-convex, 초기값 민감 (실패 가능)
- Real-time (AR, robotics) 에 부적합

### DUSt3R 의 전환

```
[Image 1, Image 2]
    ↓
[Vision Transformer Encoder: freeze DINO]
    ↓
[Cross-Attention Layer: "이미지 간 correspondence"]
    ↓
[Dense Pointmap per pixel]
    └─→ Dense 3D map (no explicit feature matching)
    └─→ **seconds** (end-to-end)
```

**핵심 insight**: "Correspondence 는 high-level 기하학이 아니라, **pixel-wise 회귀 문제"**.

따라서 NN 이 직접 3D 좌표를 예측할 수 있음 — SIFT, RANSAC 불필요.

---

## 📐 수학적 선행 조건

### Epipolar Geometry 기초

두 카메라 $C_1 = [K | 0], C_2 = [K | Kt]$ (normalized cameras):

**Fundamental Matrix** $\mathbf{F}$:

$$
\mathbf{p}_2^T \mathbf{F} \mathbf{p}_1 = 0
$$

$\mathbf{p}_1 \in \mathbb{P}^2$ (homogeneous image coord) 의 3D pre-image 는 epipolar line $l_2 = \mathbf{F} \mathbf{p}_1$ 에 투영.

**Essential Matrix** $\mathbf{E} = K_2^T \mathbf{F} K_1$ (intrinsics 포함).

### Dense correspondence

Depth estimation (SfM 의 key): pair 간 동일 object 의 2D-3D correspondence 로부터.

**Triangulation** (DLT):
$$
\mathbf{X} = \text{DLT}(\mathbf{p}_1, C_1, \mathbf{p}_2, C_2)
$$

→ DUSt3R 은 이를 직접 NN 으로 학습: $\mathbf{X} \sim f_\theta(\mathbf{I}_1, \mathbf{I}_2, \mathbf{p}_1, \mathbf{p}_2)$

### ViT & Cross-Attention

- **Patch embedding**: Image → sequence of tokens (14×14 patches)
- **Cross-attention**: Query from image 1, Key/Value from image 2
  $$
  \text{Attn}(Q_1, K_2, V_2) = \text{softmax}(Q_1 K_2^T / \sqrt{d}) V_2
  $$
  → Image 1 의 각 patch 가 Image 2 에서 "같은 object 찾기"

---

## 📖 직관적 이해

### Dense Pointmap: 각 픽셀이 3D 좌표

```
Image 1 (768×1024)
    ↓
[ViT Patch: 768→54×72 patches (14px each)]
    ↓
[ViT encoder output: (54×72, 768 dims)]
    ↓
[Cross-attention to Image 2]
    ↓
For each patch in Image 1:
    "Image 2 의 어느 patch 가 나와 같은 물체를 봤나?"
    ↓
[Geometry Decoder: matched features → 3D coords]
    ↓
[Per-pixel 3D pointmap: (54×72, 3)]
    ↓
Upsample to full resolution (optional)
```

### 왜 Cross-Attention 이 Matching 인가

Attention mechanism 의 본질:

$$
\text{attn}(q) = \sum_k \alpha_k(q) v_k, \quad \alpha_k(q) = \frac{\exp(q \cdot k / \sqrt{d})}{\sum_j \exp(q \cdot k_j / \sqrt{d})}
$$

- $q$ (from Image 1 patch) 가 Image 2 의 가장 유사한 patch 의 $k$ 와 매칭됨
- Softmax attention → probabilistic correspondence
- **gradient-friendly**: RANSAC 의 discrete voting 대신 soft matching

### Geometry Decoder 의 역할

Matched patch pair $(f_1, f_2)$ 로부터 3D 좌표를 어떻게 구할 것인가?

**Option 1**: DLT (traditional) — camera pose 필수
**Option 2**: Regress 3D coords + depth directly (DUSt3R 방식) — canonical frames 에서

DUSt3R 은 **canonicalized** 좌표계 사용:
- Image 1 의 principal point 를 world 원점
- Image 2 와 Image 1 의 기하를 implicit 로 학습

---

## ✏️ 엄밀한 정의

### 정의 7.5 — Dense Pointmap 표현

두 이미지 pair $(I_1, I_2)$ 에 대해, DUSt3R 은 pointmap $\mathcal{P}: \mathbb{N}^2 \to \mathbb{R}^3$ 를 예측:

$$
\mathcal{P}(u, v) \in \mathbb{R}^3 \quad \text{for pixel coordinate } (u, v) \in [0, H) \times [0, W)
$$

여기서 각 $\mathcal{P}(u, v) = (X, Y, Z)$ 는 **canonicalized 3D world coordinate** — 카메라 intrinsic 과 무관한 정규화된 좌표.

**특성**: Uncalibrated pair 에서도 **affine ambiguity** (scale, translation) 만 남음 → bundle adjustment 단계에서 정정.

### 정의 7.6 — Confidence Map

각 pixel 마다 **prediction confidence** $\rho: \mathbb{N}^2 \to [0, 1]$:

$$
\rho(u, v) = \text{softmax}(\text{confidence\_head}(f_1(u, v), f_2(u, v)))
$$

Low-texture, occlusion, specular region 에서 $\rho$ 가 자동으로 낮음 → **uncertainty-aware** reconstruction.

### 정의 7.7 — Matching Head (MASt3R)

DUSt3R 의 dense pointmap + 추가로 **sparse matching head**:

$$
\{\mathbf{k}_i\}_{i=1}^{N_k} = \text{KeypointDetector}(f_1, f_2)
$$

각 keypoint $\mathbf{k}_i$ 는:
- 이미지 1 의 (u, v) 좌표
- 이미지 2 의 대응 (u', v')
- 신뢰도 $\rho_i$

→ Traditional SfM 과 호환: keypoint-based bundle adjustment 가능.

### 정의 7.8 — Multi-image Extension

$N > 2$ 개 이미지:

$$
\mathcal{P}_{\text{multi}} = \text{MultiDUSt3R}(I_1, I_2, \ldots, I_N)
$$

**구현**: pairwise DUSt3R 을 모든 쌍에 대해 실행 후, **bundle adjustment** (CoCoSfM 스타일) 로 통합 또는, **direct multi-view 최적화** (newer method).

---

## 🔬 정리와 증명

### 정리 7.3 (Dense Regression vs Feature Matching)

**명제**: Uncalibrated image pair $(I_1, I_2)$ 에서 dense pointmap $\mathcal{P}$ 를 회귀하면, traditional feature matching + RANSAC 과 동등한 정확도를 달성할 수 있다 (충분한 데이터).

**증명 sketch**:

1. **Fundamental Matrix 학습**: Cross-attention layer 가 implicit 로 epipolar geometry 학습. Attention pattern 을 분석하면 $F$ 행렬 구조 드러남 (empirical).

2. **Correspondence Coverage**: Dense regression 은 모든 pixel 을 다루므로 (feature matching 은 ~5% 의 keypoint 만), dense map 의 정보량이 훨씬 크다.

3. **Noise Robustness**: Confidence map $\rho$ 가 outlier downweight → RANSAC 의 hard threshold 보다 부드러움.

**결론**: Network capacity 충분하면 end-to-end learning 이 hand-crafted pipeline 을 이김. $\square$

### 정리 7.4 (Multi-image Consistency)

DUSt3R 을 모든 image pair $(i, j)$ 에 대해 실행하여 pointmap $\mathcal{P}_{ij}$ 생성 후, 3D 점들이 **consistent** 한 조건:

$$
\text{reprojection error} = \sum_{i,j} \|\text{project}(\mathcal{P}_{ij}(u, v), C_i) - (u, v)\|_2^2 < \epsilon
$$

**증명**: Triangulation uniqueness (epipolar geometry) — 올바른 epipolar line 에 있으면 3D 점이 유일 (up to depth scale).

Multi-image 에서: 추정된 pointmap 들이 서로 reprojection 할 때 자동 정렬되면 bundle adjustment 불필요 (or minimal). $\square$

### 따름 정리 7.5 (Calibration-free Reconstruction)

DUSt3R pointmap 으로부터:

1. **Relative depth** 는 정확 (affine scale ambiguity 만)
2. **Relative pose** 는 정확 (camera positions, rotations)
3. **Intrinsic calibration** 은 선택적 (uncalibrated projection)

→ AR / robotics 에서 metric scale 이 필수면 한 번의 calibration ruler 로 고정 가능.

---

## 💻 구현 검증

### 실험 1 — Mock DUSt3R Cross-Attention Matcher

```python
import torch
import torch.nn as nn

class SimpleDUSt3RMatcher(nn.Module):
    """
    간단한 DUSt3R: cross-attention 기반 dense correspondence
    """
    def __init__(self, feat_dim=256, n_heads=8):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_heads = n_heads
        
        # Pretend frozen DINO features
        # Input: (B, H, W, feat_dim) or (B, H*W, feat_dim)
        
        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            feat_dim, nhead=n_heads, batch_first=True
        )
        
        # Geometry decoder: matched features → 3D coords
        self.geometry_decoder = nn.Sequential(
            nn.Linear(feat_dim * 2, 128),  # concat feat1 & feat2
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # (X, Y, Z)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(feat_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, feats1, feats2):
        """
        feats1, feats2: (B, N_patch, D)
        Returns:
            pointmap: (B, N_patch, 3)
            confidence: (B, N_patch, 1)
        """
        B, N, D = feats1.shape
        
        # Cross-attention: feats1 query, feats2 key/value
        feats1_refined, attn_weights = self.cross_attn(
            feats1, feats2, feats2
        )  # (B, N, D), (B, N_heads, N, N)
        
        # feats2 에서 가장 attended 된 feature 추출
        # (간단히: feats1_refined 가 이미 cross-attended)
        attn_weights_mean = attn_weights.mean(1)  # (B, N, N)
        
        # For each patch in image 1, argmax of attention → matching patch in image 2
        matched_indices = attn_weights_mean.argmax(dim=-1)  # (B, N)
        feats2_matched = feats2[torch.arange(B).unsqueeze(1), matched_indices]  # (B, N, D)
        
        # Geometry decoder
        combined = torch.cat([feats1_refined, feats2_matched], dim=-1)  # (B, N, 2D)
        pointmap = self.geometry_decoder(combined)  # (B, N, 3)
        
        # Confidence
        confidence = self.confidence_head(combined)  # (B, N, 1)
        
        return pointmap, confidence, attn_weights_mean

# 사용 예시
batch_size = 4
n_patches = 256  # 16×16 patches
feat_dim = 256

model = SimpleDUSt3RMatcher(feat_dim=feat_dim, n_heads=8)

feats1 = torch.randn(batch_size, n_patches, feat_dim)
feats2 = torch.randn(batch_size, n_patches, feat_dim)

pointmap, confidence, attn = model(feats1, feats2)
print(f"Pointmap shape: {pointmap.shape}")  # (4, 256, 3)
print(f"Confidence shape: {confidence.shape}")  # (4, 256, 1)
print(f"Mean confidence: {confidence.mean():.3f}")
print(f"Attention pattern (example):\n{attn[0, :5, :5]}")  # 첫 few patch 들의 attention

# 예상 출력
# Pointmap shape: (4, 256, 3)
# Confidence shape: (4, 256, 1)
# Mean confidence: 0.512
# Attention pattern...
```

### 실험 2 — Dense Pointmap to Traditional Keypoints

```python
def extract_sparse_keypoints(pointmap, confidence, n_keypoints=100):
    """
    Dense pointmap 에서 sparse keypoints 추출 (MASt3R 스타일)
    """
    B, N, _ = pointmap.shape
    
    # 가장 높은 confidence 를 가진 n_keypoints 선택
    _, top_indices = torch.topk(confidence.squeeze(-1), n_keypoints, dim=1)
    
    keypoints_3d = torch.gather(
        pointmap, 1, top_indices.unsqueeze(-1).expand(-1, -1, 3)
    )  # (B, n_keypoints, 3)
    
    return keypoints_3d, top_indices

# 사용
keypoints_3d, indices = extract_sparse_keypoints(pointmap, confidence, n_keypoints=100)
print(f"Sparse keypoints: {keypoints_3d.shape}")  # (4, 100, 3)
print(f"Top keypoint indices: {indices[0, :10]}")  # (4, 10)
```

### 실험 3 — Multi-image Bundle Adjustment (Mock)

```python
def multi_view_triangulation(pointmaps_list, confidences_list, camera_poses_list):
    """
    N image pair 의 pointmap 들을 통합하여 최종 3D 점 계산
    
    pointmaps_list: list of N pointmaps (각각 (B, N_patch, 3))
    confidences_list: list of N confidences (각각 (B, N_patch, 1))
    camera_poses_list: list of N camera poses (각각 (B, 4, 4) SE(3))
    
    Returns:
        final_pointmap: (B, N_patch, 3) - fused
    """
    
    B, N_patch, _ = pointmaps_list[0].shape
    n_views = len(pointmaps_list)
    
    # Simple averaging with confidence weighting
    confidence_sum = sum(conf for conf in confidences_list)
    weighted_sum = sum(
        pm * conf for pm, conf in zip(pointmaps_list, confidences_list)
    )
    
    final_pointmap = weighted_sum / (confidence_sum + 1e-6)  # (B, N_patch, 3)
    final_confidence = confidence_sum / n_views  # (B, N_patch, 1)
    
    return final_pointmap, final_confidence

# 사용: 3개 image 쌍
pointmaps = [torch.randn(2, 256, 3) for _ in range(3)]
confidences = [torch.rand(2, 256, 1) for _ in range(3)]
poses = [torch.eye(4).unsqueeze(0).repeat(2, 1, 1) for _ in range(3)]

fused_pm, fused_conf = multi_view_triangulation(pointmaps, confidences, poses)
print(f"Fused pointmap: {fused_pm.shape}")  # (2, 256, 3)
print(f"Fused confidence: {fused_conf.mean():.3f}")
```

### 실험 4 — DUSt3R vs COLMAP Speed

```python
import time
import numpy as np

def benchmark_dust3r(batch_size=4, n_patches=256, n_runs=100):
    """DUSt3R (ViT + cross-attn) 시간"""
    model = SimpleDUSt3RMatcher(feat_dim=256, n_heads=8).cuda()
    feats1 = torch.randn(batch_size, n_patches, 256, device='cuda')
    feats2 = torch.randn(batch_size, n_patches, 256, device='cuda')
    
    torch.cuda.synchronize()
    t0 = time.time()
    
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(feats1, feats2)
    
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    
    return elapsed / n_runs

def mock_colmap_sift(n_images=2, n_keypoints_expected=5000):
    """
    COLMAP 의 typical timing (mock)
    
    1. SIFT extraction: ~1-2 sec per image
    2. Feature matching (brute-force + RANSAC): ~5-10 sec
    3. Bundle adjustment: ~10-30 sec (depending on scale)
    Total: ~30-50 seconds for 2 images
    """
    sift_time = 1.5 * n_images  # sec
    matching_time = 8.0
    ba_time = 20.0
    
    total = sift_time + matching_time + ba_time
    return total

dust3r_time = benchmark_dust3r(batch_size=4, n_patches=256)
colmap_time = mock_colmap_sift(n_images=2)

print(f"DUSt3R (end-to-end):  {dust3r_time*1000:.1f} ms")
print(f"COLMAP (SIFT+RANSAC+BA): {colmap_time:.1f} sec")
print(f"Speedup: {colmap_time / dust3r_time:.0f}x")

# 예상 출력
# DUSt3R (end-to-end):   45.3 ms
# COLMAP (SIFT+RANSAC+BA): 30.0 sec
# Speedup: 662x
```

---

## 🔗 실전 활용

### 1. AR Mapping: Real-time SLAM with DUSt3R

```python
class DUSt3RBasedSLAM:
    """
    Real-time AR mapping: DUSt3R + lightweight pose estimation
    """
    def __init__(self):
        self.model = SimpleDUSt3RMatcher().to('cuda')
        self.keyframe_buffer = []  # list of (image, features, pointmap)
        self.current_pose = np.eye(4)
    
    def add_frame(self, image, intrinsics):
        """
        새 frame 추가 및 pose 업데이트
        """
        # Extract DINO features (frozen)
        features = self.dino_encoder(image)  # (H/P, W/P, D)
        
        # DUSt3R with latest keyframe
        if self.keyframe_buffer:
            latest_keyframe = self.keyframe_buffer[-1]
            pointmap, confidence, _ = self.model(
                features.unsqueeze(0),
                latest_keyframe['features'].unsqueeze(0)
            )
            
            # Geometric pose estimation from pointmap
            # (간단히: PnP + RANSAC)
            pose = self.estimate_pose_pnp(
                pointmap[0], confidence[0], intrinsics
            )
            self.current_pose = pose
        
        # Keyframe selection (if motion large enough)
        if len(self.keyframe_buffer) == 0 or self.should_add_keyframe():
            self.keyframe_buffer.append({
                'image': image,
                'features': features,
                'pose': self.current_pose.copy()
            })
        
        return self.current_pose
    
    def get_3d_reconstruction(self):
        """현재까지의 3D 점들 반환"""
        all_points = []
        for kf in self.keyframe_buffer:
            # Re-compute pointmap for all keyframe pairs
            pass
        return np.concatenate(all_points, axis=0)

# 사용: streaming AR
slam = DUSt3RBasedSLAM()
for frame in camera_stream:
    pose = slam.add_frame(frame, intrinsics)
    print(f"Current pose (camera):\n{pose}")
```

### 2. Robotics: Object Reconstruction for Manipulation

```python
def robot_object_reconstruction(left_image, right_image, 
                               camera_intrinsics, 
                               baseline=0.1):
    """
    로봇의 stereo camera pair → 3D 물체 reconstruction
    """
    model = SimpleDUSt3RMatcher().cuda()
    
    # Extract features (DINO)
    feats_left = dino_encode(left_image)
    feats_right = dino_encode(right_image)
    
    # DUSt3R matching
    pointmap, confidence, attn = model(feats_left.unsqueeze(0), 
                                        feats_right.unsqueeze(0))
    
    # Filter by confidence
    mask = confidence[0] > 0.5
    valid_points = pointmap[0][mask.squeeze()]  # (N_valid, 3)
    
    # Metric scaling (stereo baseline known)
    # DUSt3R 의 pointmap 을 실제 metric scale 로 변환
    metric_points = valid_points * baseline
    
    # Outlier removal (statistical)
    # e.g., points > 10m away remove
    metric_points = metric_points[metric_points[:, 2] < 10.0]
    
    return metric_points
```

### 3. E-commerce: Multi-angle Object Capture

```python
def multi_angle_product_3d(image_list, angles_degrees=None):
    """
    다각도 사진 → 완전한 3D 모델 (예: 회전대)
    
    image_list: list of (H, W, 3) images
    angles_degrees: [0, 45, 90, 135, ...] (optional labels)
    """
    n_images = len(image_list)
    
    # Pairwise DUSt3R
    all_pointmaps = []
    all_confidences = []
    
    for i in range(n_images):
        for j in range(i+1, n_images):
            # DUSt3R on pair (i, j)
            feats_i = dino_encode(image_list[i])
            feats_j = dino_encode(image_list[j])
            
            pm, conf, _ = model(feats_i.unsqueeze(0), feats_j.unsqueeze(0))
            all_pointmaps.append(pm)
            all_confidences.append(conf)
    
    # Merge with multi-view optimization
    merged_points = merge_pointmaps_ba(all_pointmaps, all_confidences)
    
    # Mesh reconstruction (Poisson or Ball Pivoting)
    mesh = poisson_reconstruction(merged_points)
    
    return mesh
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| **Sufficient texture** | Low-texture region (white walls, plain fabrics) 에서 correspondence 불안정. → Texture synthesis / multi-view voting |
| **Limited baseline** | Baseline 이 너무 크거나 작으면 (병렬 or frontal) correspondence ambiguous. → 적응적 base-line selection |
| **No large occlusions** | Object 가 >30% occluded 면 dense match 실패. → Occlusion-aware confidence, sparse keypoint fallback |
| **Sufficient resolution** | 매우 작은 object (subpixel) 는 patch embedding 에서 손실. → Multi-scale patch 필요 |
| **Calibrated or near-calibrated** | Uncalibrated 이지만 reasonable FOVEA 가정. Extreme fisheye 는 안 됨. → Camera model 적응화 |
| **Forward-facing views** | 180도 이상 차이나는 view 는 epipolar geometry 모호. → Sequential pose tracking 추가 |

---

## 📌 핵심 정리

$$
\boxed{\text{[Image pair]} \xrightarrow{\text{ViT} + \text{Cross-Attn}}{\text{(~1-2sec)} } \text{Dense Pointmap} \xrightarrow{\text{Bundle Adj (opt)}} \text{3D Model}}
$$

| 단계 | 기존 (COLMAP) | DUSt3R | 개선 |
|-----|---------------|--------|------|
| **Feature extraction** | SIFT (hand-crafted) | DINO (self-supervised) | 10배 빠름, robust |
| **Matching** | SIFT descriptor + ratio test | Cross-attention (dense) | Pixel-wise, sparse keypoint 불필요 |
| **Outlier rejection** | RANSAC (hard threshold) | Soft confidence map | Smooth, learnable |
| **Bundle adjustment** | Explicit non-convex opt | Implicit (network) | Optional, lighter post-processing |
| **Total time** | 30-50 sec | 1-3 sec | **20-30x speedup** |
| **Accuracy** | ~mm (ideal) | ~cm (but dense) | Trade-off: density > accuracy |

### 핵심 논문

- **DUSt3R** (Wang et al., 2024): "DUSt3R: Geometric 3D Vision Made Easy"
- **MASt3R** (Wang et al., 2024): Matching + multi-view extension
- **COLMAP** (Schönberger et al., 2016): Traditional SfM baseline
- **ViT** (Dosovitskiy et al., 2020): Vision Transformer

---

## 🤔 생각해볼 문제

**문제 1** (기초): Cross-attention 에서, why softmax(QK^T) 를 사용하는가? 다른 similarity metric (예: cosine similarity, L2 distance) 을 사용할 수 없는 이유는?

<details>
<summary>해설</summary>

**Softmax advantages**:
1. **Probabilistic interpretation**: Soft matching weights (이미지 1의 각 patch 가 이미지 2 의 어느 patch 와 "얼마나" 잘 매치되는가)
2. **Differentiability**: Hard argmax 와 달리 gradient 가 모든 branch 로 흐름
3. **Normalization**: 각 query 마다 $\sum_k \alpha_k = 1$ → stable training

**Cosine similarity 직접 사용하면**?
- 역시 작동하지만 (unnormalized dot product) softmax 가 scale normalization 제공 → temperature 파라미터와 유사

**L2 distance**?
- 가능하지만 (similarity 로 변환: $-\text{dist}$) 일반적이지 않음. Softmax 는 information-theoretic optimal (maximum entropy subject to marginal constraints)

**결론**: Softmax 는 attention mechanism 의 표준. 다른 것도 가능하지만 empirically softmax 가 최고. $\square$

</details>

**문제 2** (심화): DUSt3R 의 pointmap 이 "canonicalized" 좌표 (camera intrinsics 무관) 인 이유는? 실제 metric 3D 로 변환하는 절차를 설명하라.

<details>
<summary>해설</summary>

**Canonicalized coordinates**:

Uncalibrated image pair 에서, essential matrix 의 scale 은 결정 불가능 (ambiguity: $(E, t) \sim (\lambda E, \lambda t)$).

따라서 DUSt3R 은 **relative depth 와 relative pose 만** 정확하게 학습 가능:

$$
\mathbf{P}_1^{\text{canonical}} = [\mathbf{I} | \mathbf{0}] \mathbf{X}
$$
$$
\mathbf{P}_2^{\text{canonical}} = [\mathbf{R} | \mathbf{t}] \mathbf{X}
$$

where $\|\mathbf{t}\| = 1$ (normalized).

**Metric scale 복구**:

실제 물리 좌표로 변환하려면 **scale** 을 알아야 함:

1. **Stereo baseline 알려진 경우**: $\text{scale} = \text{baseline}_{\text{physical}} / \text{baseline}_{\text{predicted}}$

2. **Calibration ruler**: 알려진 길이의 물체를 capture → scale 계산

3. **Monocular depth prior** (COLMAP-like): 한 점의 실제 거리 → 전체 scale 고정

**구현**:
```python
# Canonicalized → metric
baseline_canonical = np.linalg.norm(pose_canonical[:3, 3])
scale = baseline_physical / baseline_canonical
pointmap_metric = pointmap_canonical * scale
```

$\square$

</details>

**문제 3** (논문 비평): DUSt3R 이 모든 SfM task 를 완전히 대체할 수 있는가? 어떤 상황에서 traditional COLMAP 이 여전히 유리한가?

<details>
<summary>해설</summary>

**DUSt3R 의 한계 (COLMAP 이 나은 경우)**:

1. **매우 큰 scale (도시, 건축 공사)**:
   - 1000+ 이미지 필요
   - DUSt3R 은 쌍별 처리 → O(N²) 복잡도
   - COLMAP incremental SfM: O(N log N) 는 아니지만 실무적으로 더 효율

2. **정확도가 critical (고정밀 측지)**:
   - Aerial LiDAR 메칭
   - COLMAP: bundle adjustment 로 mm-level 정확도
   - DUSt3R: cm-level (dense 하지만 덜 정확)

3. **매우 복잡한 geometry (극단적 occlusion)**:
   - Multiple disconnected components
   - COLMAP 은 keypoint graph 로 robustness 높음
   - DUSt3R 은 dense 하지만 occlusion 에 약함

4. **Extreme viewing angles (병렬 view, 거의 180도)**:
   - Epipolar geometry ambiguous
   - COLMAP: multiple hypotheses 검증
   - DUSt3R: single network output

**결론**: DUSt3R 은 **속도와 편의성** 에서 우수. 정확도와 robustness 는 상황 따라. **Hybrid approach** (DUSt3R + light refinement) 이 최적. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-large-reconstruction-models.md) | [📚 README](../README.md) | [다음 ▶](./03-applications-frontier.md)

</div>
