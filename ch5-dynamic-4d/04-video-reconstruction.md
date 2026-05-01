# 04. Monocular Video → 4D Reconstruction · Camera Estimation

## 🎯 핵심 질문

- 단일 비디오에서 **카메라 trajectory** 를 어떻게 추정하는가? (COLMAP SfM vs MASt3R feed-forward)
- Multi-view consistency loss 는 무엇이고, **depth + optical flow + photometric** 을 어떻게 결합하는가?
- Regularizer (as-rigid-as-possible, sparsity, smooth deformation) 은 왜 필수인가?
- 동적 scene 에서 **dynamic SfM 의 실패** (moving foreground) 와 **partial observability** (가려짐) 을 어떻게 극복하는가?

---

## 🔍 왜 Monocular Video Reconstruction 이 어려운가

다중 view (Ch4) 에서는:
- 각 시간마다 여러 카메라 각도 → strong geometric constraints
- Epipolar geometry, triangulation 으로 3D 직접 계산
- Camera intrinsic/extrinsic 은 정확히 알려짐

**Monocular (단일 카메라) 의 도전**:

1. **Camera motion ambiguity**: 같은 optical flow → multiple interpretations
   - 카메라 움직임 vs 물체 움직임 분리 불가능
   - 크기 정보 (scale) 복원 불가능

2. **Structure-from-motion (SfM) 실패**:
   - Moving foreground (사람, 차) 가 있으면 SfM 이 틀림
   - Static background 만 가정 (COLMAP, OpenMVG)
   - Dynamic objects: outliers로 제거됨

3. **Depth 모호성**:
   - Single image → $\infty$-many 3D reconstructions
   - Temporal consistency + appearance 로 정규화 필요

4. **부분 관찰 (Partial observability)**:
   - Occlusion: 가려진 부분 학습 불가
   - 시간에 따라 visible/invisible 전환

---

## 📐 수학적 선행 조건

- **Ch1-03**: Volume rendering, ray-camera geometry
- **Ch1**: SfM, epipolar geometry, triangulation
- **Ch4**: Gaussian Splatting, rasterization
- **Ch1-02**: Optical flow, photometric loss
- 최적화: Bundle adjustment, Gauss-Newton

---

## 📖 직관적 이해

### Camera Trajectory 추정의 두 가지 전략

#### 전략 1: COLMAP (Incremental SfM)

```
Video frames → Keypoint matching → Incremental SfM
                  ↓                        ↓
            Descriptor extraction    Camera poses
            (SIFT, SuperPoint)       3D points
```

**작동**:
1. Frame 들 간 keypoint matching
2. Initialize: 2 frame 에서 triangulation
3. 점진적 추가: 기존 3D point 관찰하는 new frame 추가
4. Bundle adjustment: camera + 3D point jointly optimize

**장점**: 견고한 기하학적 기초
**단점**: 동적 object (카메라 앞의 움직이는 사람) 가 outlier → SfM 파괴

#### 전략 2: MASt3R (Feed-forward Depth + Camera)

```
Pair of frames → Vision Transformer
                      ↓
              Depth + Camera relative pose
                 (feed-forward, no SfM)
```

**작동**:
1. Trained on large-scale multi-view data (DL3DV, etc.)
2. Input: 2 frames → Output: depth + relative camera pose
3. Monocular sequence: consecutive frame pair 에 apply
4. Incremental composition: $T_0 = I$, $T_i = T_{i-1} \circ T_{i, i-1}$

**장점**: 동적 object 에 robust (학습 기반), 빠름
**단점**: Scale ambiguity (monocular), 누적 drift

### Multi-View Consistency Loss

데이터 term (photometric) + geometric constraint (depth, flow):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{photo}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{flow}} \mathcal{L}_{\text{flow}} + \mathcal{L}_{\text{reg}}$$

#### 1. Photometric Loss

$$\mathcal{L}_{\text{photo}} = \frac{1}{N} \sum_{i, \mathbf{p}} \rho(I_i(\mathbf{p}) - \hat{I}_i(\mathbf{p}))$$

where $\rho$ 는 robust loss (e.g., Huber), $\hat{I}_i$ 는 rendered image.

#### 2. Depth Consistency Loss

Consecutive frames 에서 depth 가 smooth 해야 함:

$$\mathcal{L}_{\text{depth}} = \sum_{t} \|\nabla D(t) - \nabla D(t-1)\|^2$$

또는 warping 을 통한 cycle consistency:
$$\mathcal{L}_{\text{warp}} = \|D(t) - \text{project\_back}(D(t-1), \text{flow}_{t-1 \to t})\|^2$$

#### 3. Optical Flow Loss

Estimated deformation field 와 learned optical flow 의 일치:

$$\mathcal{L}_{\text{flow}} = \frac{1}{N} \sum_{\mathbf{p}} \|F(\mathbf{p}, t) - \text{flow}(\mathbf{p}, t)\|^2$$

where $F$ 는 deformation field 로 유도된 flow.

#### 4. Regularizers

**ARAP (As-Rigid-As-Possible)**:
$$\mathcal{L}_{\text{ARAP}} = \sum_{\text{edges}} \|(\mathbf{x}' - \mathbf{x}) - \mathbf{R}(\mathbf{x}_0' - \mathbf{x}_0)\|^2$$

→ 국소 정동(rigid-like) 변형 유도

**Sparsity**:
$$\mathcal{L}_{\text{sparse}} = \sum_i |\sigma_i|$$

→ 불필요한 Gaussian 제거

**Smooth Deformation**:
$$\mathcal{L}_{\text{smooth}} = \sum_t \|\nabla_t D(t)\|^2$$

→ temporal smoothness

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Monocular Camera Trajectory

Frame sequence $(I_0, I_1, \ldots, I_T)$ 에 대해, camera trajectory:

$$\mathbf{T} = \{T_0, T_1, \ldots, T_T\}, \quad T_i \in SE(3)$$

where $T_i$ 는 world-to-camera 변환 (extrinsic):

$$\mathbf{x}_{\text{cam}} = T_i \mathbf{x}_{\text{world}}$$

### 정의 4.2 — Depth Map 과 Reprojection

Depth map $D_i: \mathbb{R}^2 \to \mathbb{R}^+$, pixel coordinate $\mathbf{u} = (u, v)$ 에서:

$$\mathbf{X}_i(\mathbf{u}) = D_i(\mathbf{u}) \cdot K^{-1} \mathbf{u}_{\text{hom}} \quad \text{(camera frame)}$$

다른 frame $j$ 로의 reprojection:

$$\mathbf{u}_j = K [T_j T_i^{-1}] \mathbf{X}_i(\mathbf{u})_{\text{hom}}$$

### 정의 4.3 — Optical Flow

Consecutive frames $I_t, I_{t+1}$ 사이의 pixel-level motion:

$$\mathbf{f}(t) : \mathbb{R}^2 \to \mathbb{R}^2, \quad I_{t+1}(\mathbf{u} + \mathbf{f}(t, \mathbf{u})) \approx I_t(\mathbf{u})$$

(brightness consistency assumption)

Geometric origin (camera + deformation):

$$\mathbf{f}_{\text{geom}}(t, \mathbf{u}) = \pi(T_{t+1} T_t^{-1} \pi^{-1}(\mathbf{u}, D_t(\mathbf{u})))$$

where $\pi$ 는 perspective projection.

### 정의 4.4 — 4D 재구성 문제

주어진: monocular video $(I_0, \ldots, I_T)$, intrinsic $K$

구하기: camera trajectory $\mathbf{T}$, 4D scene representation (NeRF or 3DGS):
$$\Phi = \{F_\theta, D_\psi\} \text{ (NeRF) 또는 } \{G_i(t)\} \text{ (4DGS)}$$

목표:
$$\min_{\mathbf{T}, \Phi} \mathcal{L}_{\text{photo}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{flow}} \mathcal{L}_{\text{flow}} + \mathcal{L}_{\text{reg}}$$

---

## 🔬 정리와 증명

### 정리 4.1 — Monocular SfM Scale Ambiguity

Monocular sequence 에 대해, camera trajectory $T_i$ 를 scale factor $s > 0$ 로 scaling 하면:

$$T_i' = s \cdot T_i = (s\mathbf{R}_i, s\mathbf{t}_i) \quad \text{(scaled extrinsic)}$$

그리고 depth $D'(\mathbf{u}) = s \cdot D(\mathbf{u})$ 로 조정하면,
rendered image 는 동일:

$$I'_i(\mathbf{u}) = I_i(\mathbf{u})$$

**결론**: Monocular visual data 만으로는 absolute scale 불가능 → metric reconstruction 불가, only up-to-scale.

### 정리 4.2 — Epipolar Flow Constraint

Camera motion 으로 인한 optical flow 는 epipolar geometry 를 만족:

$$\mathbf{f}_{\text{flow}}(\mathbf{u}) \cdot [T_{t+1} T_t^{-1}]_\times D(\mathbf{u}) \approx 0$$

where $[\cdot]_\times$ 는 skew-symmetric cross-product matrix.

**적용**: Flow loss 에서 epipolar constraint 를 추가 정규화로 사용 가능.

### 정리 4.3 — Dynamic SfM 실패 (Outlier Dominance)

moving foreground 가 image 의 $p$ fraction 을 차지할 때, COLMAP 같은 robust SfM 은:

- Inlier threshold $\tau$ (일반적으로 pixel error < 3-5)
- Moving object 의 reprojection error 가 $\tau$ 초과 → outlier
- Outlier 개수 > inlier 개수 면 → SfM 완전 실패

**수학적 분석**:

N random matches, inlier ratio $1-p$ 일 때, RANSAC 의 expected iterations:

$$N_{\text{iter}} = \frac{\log(1 - \text{confidence})}{\log(1 - (1-p)^k)}$$

$p$ 가 크면 (많은 moving object) → $N_{\text{iter}} \to \infty$ → practical 실패.

**해법**: 
- Pre-segment: moving mask 먼저 찾기 (optical flow magnitude threshold)
- Or: joint optimization (camera + deformation field) 로 outlier 흡수

---

## 💻 PyTorch 구현 검증

### 실험 1 — COLMAP-style Incremental SfM (simplified)

```python
import torch
import torch.nn as nn
from torch.linalg import solve

class SimpleSfM(nn.Module):
    """Simplified incremental SfM."""
    def __init__(self, num_frames=10):
        super().__init__()
        
        # Camera intrinsic (fixed)
        self.K = torch.tensor([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # Learnable camera poses (SE3)
        self.log_rot = nn.Parameter(torch.zeros(num_frames, 3))    # axis-angle
        self.trans = nn.Parameter(torch.zeros(num_frames, 3))      # translation
        
        # 3D points (sparefully optimized)
        self.points_3d = nn.Parameter(torch.randn(1000, 3))
        self.tracks = []  # list of (point_id, frame_id, pixel_uv)
    
    def get_pose(self, frame_id):
        """Return SE(3) pose for frame."""
        # Log map to matrix (simplified: use axis-angle to rotation)
        from kornia.geometry import axis_angle_to_matrix
        
        rot = axis_angle_to_matrix(self.log_rot[frame_id])  # (3, 3)
        trans = self.trans[frame_id]                         # (3,)
        
        # SE(3) as 4x4 matrix
        pose = torch.eye(4, device=self.log_rot.device)
        pose[:3, :3] = rot
        pose[:3, 3] = trans
        return pose
    
    def reproject(self, point_3d, pose):
        """3D point -> 2D pixel via camera pose."""
        # point_3d: (3,)
        # pose: SE(3) as (4, 4)
        
        point_cam = pose[:3, :3] @ point_3d + pose[:3, 3]  # (3,)
        
        # Perspective projection
        z = point_cam[2]
        x_img = self.K[0, 0] * point_cam[0] / z + self.K[0, 2]
        y_img = self.K[1, 1] * point_cam[1] / z + self.K[1, 2]
        
        return torch.stack([x_img, y_img])
    
    def forward(self, frame_id, point_ids):
        """Compute reprojection error."""
        pose = self.get_pose(frame_id)
        
        errors = []
        for pt_id in point_ids:
            pt_3d = self.points_3d[pt_id]
            uv_pred = self.reproject(pt_3d, pose)
            # Assume ground truth uv stored in self.tracks
            # errors.append((uv_pred - uv_gt) ** 2)
        
        return errors

# Usage
sfm = SimpleSfM(num_frames=10)
optimizer = torch.optim.Adam(sfm.parameters(), lr=1e-3)

# Simulate bundle adjustment
for epoch in range(100):
    # For each observation (point, frame, pixel)
    loss = 0
    # loss += reprojection_error
    
    optimizer.zero_grad()
    # loss.backward()
    optimizer.step()
```

### 실험 2 — Multi-View Consistency Loss

```python
class MonocularVideoLoss(nn.Module):
    """Photometric + geometric consistency loss."""
    def __init__(self):
        super().__init__()
        self.photometric_loss = nn.L1Loss(reduction='mean')
    
    def photometric(self, pred_rgb, gt_rgb, mask=None):
        """Pixel-wise L1 loss."""
        if mask is not None:
            diff = torch.abs(pred_rgb[mask] - gt_rgb[mask])
        else:
            diff = torch.abs(pred_rgb - gt_rgb)
        return diff.mean()
    
    def depth_consistency(self, depth_t, depth_t1, flow_t):
        """
        depth_t: (H, W) depth at time t
        depth_t1: (H, W) depth at time t+1
        flow_t: (H, W, 2) optical flow t -> t+1
        """
        H, W = depth_t.shape
        
        # Backward warping: project depth from t to t+1
        y, x = torch.meshgrid(
            torch.arange(H, device=depth_t.device),
            torch.arange(W, device=depth_t.device),
            indexing='ij'
        )
        
        x_warped = x + flow_t[:, :, 0]
        y_warped = y + flow_t[:, :, 1]
        
        # Bilinear interpolation of depth_t
        depth_warped = self._bilinear_interp(depth_t, x_warped, y_warped)
        
        # Compare with depth_t1
        error = torch.abs(depth_warped - depth_t1)
        return error.mean()
    
    def optical_flow_loss(self, flow_predicted, flow_gt):
        """L2 loss on optical flow."""
        return torch.norm(flow_predicted - flow_gt, dim=-1).mean()
    
    def arap_loss(self, deform_field, x_canonical):
        """
        As-rigid-as-possible: encourage locally rigid deformation.
        Simplified version using Laplacian.
        """
        # Compute graph Laplacian of deformation
        grad_deform = torch.autograd.grad(
            deform_field.sum(), x_canonical, create_graph=True
        )[0]
        
        # Penalize non-rotation part of Jacobian
        # (simplified: just penalize variation)
        return (grad_deform ** 2).mean()
    
    def forward(self, pred_rgb, gt_rgb, depth_t, depth_t1, 
                flow_pred, flow_gt, deform_field, x_can,
                lambda_depth=1.0, lambda_flow=1.0, lambda_arap=0.1):
        
        loss_photo = self.photometric(pred_rgb, gt_rgb)
        loss_depth = self.depth_consistency(depth_t, depth_t1, flow_pred)
        loss_flow = self.optical_flow_loss(flow_pred, flow_gt)
        loss_arap = self.arap_loss(deform_field, x_can)
        
        total_loss = (
            loss_photo +
            lambda_depth * loss_depth +
            lambda_flow * loss_flow +
            lambda_arap * loss_arap
        )
        
        return total_loss, {
            'photo': loss_photo,
            'depth': loss_depth,
            'flow': loss_flow,
            'arap': loss_arap
        }
    
    def _bilinear_interp(self, image, x, y):
        """Bilinear interpolation."""
        x0, y0 = torch.floor(x).long(), torch.floor(y).long()
        x1, y1 = x0 + 1, y0 + 1
        
        H, W = image.shape
        x0 = torch.clamp(x0, 0, W - 1)
        x1 = torch.clamp(x1, 0, W - 1)
        y0 = torch.clamp(y0, 0, H - 1)
        y1 = torch.clamp(y1, 0, H - 1)
        
        wx = x - x0.float()
        wy = y - y0.float()
        
        v00 = image[y0, x0]
        v10 = image[y0, x1]
        v01 = image[y1, x0]
        v11 = image[y1, x1]
        
        result = (
            (1 - wx) * (1 - wy) * v00 +
            wx * (1 - wy) * v10 +
            (1 - wx) * wy * v01 +
            wx * wy * v11
        )
        return result

# Usage
loss_fn = MonocularVideoLoss()

# Training loop (simplified)
for frame_t in range(num_frames - 1):
    pred_rgb, _ = render(model, frame_t)
    gt_rgb = videos[frame_t]
    
    depth_t, _ = model.get_depth(frame_t)
    depth_t1, _ = model.get_depth(frame_t + 1)
    
    flow_pred = optical_flow_network(pred_rgb, render(model, frame_t+1)[0])
    flow_gt = compute_optical_flow(gt_rgb, videos[frame_t+1])
    
    loss, losses_dict = loss_fn(
        pred_rgb, gt_rgb, depth_t, depth_t1,
        flow_pred, flow_gt, deform_field, x_can
    )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 실험 3 — Camera Trajectory 추정 (Joint Optimization)

```python
def joint_optimize(videos, num_frames=10, num_iter=1000):
    """
    Joint optimization: camera trajectory + 4D scene.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Camera trajectory (learnable)
    camera_trajectory = SimpleSfM(num_frames=num_frames).to(device)
    
    # 4D scene (e.g., DynamicNeRF or 4DGS)
    scene = DynamicNeRF().to(device)  # or 4DGS
    
    # Optimizer
    params = list(camera_trajectory.parameters()) + list(scene.parameters())
    optimizer = torch.optim.Adam(params, lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    
    loss_fn = MonocularVideoLoss()
    
    for iteration in range(num_iter):
        total_loss = 0
        
        for frame_idx in range(num_frames - 1):
            # Get camera pose
            pose = camera_trajectory.get_pose(frame_idx)
            
            # Render current & next frame
            pred_rgb_t, depth_t = render_with_pose(scene, pose, frame_idx)
            pred_rgb_t1, depth_t1 = render_with_pose(
                scene, camera_trajectory.get_pose(frame_idx + 1), frame_idx + 1
            )
            
            # Compute losses
            gt_rgb_t = videos[frame_idx].to(device)
            gt_rgb_t1 = videos[frame_idx + 1].to(device)
            
            # Optical flow from predicted images
            flow_pred = estimate_flow(pred_rgb_t, pred_rgb_t1)  # e.g., RAFT
            flow_gt = estimate_flow(gt_rgb_t, gt_rgb_t1)
            
            loss, _ = loss_fn(
                pred_rgb_t, gt_rgb_t,
                depth_t, depth_t1,
                flow_pred, flow_gt,
                scene.deform, None
            )
            
            total_loss += loss
        
        # Average loss over all frame pairs
        total_loss = total_loss / (num_frames - 1)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if iteration % 100 == 0:
            print(f"Iter {iteration}: loss = {total_loss.item():.6f}")
    
    return camera_trajectory, scene
```

---

## 🔗 실전 활용

### 1. Depth-from-Video Pipeline (Recent Methods)

**DUSt3R-Dynamic (Wang 2024)**:
- Input: monocular video
- Output: per-frame point clouds (depth)
- Key: dynamic objects 감지 → mask out from SfM

**Shape-of-Motion (Wang 2024)**:
- Scene decomposition: static + dynamic parts
- Each part separate optimization
- Better handling of moving foreground

### 2. Multi-Modal Fusion

```python
# Combine multiple depth sources
def fuse_depths(depth_colmap, depth_mast3r, depth_mono_prior):
    """
    depth_colmap: SfM result (sparse)
    depth_mast3r: feed-forward estimate (dense)
    depth_mono_prior: monocular cue (MiDaS, etc.)
    """
    
    # Uncertainty weights
    conf_colmap = compute_confidence(depth_colmap)  # higher for well-triangulated
    conf_mast3r = compute_confidence(depth_mast3r)
    conf_prior = compute_confidence(depth_mono_prior)
    
    # Weighted average
    fused_depth = (
        conf_colmap * depth_colmap +
        conf_mast3r * depth_mast3r +
        conf_prior * depth_mono_prior
    ) / (conf_colmap + conf_mast3r + conf_prior + 1e-8)
    
    return fused_depth
```

### 3. Occlusion Handling

```python
def handle_occlusion(pred_rgb, gt_rgb, depth_t, depth_t1, flow_pred):
    """
    Detect & mask occlusion regions.
    """
    # Forward & backward flow consistency
    flow_bwd = estimate_flow(videos[t+1], videos[t])  # t+1 -> t
    
    # Occlusion mask: large flow magnitude or inconsistent flow
    occ_mag = torch.norm(flow_pred + flow_bwd, dim=-1)  # cycle consistency
    occ_mask = occ_mag > THRESHOLD
    
    # Mask out occluded regions from loss
    valid_mask = ~occ_mask
    
    # Recompute loss with valid_mask
    loss_photo = torch.abs(pred_rgb[valid_mask] - gt_rgb[valid_mask]).mean()
    
    return loss_photo, valid_mask
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Static background (SfM assumption) | Dynamic object 많으면 SfM fail → joint opt. 또는 segmentation |
| Smooth camera motion | Fast panning/zoom → temporal window 짧게 |
| Visible surface consistency | Heavy occlusion → inpainting loss 추가 |
| Brightness constancy | Illumination change → photometric loss robust variant |
| Known intrinsic K | Auto-calibration 필요 (Hartley 방법, 또는 learnable) |
| Single scale | Monocular scale ambiguity → depth prior (MiDaS) 또는 IMU fusion |

---

## 📌 핵심 정리

$$\boxed{\min_{\mathbf{T}, \Phi} \mathcal{L}_{\text{photo}} + \lambda_d \mathcal{L}_{\text{depth}} + \lambda_f \mathcal{L}_{\text{flow}} + \mathcal{L}_{\text{reg}}}$$

| 성분 | 역할 | 도전 |
|------|------|------|
| $\mathcal{L}_{\text{photo}}$ | RGB 일치 | illumination 변화 |
| $\mathcal{L}_{\text{depth}}$ | depth smoothness, consistency | occlusion 불확실성 |
| $\mathcal{L}_{\text{flow}}$ | motion field 정합성 | dynamic object ambiguity |
| Camera $\mathbf{T}$ | trajectory 추정 | scale ambiguity, drift |
| Deformation $D_\psi$ | dynamic object motion | structure uniqueness |

**핵심**: Monocular → ambiguous → multi-term loss + strong regularization.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Optical flow loss 와 depth consistency loss 는 왜 둘 다 필요한가? Flow 만으로는 부족한가?

<details>
<summary>해설</summary>

Optical flow 만으로:
- 2D pixel motion 만 알 수 있음
- 3D camera motion과 object deformation 구분 불가

Depth consistency:
- 3D geometric constraint 추가
- 같은 물체의 다른 부분이 일관된 depth 가져야 함
- Deformation field 의 물리학적 타당성 강제

**예**: 두 개의 다른 optical flow 가 같은 depth 변화를 yield 할 수 있음. 
- Camera rotates: flow large, depth constant
- Object deforms: flow small, depth changes

따라서 **둘 다 필요** — geometric ambiguity 해소. $\square$

</details>

**문제 2** (심화): Dynamic SfM (moving foreground 처리) 의 해결책은? RANSAC threshold 를 높이면 어떻게 되는가?

<details>
<summary>해설</summary>

RANSAC threshold 증가:
- Pros: moving object 의 inliers 포함 → SfM "돌아감"
- Cons: reprojection error threshold 가 매우 커짐 (5→20 pixels)
  - Actual outliers (noise, mismatch) 도 inlier 로 pass
  - SfM 정확도 급격히 감소 → camera pose 틀림

**올바른 접근**:

1. **Pre-segmentation**: 
   - Optical flow magnitude threshold (큰 flow = moving object)
   - Remove moving pixels before SfM

2. **Robust SfM**:
   - Multiple hypothesis testing (세그먼트별 separate SfM)
   - Select hypothesis with best photometric consistency

3. **Joint optimization**:
   - SfM + scene representation joint optimize
   - Camera pose 와 deformation field 가 서로 correct

**최신**: DUSt3R-Dynamic, Shape-of-Motion 이 이를 구현. $\square$

</details>

**問題 3** (论文评论): Scale ambiguity 를 해결하는 현실적인 방법은? Depth prior (e.g., MiDaS) 는 충분히 정확한가?

<details>
<summary>해설</summary>

Monocular scale ambiguity:
- Visual data 만으로는 해결 불가 (정리 4.1)
- 필요: scale reference (metric depth, IMU, LiDAR)

**Depth prior (MiDaS, DPT) 의 한계**:
- 절대 scale: 부정확 (average depth 예측, 통계적)
- 상대 scale (nearby points): 더 정확
- Confidence: 카메라로부터 거리가 멀수록 noise 증가

**실무 해결책**:

1. **Metric depth device** (RGB-D camera):
   - 몇 frame 에서 ground truth depth 획득
   - Scale 정규화: $D_{\text{scale}} = s \cdot D_{\text{learned}}$ where $s = \text{median}(D_{\text{gt}} / D_{\text{learned}})$

2. **Camera height prior**:
   - 카메라 높이 알려짐 (카메라 탑재 기계 높이)
   - Ground plane depth 로부터 scale 복원

3. **Temporal consistency + temporal depth prior**:
   - Consecutive frames 에서 camera motion 이 small → temporal derivatives 로 scale 자체 복원 가능 (iterative refinement)

**결론**: Monocular depth prior 는 supportive 만 (초기값), metric reconstruction 은 추가 sensor 필요. $\square$

</details>

---

<div align="center">

[◀ 이전](./03-4d-gaussian-splatting.md) | [📚 README](../README.md) | [다음 ▶](../ch6-text-to-3d/01-problem-setup.md)

</div>
