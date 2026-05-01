# 03. 응용과 Frontier — Spatial Computing · Robotics · Autonomous Driving

## 🎯 핵심 질문

- **Apple Vision Pro** 같은 spatial computing platform 에서, LRM · DUSt3R 의 어떤 특성이 real-time 3D capture 를 가능하게 하는가?
- Robotics simulation (Habitat, Isaac Sim) 에서 NeRF · GS 기반 assets 이 "렌더링만" 해주는 것 이상의 가치를 제공하는 이유는?
- **4D (video) reconstruction** 이 3D foundation models 의 자연스러운 확장인 이유는?
- Autonomous driving 에서 3D reconstruction 이 sensor fusion, path planning, safety 에 어떻게 기여하는가?
- 2024~2025 의 frontier: world models + 3D 의 fusion 이 왜 다음 혁신인가?

---

## 🔍 왜 Foundation Models 가 응용의 중심이 되는가

### 2023-2024 의 패러다임 변화

**이전** (2022): NeRF · SDS 는 **연구 도구** — "신기한 것" 을 만들지만 실시간 엔지니어링에는 부적합.

**현재** (2024): LRM · DUSt3R · GS-SLAM 은 **프로덕션 infrastructure** — real-time 3D capture, reconstruction, rendering 이 초 단위 안에.

**동인**:
1. **Speed**: SDS 의 500초 → LRM 의 5초 (100배)
2. **Determinism**: Generative 의 불확실성 제거, reproducible results
3. **Scale**: Billions of 3D models (Objaverse) 로부터 학습된 3D prior
4. **Hardware acceleration**: CUDA triplane, Gaussian splatting kernel

→ **"3D 가 이제 practical data modality"** — 이미지처럼 capture, compress, render.

---

## 📐 수학적 선행 조건

### NeRF Rendering (복습)

$$
C(\mathbf{r}) = \int_0^\infty T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t)) dt, \quad T(t) = \exp\left(-\int_0^t \sigma(\mathbf{r}(s)) ds\right)
$$

**discretized** (volume rendering integrator):

$$
C(\mathbf{r}) \approx \sum_{i=1}^{N} T_i \alpha_i \mathbf{c}_i, \quad T_i = \prod_{j=1}^{i-1} (1 - \alpha_j), \quad \alpha_i = 1 - \exp(-\sigma_i \delta_i)
$$

### 3D Gaussian Splatting

각 Gaussian $\mathcal{G}_k$ 는 $(\\boldsymbol{\mu}_k, \Sigma_k, c_k, o_k)$ (center, covariance, color, opacity):

$$
G_k(\mathbf{x}) = o_k \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_k)^T \Sigma_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k)\right)
$$

**Rasterization** (Kerbl 2023): 각 pixel 마다 sorted order 로 blending.

$$
\mathbf{C}(u) = \sum_k \mathbf{c}_k \alpha_k \prod_{j < k} (1 - \alpha_j)
$$

→ **100+ FPS** (NeRF 의 ~1 FPS 대비).

### World Models (Sora, Genie)

비디오 diffusion 에서 "world model" = "어떤 environment 의 implicit 3D + dynamics 이해" (학습되지만 unsupervised).

$$
\mathbf{v}_{t+1} = f_\theta(\mathbf{v}_t, \mathbf{a}_t) + \epsilon_t
$$

(여기서 $\mathbf{v}_t$ 는 latent video state, $\mathbf{a}_t$ 는 action).

---

## 📖 직관적 이해

### Application 1: Spatial Computing (Apple Vision Pro, Meta Quest)

```
User captures room with LiDAR + camera
    ↓
[DUSt3R: 이전 frame 과의 pair → dense pointcloud]
    ↓
[GS-SLAM: pointcloud → Gaussians (Matsuki 2024)]
    ↓
[Real-time rendering: 100+ FPS at 4K/eye]
    ↓
[Spatial UI, occlusion, shadows]
    ↓
Digital objects 가 physical space 에 자연스럽게 배치
```

**이전**: 전용 reconstruction engine (Intel RealSense Viewer 같은) — 무겁고, camera 종속적.

**현재**: Foundation model 기반 — 어떤 input (monocular, stereo, LiDAR) 이든 호환.

### Application 2: Robotics Simulation

```
Real robot camera frame
    ↓
[LRM or DUSt3R: 3D reconstruction]
    ↓
[NeRF or GS 로 Habitat-Sim 에 import]
    ↓
[Simulator 에서 robot controller policy training]
    ↓
[Sim → Real transfer learning]
```

**가치**:
- **Speed**: Real-world capture → sim asset in seconds (vs manual 3D modeling days)
- **Photorealism**: NeRF/GS rendering 이 "visual fidelity" → 더 나은 policy 학습
- **Generalization**: Diverse assets → diverse policies

**예**: NVIDIA Habitat-Sim with NeRF assets (Tancik 2022 Block-NeRF extension).

### Application 3: Autonomous Driving

```
Multi-camera (8x) + LiDAR + Radar
    ↓
[DUSt3R on sliding window pairs: dense depth + optical flow]
    ↓
[3D Scene reconstruction: cars, pedestrians, road geometry]
    ↓
[4D analysis: motion prediction, collision avoidance]
    ↓
[End-to-end planning (Imitation Learning / RL)]
```

**vs traditional** (BEV-based, learned depth from monocular):
- **Density**: Explicit dense geometry (not learned, implicit)
- **Uncertainty**: Confidence maps → "we don't know" regions 인식
- **Multi-modal**: Multiple cameras → robust fusion

---

## ✏️ 엄밀한 정의

### 정의 7.9 — Spatial Computing Pipeline

**입력**: 모바일 AR device 의 camera stream $\{I_t\}_{t=1}^{T}$ (30 FPS) + optional depth sensor.

**Processing**:
1. **Keyframe selection**: motion 이 충분하면 keyframe 으로 추가 (모든 frame 아님)
2. **Pairwise matching**: keyframe $I_i$ 와 $I_j$ 사이 DUSt3R 또는 incremental LRM
3. **Pointcloud fusion**: SLAM-style bundle adjustment (light)
4. **Gaussian optimization**: pointcloud → 3D Gaussians (online optimization)
5. **Rendering**: user 시점에서 real-time GS splatting

**출력**: 
- Rendered RGB + Depth (per eye, spatial)
- Confidence map (uncertain region)
- Geometry for occlusion, physics

### 정의 7.10 — 4D Reconstruction (Video)

3D model 이 시간에 따라 변하는 경우:

$$
\text{Scene}(t) = \{\text{3D assets, camera pose, deformation}\}_t
$$

**Representation**:
- **Baseline**: Video diffusion (Sora, Genie) 의 learned latent space
- **Explicit**: 시간별 NeRF or GS (Neural Volumes, ENeRF, 4D-GS)
- **Hybrid**: 3D foundation + video diffusion (multimodal training)

### 정의 7.11 — Autonomous Driving 의 3D Foundation Role

Perception → Planning 파이프라인:

$$
\{\mathbf{I}_1, \ldots, \mathbf{I}_8, \text{LiDAR}\} \xrightarrow{3D \text{ foundation}} \text{Scene Graph} \xrightarrow{\text{Planner}} \text{Actions}
$$

**Scene Graph**: 
$$
G = (V, E), \quad v \in V = \{\text{agents, road, obstacles}\}, \quad e \in E = \{\text{spatial, temporal relations}\}
$$

---

## 🔬 정리와 증명

### 정리 7.5 (Spatial Computing 의 Real-time Feasibility)

LRM (5ms) + GS rendering (10ms per frame) = 15ms latency → 66 FPS.

**증명**:

Mobile GPU (A17 Pro, Snapdragon 8 Gen 2) 의 throughput 분석:
- ViT-L: ~50 TFLOPS (available)
- LRM 추론: ~100 G multiplications = 2ms (with optimizations: quantization, distillation)
- GS splatting: ~10ms (tile-based rasterization, per-pixel sort)

Total: ~15ms per frame → **66 FPS 가능**, 현실: 30-40 FPS (conservative).

**의의**: "Real-time spatial computing 이 가능" 은 이제 hardware + algorithm 측면에서 증명됨. $\square$

### 정리 7.6 (Robotics Sim-to-Real Gap 감소)

**명제**: NeRF/GS 로부터 생성한 synthetic scene 으로 훈련한 policy 는, equivalent photorealistic simulation (Mujoco-100x slower) 으로 훈련한 것과 similar downstream task performance 를 달성한다.

**증명 sketch** (empirical):

1. **Visual domain randomization**: NeRF rendering + random lighting → diverse visual inputs
2. **Geometric accuracy**: DUSt3R dense reconstruction → real geometry 근접
3. **Embodied AI 에서 "sufficiently realistic"** : Robot grasping, navigation 은 사실상 visual feature learning — pixel-level photorealism 은 critical 아님 (semantic consistency 만 필요)

**실험** (가상):
- Habitat-Sim with NeRF: policy A (accuracy 82%)
- Real robot evaluation: 78% success → sim-to-real gap 4%
- Traditional depth-only sim: gap 15-20%

$\square$

### 따름 정리 7.7 (4D 재구성의 Scalability)

Temporal coherence 있는 4D NeRF training 은, 3D NeRF + frame-wise deformation 으로 분해 가능:

$$
\mathbf{f}(\mathbf{x}, t) = \mathbf{T}_t(\mathbf{x}) \to \sigma(\mathbf{f}(\mathbf{x})), \mathbf{c}(\mathbf{f}(\mathbf{x}))
$$

→ Efficient training + inference: 시간별 warp field 만 학습하면 됨 (전체 NeRF 재훈련 아님).

---

## 💻 구현 검증

### 실험 1 — Spatial Computing Mock (Keyframe SLAM + GS)

```python
import torch
import numpy as np
from collections import deque

class RealTimeSpatialComputing:
    """
    AR device 에서 spatial computing: 
    DUSt3R-based SLAM + Gaussian splatting
    """
    def __init__(self, max_keyframes=20, device='cuda'):
        self.device = device
        self.keyframes = deque(maxlen=max_keyframes)
        self.pointcloud = None
        self.gaussians = None
        self.current_pose = np.eye(4)
        
        # Models (mock)
        self.dust3r = SimpleDUSt3RMatcher().to(device)
        self.renderer = GaussianSplatRenderer(device=device)
    
    def add_frame(self, rgb_frame, intrinsics, prev_gray=None):
        """
        새 frame 처리 (30 FPS input, sparse keyframe selection)
        """
        # Quick motion estimation (optical flow or descriptor matching)
        if len(self.keyframes) > 0:
            prev_kf = self.keyframes[-1]
            motion = self.estimate_motion(rgb_frame, prev_kf['rgb'])
            
            if np.linalg.norm(motion[:3, 3]) < 0.01:  # motion too small
                return None  # skip this frame
        
        # Keyframe 으로 등록
        image_tensor = torch.from_numpy(rgb_frame).float().to(self.device)
        
        # Feature extraction (DINO mock)
        features = self.dino_encode_mock(image_tensor)
        
        # DUSt3R matching with previous keyframe
        if len(self.keyframes) > 0:
            prev_features = self.keyframes[-1]['features']
            pointmap, confidence, _ = self.dust3r(
                features.unsqueeze(0), prev_features.unsqueeze(0)
            )
            
            # Add points to cloud
            valid_mask = confidence[0, :, 0] > 0.3
            new_points = pointmap[0][valid_mask].cpu().numpy()
            
            if self.pointcloud is not None:
                self.pointcloud = np.vstack([self.pointcloud, new_points])
            else:
                self.pointcloud = new_points
            
            # Pose estimation (PnP)
            self.current_pose = self.estimate_pose_pnp(
                pointmap[0], confidence[0], intrinsics
            )
        
        # Store keyframe
        self.keyframes.append({
            'rgb': rgb_frame,
            'features': features,
            'pose': self.current_pose.copy()
        })
        
        # Periodically optimize Gaussians
        if len(self.keyframes) % 5 == 0:
            self.optimize_gaussians()
        
        return self.current_pose
    
    def optimize_gaussians(self):
        """pointcloud → 3D Gaussians (sparse sampling)"""
        if self.pointcloud is None or len(self.pointcloud) == 0:
            return
        
        # Voxel downsampling
        from sklearn.cluster import KMeans
        n_gaussians = min(max(1000, len(self.pointcloud) // 100), 10000)
        kmeans = KMeans(n_clusters=n_gaussians, random_state=0)
        cluster_centers = kmeans.fit_predict(self.pointcloud)
        
        # Initialize Gaussians at cluster centers
        self.gaussians = {
            'centers': torch.from_numpy(kmeans.cluster_centers_).float(),
            'opacities': torch.ones(n_gaussians) * 0.5,
            'scales': torch.ones(n_gaussians, 3) * 0.1,
            'colors': torch.rand(n_gaussians, 3)
        }
    
    def render_view(self, camera_pose, intrinsics, H=1080, W=1920):
        """현재 카메라 시점에서 rendering"""
        if self.gaussians is None:
            return np.zeros((H, W, 3), dtype=np.uint8)
        
        # GS splatting (mock)
        rgb = self.renderer.render(
            self.gaussians, camera_pose, intrinsics, H, W
        )
        
        return (rgb * 255).astype(np.uint8)
    
    def dino_encode_mock(self, image_tensor):
        """Mock DINO encoder (real: frozen ViT-L)"""
        # Simplified: conv layers → patch features
        h, w = image_tensor.shape[-2:]
        patch_h, patch_w = h // 14, w // 14
        return torch.randn(patch_h, patch_w, 256)  # mock features

# 사용: AR 카메라 stream
device = 'cuda' if torch.cuda.is_available() else 'cpu'
spatial_computer = RealTimeSpatialComputing(device=device)

# Simulate frame streaming (30 fps)
for frame_idx in range(100):
    rgb = np.random.rand(720, 1280, 3).astype(np.float32)
    intrinsics = np.array([
        [500, 0, 640],
        [0, 500, 360],
        [0, 0, 1]
    ]).astype(np.float32)
    
    pose = spatial_computer.add_frame(rgb, intrinsics)
    
    if frame_idx % 10 == 0:
        # Render to display
        rendered = spatial_computer.render_view(pose, intrinsics)
        print(f"Frame {frame_idx}: rendered shape {rendered.shape}, "
              f"n_gaussians={len(spatial_computer.gaussians['centers']) if spatial_computer.gaussians else 0}")
```

### 실험 2 — Robotics Asset Import for Simulation

```python
def robot_sim_asset_from_capture(real_robot_image, sim_engine='habitat'):
    """
    로봇이 본 실제 물체 → simulation asset
    """
    # Step 1: 3D reconstruction (LRM)
    lrm_model = TriplaneRegressor().cuda()
    
    # Mock feature extraction
    features = torch.randn(1, 256, 256).cuda()
    triplane = lrm_model(features.unsqueeze(0))
    
    # Step 2: Export to mesh
    verts, faces = triplane_to_mesh(triplane)  # marching cubes
    
    # Step 3: Import to simulator
    if sim_engine == 'habitat':
        # Habitat asset format
        asset = {
            'vertices': verts,
            'faces': faces,
            'texture': render_triplane_texture(triplane),
        }
        return asset
    
    elif sim_engine == 'isaac':
        # NVIDIA Isaac format
        import pxr
        from pxr import Usd, UsdGeom
        
        stage = Usd.Stage.CreateNew("asset.usd")
        xform = UsdGeom.Xform.Define(stage, "/asset")
        mesh = UsdGeom.Mesh.Define(stage, "/asset/mesh")
        mesh.GetPointsAttr().Set(verts)
        mesh.GetFaceVertexIndicesAttr().Set(faces)
        stage.Save()
        
        return "asset.usd"

# 사용
robot_camera_image = np.random.rand(480, 640, 3).astype(np.float32)
asset = robot_sim_asset_from_capture(robot_camera_image, sim_engine='habitat')
print(f"Generated asset: {len(asset['vertices'])} vertices, "
      f"{len(asset['faces'])} faces")
```

### 실험 3 — 4D Video Reconstruction (Frame-wise NeRF)

```python
class TemporalNeRFReconstructor:
    """
    Video frames → 4D NeRF (per-frame NeRF chains)
    """
    def __init__(self, n_frames=30):
        self.n_frames = n_frames
        self.nerfs = []  # List of per-frame NeRF models
        self.flow_fields = []  # Optional: optical flow between frames
    
    def reconstruct_video(self, video_frames, cameras):
        """
        video_frames: list of (H, W, 3) arrays
        cameras: list of camera intrinsics/extrinsics
        """
        for t, frame in enumerate(video_frames):
            # Train NeRF for frame t (or use LRM single-frame)
            nerf_t = self.train_nerf_frame(frame, cameras[t])
            self.nerfs.append(nerf_t)
            
            # Optional: estimate flow to previous frame
            if t > 0:
                flow = self.estimate_optical_flow(
                    video_frames[t-1], frame
                )
                self.flow_fields.append(flow)
        
        return self
    
    def train_nerf_frame(self, image, camera, n_steps=100):
        """Single frame 에 대한 빠른 NeRF training (mock)"""
        # In practice: LRM 으로 이미 triplane 있음, 여기선 implicit training
        nerf = nn.Sequential(
            nn.Linear(3 + 8*3*2, 256),  # pos + encoded pos
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # σ, c_r, c_g, c_b
        )
        
        # Mock training
        return nerf
    
    def estimate_optical_flow(self, frame1, frame2):
        """연속된 frame 간 optical flow (mock)"""
        flow = np.random.randn(frame1.shape[0], frame1.shape[1], 2) * 0.5
        return flow
    
    def render_4d(self, time_t, camera, H=512, W=512):
        """시간 t 에서의 뷰 렌더링"""
        # Bilinear interpolation between NeRF(t) and NeRF(t+1)
        # + optical flow warp
        
        nerf_t = self.nerfs[int(time_t)]
        # ... volume rendering ...
        rgb = torch.ones(H, W, 3) * 0.5  # mock output
        return rgb

# 사용
video_frames = [np.random.rand(256, 256, 3) for _ in range(30)]
cameras = [np.eye(4)] * 30

reconstructor = TemporalNeRFReconstructor(n_frames=30)
reconstructor.reconstruct_video(video_frames, cameras)

# Render 4D at t=15.5 (interpolated between frames 15 and 16)
output = reconstructor.render_4d(time_t=15.5, camera=np.eye(4))
print(f"4D render: {output.shape}")  # (512, 512, 3)
```

### 실험 4 — Autonomous Driving Perception

```python
class AutonomousDrivingPerception:
    """
    Multi-camera + LiDAR → 3D scene understanding
    """
    def __init__(self, n_cameras=8):
        self.n_cameras = n_cameras
        self.dust3r = SimpleDUSt3RMatcher().cuda()
        self.scene_graph = None
    
    def process_frame(self, camera_images, lidar_points, ego_pose):
        """
        camera_images: dict {cam_id: (H, W, 3)}
        lidar_points: (N_points, 3)
        ego_pose: (4, 4) vehicle pose in world frame
        """
        
        # Step 1: Multi-view 3D reconstruction (DUSt3R on camera pairs)
        pointclouds = []
        for cam_id_1, cam_id_2 in self.get_adjacent_cameras():
            img1 = camera_images[cam_id_1]
            img2 = camera_images[cam_id_2]
            
            feats1 = self.dino_encode(img1)
            feats2 = self.dino_encode(img2)
            
            pointmap, confidence, _ = self.dust3r(
                feats1.unsqueeze(0), feats2.unsqueeze(0)
            )
            
            valid = confidence[0] > 0.5
            pointclouds.append(pointmap[0][valid.squeeze()].cpu().numpy())
        
        # Merge camera + LiDAR pointclouds
        camera_pts = np.vstack(pointclouds)
        all_pts = np.vstack([camera_pts, lidar_points])
        
        # Step 2: Scene understanding (detection, tracking, prediction)
        detected_objects = self.detect_objects(all_pts)  # mock: clustering + classification
        
        # Step 3: Generate scene graph
        self.scene_graph = {
            'agents': detected_objects,
            'ego_pose': ego_pose,
            'confidence': confidence.mean().item(),
        }
        
        return self.scene_graph
    
    def plan_trajectory(self):
        """주어진 scene 에서 trajectory planning"""
        if self.scene_graph is None:
            return None
        
        # Behavior planning: risk assessment, goal selection
        # Motion planning: path, velocity profile
        trajectory = self.compute_safe_path(self.scene_graph)
        
        return trajectory
    
    def get_adjacent_cameras(self):
        """인접한 카메라 쌍 반환"""
        pairs = [(i, (i+1) % self.n_cameras) for i in range(self.n_cameras)]
        return pairs
    
    def dino_encode(self, image):
        """Mock DINO feature extraction"""
        h, w = image.shape[:2]
        return torch.randn(h // 14, w // 14, 256)
    
    def detect_objects(self, pointcloud):
        """3D clustering + classification (mock)"""
        from sklearn.cluster import DBSCAN
        
        clustering = DBSCAN(eps=0.5, min_samples=10).fit(pointcloud)
        labels = clustering.labels_
        
        objects = []
        for label in set(labels):
            if label == -1:  # noise
                continue
            cluster = pointcloud[labels == label]
            center = cluster.mean(axis=0)
            size = cluster.std(axis=0) * 2
            objects.append({'center': center, 'size': size})
        
        return objects
    
    def compute_safe_path(self, scene_graph):
        """Safe path computation (mock)"""
        # In practice: frenet-frame trajectory generation + collision avoidance
        return {'waypoints': []}

# 사용
perception = AutonomousDrivingPerception(n_cameras=8)

# Simulate sensor data
cameras = {i: np.random.rand(480, 640, 3) for i in range(8)}
lidar = np.random.randn(100000, 3)
ego_pose = np.eye(4)

scene = perception.process_frame(cameras, lidar, ego_pose)
traj = perception.plan_trajectory()

print(f"Detected {len(scene['agents'])} objects")
print(f"Ego confidence: {scene['confidence']:.3f}")
```

---

## 🔗 실전 활용

### 1. Vision Pro Spatial Video Pipeline

```
[Live capture: LiDAR 10fps + wide/stereo cameras 60fps]
    ↓
[DUSt3R on adjacent frame pairs]
    ↓
[NeRF or GS scene asset generation]
    ↓
[Real-time rendering + user interaction]
    ↓
[Spatial video file (H265 + 3D metadata)]
```

**Framework**: ARKit 의 frame stream → PyTorch inference (NN Compress) → Metal renderer.

### 2. Habitat Embodied AI Training

```
NeRF asset generation:
  Step 1: Scan room → multi-view images
  Step 2: LRM or NeRF training (optional)
  Step 3: Habitat importer → GLB/GLTF

Policy training:
  Robot task (e.g., "pick up the cup") 를 여러 diverse scenes 에서 train
  → sim-to-real transfer learning 개선
```

**Benefit**: Photorealistic rendering (OpenGL > flat textures) 으로 visual feature learning 강화.

### 3. 자율주행 Sensor Fusion

```
Synchronize 8-camera + LiDAR → ego frame
    ↓
Dense 3D from DUSt3R (vision), explicit 3D from LiDAR
    ↓
Confidence fusion: high-confidence regions from vision,
                    coverage from LiDAR
    ↓
BEV (bird's eye view) scene representation
    ↓
Planning module (learned or classical)
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 해결 |
|------|-----------|
| **Real-time processing** | Keyframe selection, sparse pointcloud 필수. Dense 는 bottleneck |
| **Good lighting (spatial computing)** | Low-light AR 에서 feature 추출 실패. Thermal/IR input 필요 |
| **Static scene (robotics sim)** | Dynamic object (moving robot, articulated hand) 는 별도 modeling 필요 |
| **Structured environment (driving)** | Extreme weather (heavy rain, snow) 에서 LiDAR + vision 모두 degraded |
| **Single-viewpoint generalization** | Out-of-distribution 에서 3D prior weak. Domain adaptation 필수 |
| **Metric scale ambiguity** | Uncalibrated 에서는 affine 모호. Calibration ruler or LiDAR 필수 |
| **Temporal coherence (4D)** | Frame간 large motion 으로 correspondence fail. Optical flow regularization 필요 |

---

## 📌 핵심 정리

$$
\boxed{\text{3D Foundation Models} \xrightarrow{\text{Speed, Scale, Diversity}} \text{Spatial Computing, Robotics, Autonomous Driving}}
$$

### Application Landscape

| 분야 | 입력 | LRM/GS | DUSt3R | 이점 | 도전 |
|------|------|--------|--------|------|------|
| **AR/VR** | Single/stereo | ✓ (LRM) | ✓ (real-time) | 5초 capture | Occlusion |
| **Robotics** | Mobile camera | ✓ (asset) | ✓ (mapping) | Sim training | Physics sim |
| **Driving** | Multi-camera | - | ✓✓ (dense) | Dense depth | Weather |
| **4D Video** | Video frames | ✓ (per-frame) | - (temporal?) | Novel view | Occlusion |
| **Digital Twin** | Scan + LiDAR | ✓ (detail) | ✓ (structure) | Hybrid | Scale |

### Frontier (2025)

1. **World Models + 3D**: Sora-like video diffusion 이 implicit 3D 학습 → explicit 3D decoding
2. **Continuous Foundation Model**: Tri-modal (image, 3D, video) 통합 training
3. **On-device Inference**: Quantized LRM/DUSt3R → mobile AR (not just cloud)
4. **Nerf-to-Mesh**: Efficient, editable mesh export → traditional 3D pipeline 호환
5. **Uncertainty Quantification**: Confidence prediction → safety-critical 응용

---

## 🤔 생각해볼 문제

**문제 1** (기초): LRM 이 단일 이미지에서 3D 를 출력할 수 있다면, "같은 모양이지만 다른 크기" (예: 작은 컵, 큰 컵) 를 구별하는 방법은?

<details>
<summary>해설</summary>

**문제의 핵심**: LRM 의 triplane 은 "절대적 스케일" 을 알 수 없음 (affine ambiguity).

**해결책**:

1. **Multi-view**: 여러 각도 → DUSt3R 으로 상대 스케일 복구
2. **Contextual**: 이미지에 알려진 물체 (손, ruler) 포함 → scale 추론
3. **Category-specific prior**: "이것은 컵 category" → 평균 크기 적용
4. **Metric from depth sensor**: LiDAR/ToF → 절대 스케일 고정

**실무**: Mobile AR 에서는 gesture (pinch) 로 user 가 명시적으로 scale 조정.

$\square$

</details>

**문제 2** (심화): Robotics 에서 NeRF asset 으로 training 한 policy 가 real robot 으로 직접 전이되려면, 어떤 "reality gap" 이 최소화되어야 하는가?

<details>
<summary>해설</summary>

**Sim-to-Real Gap 의 3가지 source**:

1. **Visual gap**: NeRF rendering (photorealistic) vs real camera (noise, compression, artifacts)
   - **해결**: Domain randomization (lighting, texture variation in NeRF)
   - **더 나은**: Real-world image augmentation during training

2. **Geometry gap**: NeRF 의 estimated 3D vs 실제 물리 geometry
   - **해결**: DUSt3R dense points → precise geometry
   - **한계**: small detail (sharp edge, friction) 는 reconstruction miss

3. **Dynamic gap**: 시뮬레이션의 dynamics (gravity, friction coefficient) vs 현실
   - **해결**: Learned dynamics model (추가 학습) 또는 domain randomization
   - **독립적 문제**: NeRF/GS 와 직접 연관 아님 (policy 자체 문제)

**결론**: NeRF asset 만으로는 sim-to-real gap 을 완전히 없애지 못함. "Visually rich training" 은 필요하지만, dynamics 는 별도 대응 필요. Embodied AI 는 multi-modal training (vision + proprio) 필수.

$\square$

</details>

**문제 3** (논문 비평): 자율주행에서 DUSt3R 의 dense reconstruction 이 implicit depth prediction (monocular 학습) 을 대체할 수 있는가? 어떤 상황에서 여전히 학습 기반 depth 가 필요한가?

<details>
<summary>해설</summary>

**DUSt3R 의 장점**:
- Explicit, dense, uncalibrated
- RANSAC-free, fast
- Multiple camera 에 쉽게 확장

**그래도 학습 기반 depth 가 필요한 경우**:

1. **Monocular (single camera)**: DUSt3R 은 stereo pair 필요 → monocular 는 "자기 자신과의 epipolar geometry" 불가능

2. **Fast motion**: Consecutive frame 간 baseline 이 너무 크거나, 극도로 빠른 움직임 → correspondence fail

3. **Transparent/reflective**: Glass, water 는 reflection 때문에 dense match 실패. Learned depth 모델이 implicit context 활용

4. **Long-range depth**: DUSt3R 은 near-field 에 최적 (5-50m 정도). Far-field (수백 m) 는 disparity 너무 작음 → monocular depth map (learned) 이 유리

**하이브리드 접근**:
```
DUSt3R (stereo pair) + learned monocular depth (single frame) 의 fusion
→ reliability map (어느 source 를 신뢰할 것인가)
```

자율주행 production: 둘 다 사용. DUSt3R 은 "일반적인 scene", monocular learned depth 는 "edge case" backup.

$\square$

</details>

---

<div align="center">

[◀ 이전](./02-dust3r-mast3r.md) | [📚 README](../README.md) | [다음](../README.md)

</div>
