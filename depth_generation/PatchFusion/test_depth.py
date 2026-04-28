import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from mmengine.config import Config
from estimator.models import build_model

# ===== STEP 1: Set your checkpoint paths =====
coarse_pretrain_path = '/Vaibhav/shivasish1/sam2/PatchFusion/pretrained_models/coarse_pretrain.pth'
fine_pretrain_path = '/Vaibhav/shivasish1/sam2/PatchFusion/pretrained_models/fine_pretrain.pth'
patchfusion_path = '/Vaibhav/shivasish1/sam2/PatchFusion/pretrained_models/pathfusion.pth'

# ===== STEP 2: Load config file =====
# This config file defines the model architecture
cfg_path = '/Vaibhav/shivasish1/sam2/PatchFusion/configs/patchfusion_depthanything/depthanything_vitl_patchfusion_u4k.py'
cfg = Config.fromfile(cfg_path)

# ===== STEP 3: Override config to use your local checkpoints =====
# This tells the model to load coarse and fine models from your paths
cfg.model.config.pretrain_model = [coarse_pretrain_path, fine_pretrain_path]

# ===== STEP 4: Build model =====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Build the model architecture
model = build_model(cfg.model)

# Load the patchfusion checkpoint (fusion network weights)
print("Loading patchfusion checkpoint...")
checkpoint = torch.load(patchfusion_path, map_location=DEVICE)
model.load_dict(checkpoint['model_state_dict'])

# Move model to device and set to evaluation mode
model = model.to(DEVICE).eval()
print("✓ Model loaded successfully from local checkpoints!")

# ===== STEP 5: Load and preprocess your image =====
image_path = './examples/example_1.jpeg'  # CHANGE THIS to your image path
print(f"Loading image: {image_path}")

image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Could not load image from {image_path}")

# Convert BGR to RGB and normalize to [0, 1]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
image = transforms.ToTensor()(np.asarray(image))  # Shape: [3, H, W]

# ===== STEP 6: Prepare image inputs =====
# Get model configuration
image_raw_shape = model.tile_cfg['image_raw_shape']  # Target resolution
image_resizer = model.resizer  # Resizer for low-res version

# Create low-resolution version (for global context)
image_lr = image_resizer(image.unsqueeze(dim=0)).float().to(DEVICE)

# Create high-resolution version (for patch processing)
image_hr = F.interpolate(
    image.unsqueeze(dim=0), 
    image_raw_shape, 
    mode='bicubic', 
    align_corners=True
).float().to(DEVICE)

# ===== STEP 7: Run inference =====
print("Running depth estimation...")
mode = 'r128'  # Inference mode: r32 (fast) | r64 (medium) | r128 (best quality)
process_num = 4  # Batch size for patches (reduce if GPU memory error)

with torch.no_grad():
    depth_prediction, _ = model(
        mode='infer', 
        cai_mode=mode, 
        process_num=process_num, 
        image_lr=image_lr, 
        image_hr=image_hr
    )

# ===== STEP 8: Post-process depth map =====
# Resize depth to original image size
depth = F.interpolate(
    depth_prediction, 
    image.shape[-2:]  # [height, width]
)[0, 0].detach().cpu().numpy()

# ===== STEP 9: Save outputs =====
# Save 16-bit depth map (multiply by 256 as per repo instructions)
depth_16bit = (depth * 256).astype(np.uint16)
cv2.imwrite('./depth_16bit.png', depth_16bit)
print("✓ Saved: depth_16bit.png")

# Save colored visualization
depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255
depth_colored = cv2.applyColorMap(
    depth_normalized.astype(np.uint8), 
    cv2.COLORMAP_INFERNO
)
cv2.imwrite('./depth_colored.png', depth_colored)
print("✓ Saved: depth_colored.png")

print("\n✓ Depth estimation completed successfully!")
print(f"  Depth map shape: {depth.shape}")
print(f"  Depth range: [{depth.min():.2f}, {depth.max():.2f}]")