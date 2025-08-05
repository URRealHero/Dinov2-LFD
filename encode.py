import os
os.environ['HF_HOME'] = '/scratch/gpfs/bz1474/sp2526/huggingface_cache'  # Set Hugging Face cache directory
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Use EGL for OpenGL context
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
import subprocess
import json
import tempfile
import shutil
import torch.nn as nn
from torchvision import transforms as pth_transforms
import yaml
import hydra
from hydra.core.global_hydra import GlobalHydra
import trimesh
import pyrender
import math

# --- NOTE: Set this to the path of your Blender executable ---
BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '.'
BLENDER_EXECUTABLE_PATH = 'blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_EXECUTABLE_PATH):
        os.system('sudo apt-get update -y')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')
        os.system(f'rm -rf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz')

# --- Import SAM2 ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("⚠️ Warning: SAM2 library not found. The 'sam2' model type will not be available.")


# --- All Model Loading Functions (Unchanged) ---

def load_sscd_model(model_path):
    print("Loading SSCD TorchScript model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_path).to(device)
    model.eval()
    print(f"✅ SSCD model loaded on {device}.")
    return (model, device)

def load_dinov1_model():
    print("Loading DinoV1 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dino-vitb8")
    model = AutoModel.from_pretrained("facebook/dino-vitb8").to(device)
    print(f"✅ DinoV1 model loaded on {device}.")
    return (processor, model, device)

def load_dinov2_model():
    print("Loading DinoV2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
    model = AutoModel.from_pretrained("facebook/dinov2-giant").to(device)
    print(f"✅ DinoV2 model loaded on {device}.")
    return processor, model, device

def load_clip_model():
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    print(f"✅ CLIP model loaded on {device}.")
    return (processor, model, device)

def load_sam2_model(config_path, checkpoint_path):
    if not SAM2_AVAILABLE:
        raise RuntimeError("SAM2 library is not installed. Cannot load SAM2 model.")
    print("Loading SAM2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_dir, config_name = os.path.dirname(config_path), os.path.basename(config_path)
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base=None):
        sam2_model = build_sam2(config_name, checkpoint_path).to(device)
    predictor = SAM2ImagePredictor(sam2_model)
    print(f"✅ SAM2 model loaded on {device}.")
    return predictor, device

# --- Helper Functions (Unchanged) ---
def generate_spherical_views(num_views=150, radius=2.5, fov_deg=50):
    views = []
    phi = np.pi * (3. - np.sqrt(5.))
    for i in range(num_views):
        y = 1 - (i / float(num_views - 1)) * 2
        r = np.sqrt(1 - y * y)
        theta = phi * i
        pitch, yaw = np.arcsin(y), theta
        views.append({'yaw': yaw, 'pitch': pitch, 'radius': radius, 'fov': np.deg2rad(fov_deg)})
    print(f"✅ Generated {num_views} camera views for Blender.")
    return views

# --- All Feature Extraction Functions (Unchanged) ---
# These functions now take a list of PIL Images as input
def extract_sscd_features(images, model, device):
    preprocess = pth_transforms.Compose([pth_transforms.Resize(288), pth_transforms.ToTensor(), pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    feature_list = []
    with torch.no_grad():
        for image in tqdm(images, desc="Extracting SSCD Features", leave=False):
            batch = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
            embedding = model(batch)[0, :]
            feature_list.append(embedding.cpu().numpy())
    return np.vstack(feature_list)

def extract_dinov1_features(images, processor, model, device):
    """Corrected version using list slicing for batching."""
    feature_list = []
    batch_size = 16  # Define a batch size
    with torch.no_grad():
        # REPLACED np.array_split with a standard python loop for batching
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting DINOv1 Features", leave=False):
            img_batch = images[i : i + batch_size]
            if not img_batch: continue
            
            inputs = processor(images=[img.convert("RGB") for img in img_batch], return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            cls_token, patch_tokens = outputs.last_hidden_state[:, 0, :], outputs.last_hidden_state[:, 1:, :]
            patch_size = model.config.patch_size
            patch_h, patch_w = inputs.pixel_values.shape[2] // patch_size, inputs.pixel_values.shape[3] // patch_size
            b, _, d = patch_tokens.shape
            patch_tokens_grid = patch_tokens.reshape(b, patch_h, patch_w, d).permute(0, 3, 1, 2)
            gem_pooled = nn.functional.avg_pool2d(patch_tokens_grid.clamp(min=1e-6).pow(4), (patch_h, patch_w)).pow(1./4).reshape(b, -1)
            feature_list.append(torch.cat((cls_token, gem_pooled), dim=1).cpu().numpy())
    return np.vstack(feature_list)

def extract_dinov2_features(images, processor, model, device):
    """Corrected version using list slicing for batching."""
    feature_list = []
    batch_size = 16  # Define a batch size
    with torch.no_grad():
        # REPLACED np.array_split with a standard python loop for batching
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting DINOv2 Features", leave=False):
            img_batch = images[i : i + batch_size]
            if not img_batch: continue

            inputs = processor(images=[img.convert("RGB") for img in img_batch], return_tensors="pt").to(device)
            cls_tokens = model(**inputs).last_hidden_state[:, 0, :]
            feature_list.append(cls_tokens.cpu().numpy())
    return np.vstack(feature_list)

def extract_clip_features(images, processor, model, device):
    """Corrected version using list slicing for batching."""
    feature_list = []
    batch_size = 16 # Define a batch size
    with torch.no_grad():
        # REPLACED np.array_split with a standard python loop for batching
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting CLIP Features", leave=False):
            img_batch = images[i : i + batch_size]
            if not img_batch: continue
            
            inputs = processor(images=[img.convert("RGB") for img in img_batch], return_tensors="pt", padding=True).to(device)
            cls_tokens = model.vision_model(**inputs).last_hidden_state[:, 0, :]
            feature_list.append(cls_tokens.cpu().numpy())
    return np.vstack(feature_list)

def extract_sam2_features(images, predictor, device):
    if not SAM2_AVAILABLE: raise RuntimeError("SAM2 library is not installed.")
    feature_list = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for image in tqdm(images, desc="Extracting SAM2 Features", leave=False):
            predictor.set_image(np.array(image.convert("RGB")))
            feature_vector = predictor._features["image_embed"].mean(dim=[-1, -2]).squeeze()
            feature_list.append(feature_vector.cpu().numpy())
    return np.vstack(feature_list)


# ------------------------------------------------------------------
# --- REFACTORED LOGIC: The two new functions for the split logic ---
# ------------------------------------------------------------------

def create_look_at_pose(eye, target, up):
    """
    Generates a 4x4 camera-to-world pose matrix.
    This implementation is a replacement for the missing trimesh.transformations.look_at
    """
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    # 1. Z-axis: The direction of the camera's gaze (from target to eye), normalized.
    # In a right-handed system, the camera looks along its -Z axis.
    z_axis = eye - target
    z_axis /= np.linalg.norm(z_axis)

    # 2. X-axis: The "right" vector, perpendicular to the up vector and z-axis.
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # 3. Y-axis: The "up" vector for the camera, perpendicular to z-axis and x-axis.
    y_axis = np.cross(z_axis, x_axis)

    # 4. Create the 4x4 pose matrix
    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye

    return pose


# --- UPDATED RENDERER FUNCTION ---
def render_views_to_tempdir(object_path, views, resolution=512, **kwargs):
    """
    Renders views of a 3D object using pyrender and saves them to a temporary directory.
    This function mimics the lighting and material setup from the provided three.js template.
    """
    output_dir = tempfile.mkdtemp()
    
    try:
        mesh = trimesh.load(object_path, force='mesh', process=False)
    except Exception as e:
        print(f"❌ Failed to load mesh {object_path}: {e}")
        shutil.rmtree(output_dir)
        return None

    # --- Normalize the mesh ---
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)
    max_dim = np.max(mesh.extents)
    if max_dim > 0:
        scale_factor = 1.5 / max_dim
        mesh.apply_scale(scale_factor)

    # --- Create the pyrender scene ---
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0xAA/255, 0xAA/255, 0xAA/255, 1.0],
        metallicFactor=0.1,
        roughnessFactor=0.7,
        doubleSided=True
    )
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene = pyrender.Scene(
        bg_color=[0xf4/255, 0xf4/255, 0xf4/255],
        ambient_light=[0.7, 0.7, 0.7]
    )
    scene.add(pyrender_mesh)

    light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.6 * np.pi * 2)
    light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.4 * np.pi * 2)

    # --- FIXED: Use our new 'create_look_at_pose' function ---
    light_pose1 = create_look_at_pose(eye=[5, 10, 7.5], target=[0, 0, 0], up=[0, 1, 0])
    light_pose2 = create_look_at_pose(eye=[-5, -5, -7.5], target=[0, 0, 0], up=[0, 1, 0])
    scene.add(light1, pose=light_pose1)
    scene.add(light2, pose=light_pose2)

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    
    for i, view in enumerate(tqdm(views, desc=f"Pyrendering {os.path.basename(object_path)}", leave=False)):
        radius = view['radius']
        x = radius * math.cos(view['pitch']) * math.sin(view['yaw'])
        y = radius * math.sin(view['pitch'])
        z = radius * math.cos(view['pitch']) * math.cos(view['yaw'])
        
        # --- FIXED: Use our new 'create_look_at_pose' function ---
        camera_pose = create_look_at_pose(eye=[x, y, z], target=[0, 0, 0], up=[0, 1, 0])
        
        camera = pyrender.PerspectiveCamera(yfov=view['fov'], aspectRatio=1.0)
        camera_node = scene.add(camera, pose=camera_pose)
        
        color, _ = renderer.render(scene)
        
        img = Image.fromarray(color)
        img.save(os.path.join(output_dir, f'view_{i:04d}.png'))
        
        scene.remove_node(camera_node)

    renderer.delete()
    return output_dir


def create_descriptor_from_images(image_dir, model_type, model_assets):
    """
    NEW FUNCTION 2: Loads rendered images from a directory and extracts features.
    """
    if model_assets is None:
        raise ValueError("model_assets must be provided")

    # Load images from the directory
    try:
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        images = [Image.open(os.path.join(image_dir, f)) for f in image_files]
        if not images:
            print(f"⚠️ No images found in {image_dir}, returning None.")
            return None
    except Exception as e:
        print(f"❌ Failed to load images from {image_dir}: {e}")
        return None

    # Dispatch to the correct feature extraction function
    if model_type == 'sscd':
        descriptor = extract_sscd_features(images, *model_assets)
    elif model_type == 'dinov1':
        descriptor = extract_dinov1_features(images, *model_assets)
    elif model_type == 'dinov2':
        descriptor = extract_dinov2_features(images, *model_assets)
    elif model_type == 'sam2':
        descriptor = extract_sam2_features(images, *model_assets)
    elif model_type == 'clip':
        descriptor = extract_clip_features(images, *model_assets)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return descriptor

# --- Original 'generate_descriptor' and 'render_with_blender' are now removed ---

def save_features(features, output_path):
    """ Saves the extracted features to a file. """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, features)
    print(f"✅ Features saved to {output_path}")