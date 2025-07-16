import os
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

            inputs = processor(images=[img.convert("RGB") for img in img_batch], return_tensors="pt", padding=True).to(device)
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

def render_views_to_tempdir(object_path, views, quality='FAST', resolution=512):
    """
    NEW FUNCTION 1: Renders views and saves them to a persistent temporary directory.
    It returns the path to this directory. The caller is responsible for cleanup.
    """
    if not os.path.exists(BLENDER_EXECUTABLE_PATH):
        _install_blender()

    # Create a directory that will persist after this function returns.
    output_dir = tempfile.mkdtemp()
    
    views_json = json.dumps(views)
    blender_script_path = os.path.join(os.path.dirname(__file__), 'render_blender.py')
    
    command = [
        'xvfb-run', '-a',
        BLENDER_EXECUTABLE_PATH, '--background', '--python', blender_script_path,
        '--', '--object', object_path, '--output_folder', output_dir,
        '--views', views_json, '--resolution', str(resolution), '--quality', quality
    ]
    
    print(f"--- Calling Blender for {os.path.basename(object_path)} (Output: {output_dir}) ---")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ BLENDER FAILED for {os.path.basename(object_path)}.")
        print("--- Blender's Full Output ---\n", result.stdout, result.stderr, "\n--- End of Blender Output ---")
        shutil.rmtree(output_dir) # Clean up on failure
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout, stderr=result.stderr)
    
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