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
# The user specified this path.
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


# --- All Model Loading Functions ---

def load_sscd_model(model_path):
    """Loads a TorchScript model for SSCD."""
    print("Loading SSCD TorchScript model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_path).to(device)
    model.eval()
    print(f"✅ SSCD model loaded on {device}.")
    return (model, device)

def load_dinov1_model():
    """Loads the DINOv1 model and image processor from Hugging Face."""
    print("Loading DinoV1 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dino-vitb8")
    model = AutoModel.from_pretrained("facebook/dino-vitb8").to(device)
    print(f"✅ DinoV1 model loaded on {device}.")
    return (processor, model, device)

def load_dinov2_model():
    """Loads the DinoV2 model and image processor from Hugging Face."""
    print("Loading DinoV2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
    model = AutoModel.from_pretrained("facebook/dinov2-giant").to(device)
    print(f"✅ DinoV2 model loaded on {device}.")
    return processor, model, device

def load_clip_model():
    """Loads the CLIP model and processor from Hugging Face."""
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    print(f"✅ CLIP model loaded on {device}.")
    return (processor, model, device)

def load_sam2_model(config_path, checkpoint_path):
    """Loads the SAM2 model and predictor from a config and checkpoint."""
    if not SAM2_AVAILABLE:
        raise RuntimeError("SAM2 library is not installed. Cannot load SAM2 model.")
    print("Loading SAM2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_dir = os.path.abspath(os.path.dirname(config_path))
    config_name = os.path.basename(config_path)
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        sam2_model = build_sam2(config_name, checkpoint_path).to(device)

    predictor = SAM2ImagePredictor(sam2_model)
    print(f"✅ SAM2 model loaded on {device}.")
    return predictor, device


# --- Blender Rendering Functions ---

def generate_spherical_views(num_views=150, radius=2.5, fov_deg=50):
    """
    Generates a list of view dictionaries for Blender.
    """
    views = []
    phi = np.pi * (3. - np.sqrt(5.))

    for i in range(num_views):
        y = 1 - (i / float(num_views - 1)) * 2
        r = np.sqrt(1 - y * y)
        theta = phi * i
        pitch = np.arcsin(y)
        yaw = theta

        views.append({
            'yaw': yaw, 'pitch': pitch, 'radius': radius, 'fov': np.deg2rad(fov_deg)
        })
    print(f"✅ Generated {num_views} camera views for Blender.")
    return views

def render_with_blender(object_path, views, resolution=512, quality='FAST'):
    """
    Calls the Blender script as a subprocess to render views.
    It now captures output and only prints it if an error occurs.
    """
    if not os.path.exists(BLENDER_EXECUTABLE_PATH):
        _install_blender()

    with tempfile.TemporaryDirectory() as temp_dir:
        views_json = json.dumps(views)
        blender_script_path = os.path.join(os.path.dirname(__file__), 'render_blender.py')

        command = [
            'xvfb-run', '-a', # Use a virtual display for headless EEVEE
            BLENDER_EXECUTABLE_PATH, '--background', '--python', blender_script_path,
            '--', '--object', object_path, '--output_folder', temp_dir,
            '--views', views_json, '--resolution', str(resolution), '--quality', quality
        ]
        
        print(f"--- Calling Blender for {os.path.basename(object_path)} (Quality: {quality}) ---")
        
        # Run the process and wait for it to complete, capturing all output.
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check if the process failed.
        if result.returncode != 0:
            print(f"❌ BLENDER FAILED for {os.path.basename(object_path)}.")
            print("--- Blender's Full Output ---")
            # Print the combined stdout and stderr to show the error.
            print(result.stdout)
            print(result.stderr)
            print("--- End of Blender Output ---")
            raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout, stderr=result.stderr)
        
        # If successful, load the images.
        images = [Image.open(os.path.join(temp_dir, f'{i:03d}.png')) for i in range(len(views)) if os.path.exists(os.path.join(temp_dir, f'{i:03d}.png'))]
        
        if len(images) != len(views):
            print(f"⚠️ Warning: Expected {len(views)} rendered images, but found {len(images)}.")
            print("This might indicate a problem during rendering. Full Blender log:")
            print(result.stdout) # Print log if image count mismatches

        return images


# --- All Feature Extraction Functions ---

def extract_sscd_features(images, model, device):
    preprocess = pth_transforms.Compose([
        pth_transforms.Resize(288), pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    feature_list = []
    with torch.no_grad():
        for image in tqdm(images, desc="Extracting SSCD Features"):
            batch = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
            embedding = model(batch)[0, :]
            feature_list.append(embedding.cpu().numpy())
    return np.vstack(feature_list)

def extract_dinov1_features(images, processor, model, device):
    feature_list = []
    batch_size = 16
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting DINOv1 Features"):
            batch_images = [img.convert("RGB") for img in images[i:i+batch_size]]
            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            cls_token = last_hidden_state[:, 0, :]
            patch_tokens = last_hidden_state[:, 1:, :]
            patch_size = model.config.patch_size
            patch_h = inputs.pixel_values.shape[2] // patch_size
            patch_w = inputs.pixel_values.shape[3] // patch_size
            b, _, d = patch_tokens.shape
            patch_tokens_grid = patch_tokens.reshape(b, patch_h, patch_w, d).permute(0, 3, 1, 2)
            gem_pooled = nn.functional.avg_pool2d(patch_tokens_grid.clamp(min=1e-6).pow(4), (patch_h, patch_w)).pow(1./4)
            gem_pooled = gem_pooled.reshape(b, -1)
            final_feature = torch.cat((cls_token, gem_pooled), dim=1)
            feature_list.append(final_feature.cpu().numpy())
    return np.vstack(feature_list)

def extract_dinov2_features(images, processor, model, device):
    feature_list = []
    batch_size = 16
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting DINOv2 Features"):
            batch_images = [img.convert("RGB") for img in images[i:i+batch_size]]
            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            cls_tokens = outputs.last_hidden_state[:, 0, :]
            feature_list.append(cls_tokens.cpu().numpy())
    return np.vstack(feature_list)

def extract_clip_features(images, processor, model, device):
    feature_list = []
    batch_size = 16
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting CLIP Features"):
            batch_images = [img.convert("RGB") for img in images[i:i+batch_size]]
            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
            vision_outputs = model.vision_model(**inputs)
            cls_tokens = vision_outputs.last_hidden_state[:, 0, :]
            feature_list.append(cls_tokens.cpu().numpy())
    return np.vstack(feature_list)

def extract_sam2_features(images, predictor, device):
    if not SAM2_AVAILABLE:
        raise RuntimeError("SAM2 library is not installed. Cannot extract SAM2 features.")
    feature_list = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for image in tqdm(images, desc="Extracting SAM2 Features"):
            predictor.set_image(np.array(image.convert("RGB")))
            feature_map = predictor._features["image_embed"]
            feature_vector = feature_map.mean(dim=[-1, -2]).squeeze()
            feature_list.append(feature_vector.cpu().numpy())
    return np.vstack(feature_list)


# --- Main Descriptor Generation Function ---

def generate_descriptor(object_path, views, model_type, model_assets, quality='FAST'):
    """
    Generates a single descriptor (set of features) for a 3D model by
    rendering it with Blender and then extracting features.
    """
    if model_assets is None:
        raise ValueError("model_assets must be provided")

    # Render all views using Blender
    rendered_images = render_with_blender(object_path, views, quality=quality)

    # Dispatch to the correct feature extraction function
    if model_type == 'sscd':
        descriptor = extract_sscd_features(rendered_images, *model_assets)
    elif model_type == 'dinov1':
        descriptor = extract_dinov1_features(rendered_images, *model_assets)
    elif model_type == 'dinov2':
        descriptor = extract_dinov2_features(rendered_images, *model_assets)
    elif model_type == 'sam2':
        descriptor = extract_sam2_features(rendered_images, *model_assets)
    elif model_type == 'clip':
        descriptor = extract_clip_features(rendered_images, *model_assets)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return descriptor

# --- Main block for standalone testing ---

if __name__ == '__main__':
    # This main block is for testing the rendering of 150 views per model.
    # It does not extract features, only saves the rendered images to disk.

    # --- Configuration ---
    MODEL_PATHS = ['assets/model1.glb', 'assets/model2.glb']
    MAIN_OUTPUT_DIR = 'assets/render_images'
    NUM_VIEWS = 50
    RESOLUTION = 512

    os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)

    # --- Setup ---
    print("--- Initializing Rendering Test ---")
    views = generate_spherical_views(num_views=NUM_VIEWS)

    # --- Process Each Model ---
    for model_path in MODEL_PATHS:
        if not os.path.exists(model_path):
            print(f"⚠️ Warning: Model file not found at '{model_path}', skipping.")
            continue

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_view_dir = os.path.join(MAIN_OUTPUT_DIR, f'{model_name}_views_{NUM_VIEWS}')
        os.makedirs(output_view_dir, exist_ok=True)

        print(f"\n--- Processing {model_path} ---")

        try:
            # Render all views using the Blender pipeline
            images = render_with_blender(model_path, views, resolution=RESOLUTION)

            # Save the rendered images to disk for inspection
            print(f"Saving {len(images)} rendered views to '{output_view_dir}'...")
            for i, img in enumerate(images):
                img.save(os.path.join(output_view_dir, f"view_{i:03d}.png"))

            print(f"✅ Finished processing {model_name}.")

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\n❌ FAILED to process {model_name}. An error occurred during rendering.")
            # The detailed error is already printed by the render_with_blender function
            print("Skipping to the next model.")
            continue

    print("\nDone.")
    print(f"All rendered images saved to '{MAIN_OUTPUT_DIR}'.")