import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
import numpy as np
import pyrender
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
import sys
import yaml
import hydra
from hydra.core.global_hydra import GlobalHydra


# --- Import SAM2 ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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

def extract_clip_features(images, processor, model, device):
    """Extracts CLIP features from a list of PIL Images."""
    feature_list = []
    with torch.no_grad():
        for image in images:
            rgb_image = image.convert("RGB")
            inputs = processor(images=rgb_image, return_tensors="pt").to(device)
            # Get the pooled image features
            features = model.get_image_features(**inputs)
            feature_list.append(features.cpu().numpy())
    return np.vstack(feature_list)



# --- NEW: Function to load the SAM2 model ---
def load_sam2_model(config_path, checkpoint_path):
    """Loads the SAM2 model and predictor from a config and checkpoint."""
    print("Loading SAM2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model configuration
    # with open(config_path, 'r') as f:
        # model_cfg = yaml.safe_load(f)

    config_dir = os.path.abspath(os.path.dirname(config_path))
    config_name = os.path.basename(config_path)
    GlobalHydra.instance().clear()
    # Use hydra's context manager to temporarily set the config search path
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        # Now, build_sam2 will find the config file in the directory we specified.
        # We pass only the filename to build_sam2.
        sam2_model = build_sam2(config_name, checkpoint_path).to(device)
    
    # Create the predictor
    predictor = SAM2ImagePredictor(sam2_model)
    
    print(f"✅ SAM2 model loaded on {device}.")
    return predictor, device



def load_camera_positions(directory, num_rigs=10):
    """
    Loads camera positions from .obj files, structuring them by rig.
    This is required for the test.py script.
    """
    camera_rigs = []
    print(f"Loading {num_rigs} camera rigs from '{directory}'...")
    for i in range(num_rigs):
        cam_file = os.path.join(directory, f'12_{i}.obj')
        if os.path.exists(cam_file):
            # Each file contains vertices for one camera rig. We take the first 10.
            cam_mesh = trimesh.load(cam_file, force='mesh')
            camera_rigs.append(cam_mesh.vertices[:10])
    if not camera_rigs:
        print(f"❌ Error: No camera rigs found in '{directory}'.")
        sys.exit(1)
    print(f"✅ Loaded {len(camera_rigs)} camera rigs, each with {camera_rigs[0].shape[0]} views.")
    return camera_rigs

def create_view_matrix(camera_position, target, up_vector, distance=1.2):
    """
    Creates a view matrix (world-to-camera).
    This matrix positions the camera to look at the target from a specific point.
    """
    camera_position = camera_position * distance
    forward = target - camera_position
    if np.linalg.norm(forward) < 1e-6: return np.eye(4)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_vector)
    if np.linalg.norm(right) < 1e-6:
        if abs(forward[1]) > 0.99:
            right = np.cross(forward, [0, 0, 1])
        else:
            right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)
    matrix = np.eye(4)
    matrix[0,:3], matrix[1,:3], matrix[2,:3] = right, new_up, -forward
    matrix[:3, 3] = -np.dot(matrix[:3,:3], camera_position)
    return matrix

def setup_scene(mesh):
    """
    Creates a pyrender scene with a transparent background, neutral material,
    and strong lighting, inspired by the Blender script.
    """
    # 1. Define a neutral, off-white material since the GLB has no material info.
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.8, 0.8, 0.8, 1.0], # Light grey color
        metallicFactor=0.1,
        roughnessFactor=0.7
    )
    render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    
    # 2. Set up the scene with a transparent background.
    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0], # Transparent RGBA
        ambient_light=[0.3, 0.3, 0.3]
    )
    scene.add(render_mesh)
    
    # 3. Add a strong key light from above, similar to Blender's AREA light.
    key_light = pyrender.PointLight(color=np.ones(3), intensity=10.0)
    light_pose = np.eye(4)
    light_pose[2, 3] = 4.0 # Position light above the object
    scene.add(key_light, pose=light_pose, name='light')

    # 4. Use an orthographic camera to match the LFD paper's methodology.
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.01, zfar=100.0)
    scene.add(camera, name='camera')

    return scene


def extract_dino_features(images, processor, model, device):
    """
    Extracts DinoV2 features from a list of PIL Images.
    """
    feature_list = []
    with torch.no_grad():
        for image in images:
            # Convert RGBA image to RGB before processing
            rgb_image = image.convert("RGB")
            inputs = processor(images=rgb_image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0].detach().cpu().numpy()
            feature_list.append(features)
    return np.vstack(feature_list)

# --- NEW: Function to extract features using SAM2 ---
def extract_sam2_features(images, predictor, device):
    """
    Extracts SAM2 features from a list of PIL Images.
    """
    feature_list = []
    print("Extracting features with SAM2...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for image in images:
            # 1. Set the image in the predictor
            # The image must be converted to a numpy array of shape (H, W, 3)
            rgb_image = image.convert("RGB")
            predictor.set_image(np.array(rgb_image))

            # 2. Access the feature maps
            # We use the main 'image_embed' feature map
            feature_map = predictor._features["image_embed"]

            # 3. Convert the feature map to a single vector using Global Average Pooling
            # The feature_map shape is (1, C, H, W). We average over H and W.
            feature_vector = feature_map.mean(dim=[-1, -2]).squeeze()
            
            # 4. Convert to numpy and append
            feature_list.append(feature_vector.cpu().numpy())
            
    return np.vstack(feature_list)

def render_and_capture_views(mesh, camera_positions, renderer, distance=1.2):
    """
    Renders a mesh from multiple viewpoints and returns in-memory PIL images.
    This version does not save files, it just returns the image objects.
    """
    scene = setup_scene(mesh)
    camera_node = next(iter(scene.get_nodes(name='camera')))
    light_node  = next(iter(scene.get_nodes(name='light')))
    target, up = np.array([0, 0, 0]), np.array([0, 1, 0])
    captured_images = []
    for pos in camera_positions:
        view_matrix = create_view_matrix(pos, target, up, distance)
        cam_pose = np.linalg.inv(view_matrix)
        scene.set_pose(camera_node, cam_pose)
        scene.set_pose(light_node,  cam_pose)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        captured_images.append(Image.fromarray(color, 'RGBA'))
    return captured_images

# --- Main Generation Function (MODIFIED) ---
def generate_features(mesh, camera_rigs, renderer, model_type='dinov2', model_assets=None):
    """
    Generates a structured set of features using the specified model type.
    """
    if model_assets is None:
        raise ValueError("model_assets must be provided (the loaded model objects).")

    # --- Normalise Mesh ---
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    radius = np.linalg.norm(mesh.vertices, axis=1).max()
    if radius > 1e-6:
        mesh.apply_scale(1.0 / radius)

    all_descriptors = []
    for rig_cameras in tqdm(camera_rigs, desc=f"Encoding with {model_type}"):
        rendered_images = render_and_capture_views(mesh, rig_cameras, renderer)
        
        # --- NEW: Switch between feature extractors ---
        if model_type == 'dinov2':
            processor, model, device = model_assets
            descriptor_features = extract_dino_features(rendered_images, processor, model, device)
        elif model_type == 'sam2':
            predictor, device = model_assets
            descriptor_features = extract_sam2_features(rendered_images, predictor, device)
        elif model_type == 'clip':
            processor, model, device = model_assets
            descriptor_features = extract_clip_features(rendered_images, processor, model, device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        all_descriptors.append(descriptor_features)
            
    return np.array(all_descriptors)

def save_features(features, output_file):
    """
    Saves the features to a file. Required by test.py.
    """
    print(f"Saving features to '{output_file}'...")
    np.save(output_file, features)
    print(f"✅ Features saved with shape {features.shape}.")

if __name__ == '__main__':
    # This main block is for simple rendering and saving of single feature sets,
    # not the full LFD generation which is handled by test.py
    # --- Configuration ---
    MODEL_PATHS = ['assets/model1.glb', 'assets/model2.glb']
    CAMERA_DATA_DIR = 'data'
    MAIN_OUTPUT_DIR = 'assets/render_images'
    # FEATURES_DIR = 'features'
    
    os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)
    # os.makedirs(FEATURES_DIR, exist_ok=True)

    # --- Setup ---
    cam_file = os.path.join(CAMERA_DATA_DIR, '12_0.obj')
    if not os.path.exists(cam_file):
        print(f"❌ Error: Camera file not found at '{cam_file}'")
        sys.exit(1)
        
    camera_positions = trimesh.load(cam_file, force='mesh').vertices[:10]
    renderer = pyrender.OffscreenRenderer(256, 256)
    # processor, model, device = load_dinov2_model()

    # --- Process Each Model ---
    for model_path in MODEL_PATHS:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_view_dir = os.path.join(MAIN_OUTPUT_DIR, f'{model_name}_views')
        os.makedirs(output_view_dir, exist_ok=True)
        
        print(f"\n--- Processing {model_path} ---")
        mesh = trimesh.load(model_path, force='mesh')
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        radius = np.linalg.norm(mesh.vertices, axis=1).max()
        mesh.apply_scale(1.0 / radius)

        # 1. Render views and save them to disk
        images = render_and_capture_views(mesh, camera_positions, renderer)
        print(f"Saving {len(images)} rendered views to '{output_view_dir}'...")
        for i, img in enumerate(images):
            img.save(os.path.join(output_view_dir, f"view_{i:02d}.png"))
        
        # # 2. Extract features from the in-memory images
        # features = extract_features(images, processor, model, device)
        
        # # 3. Save the features
        # feature_filename = os.path.join(FEATURES_DIR, f'{model_name}_features.npy')
        # np.save(feature_filename, features)
        # print(f"✅ Single feature set for {model_name} saved to {feature_filename} with shape {features.shape}")

    # --- Cleanup ---
    renderer.delete()
    print("\nDone.")
