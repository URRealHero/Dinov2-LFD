import trimesh
import numpy as np
import os
import pyrender
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

# ----------------------------------------------------------------------
# NEW FUNCTION to load the DinoV2 model
# ----------------------------------------------------------------------
def load_dinov2_model():
    """Loads the DinoV2 model and image processor from Hugging Face."""
    print("🦖 Loading DinoV2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    
    print("✅ DinoV2 model loaded.")
    return processor, model, device

# ----------------------------------------------------------------------
# NEW HELPER FUNCTION for creating the camera view matrix
# ----------------------------------------------------------------------
def create_view_matrix(camera_position, target, up_vector):
    """Creates a view matrix, equivalent to gluLookAt."""
    forward = target - camera_position
    forward /= np.linalg.norm(forward)
    
    right = np.cross(forward, up_vector)
    # Handle cases where forward and up are parallel
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, [0, 0, 1]) if abs(forward[2]) < 0.99 else np.cross(forward, [1, 0, 0])
    right /= np.linalg.norm(right)
    
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)

    matrix = np.eye(4)
    matrix[0,:3] = right
    matrix[1,:3] = new_up
    matrix[2,:3] = -forward
    matrix[:3, 3] = -np.dot(matrix[:3,:3], camera_position)
    return matrix

# --- (Your existing functions: load_model, load_camera_positions, setup_scene) ---

def load_model(filepath):
    """Loads a 3D model from a file and normalizes it."""
    print(f"🔄 Loading model from: {filepath}")
    # It's good practice to check if the file exists
    if not os.path.exists(filepath):
        print(f"❌ ERROR: Model file not found at {filepath}")
        return None
    
    mesh = trimesh.load(filepath, force='mesh')

    # Center the model at the origin
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)

    # Scale the model to fit within a unit sphere
    max_extent = np.max(np.linalg.norm(mesh.vertices, axis=1))
    if max_extent == 0: # Avoid division by zero for empty meshes
        return mesh
    mesh.apply_scale(1 / max_extent)
    
    print(f"✅ Model loaded successfully. It has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
    return mesh

def load_camera_positions(directory, num_rigs=10):
    """Loads all camera positions from a series of camera rig .obj files."""
    all_camera_positions = []
    print(f"🔄 Loading camera positions from {num_rigs} rig files in: {directory}")
    
    # In the C code, the outer loop is ANGLE. We will assume ANGLE=10 for now.
    for i in range(num_rigs):
        cam_file = os.path.join(directory, f'12_{i}.obj')
        if os.path.exists(cam_file):
            # Load the mesh, which is just a collection of vertices for these files.
            cam_mesh = trimesh.load(cam_file, force='mesh')
            # Add all vertices from this file to our list.
            all_camera_positions.extend(cam_mesh.vertices)
        else:
            print(f"⚠️ Warning: Camera rig file not found: {cam_file}")
            
    print(f"✅ Loaded a total of {len(all_camera_positions)} camera positions.")
    return np.array(all_camera_positions)

            
def setup_scene(mesh):
    """Creates a pyrender scene with the model and an orthographic camera."""
    print("🔧 Setting up the rendering scene...")
    
    # 1. Create a pyrender.Mesh from the trimesh.Trimesh object
    render_mesh = pyrender.Mesh.from_trimesh(mesh)
    
    # 2. Set up the scene with a white background, matching glClearColor
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])
    
    # 3. Add the mesh to the scene
    scene.add(render_mesh)
    
    # 4. Define the orthographic camera based on glOrtho(-1, 1, -1, 1, 0.0, 2.0)
    # The parameters for pyrender are xmag, ymag, znear, zfar.
    # xmag/ymag are half the width/height of the view volume.
    # znear/zfar correspond to the last two args of glOrtho.
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.0, zfar=2.0)
    scene.add(camera, name='camera') # Add camera to the scene as well

    # The original C code does not explicitly define lights, relying on
    # default OpenGL lighting. We add a directional light for basic illumination.
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, name='light')

    print("✅ Scene setup complete.")
    return scene

# ----------------------------------------------------------------------
# NEW FUNCTION to extract features using DinoV2
# ----------------------------------------------------------------------
def extract_features(images, processor, model, device):
    """Extracts DinoV2 features for a list of images."""
    print(f"⚡️ Extracting features for {len(images)} images...")
    feature_list = []
    
    with torch.no_grad():
        for image in tqdm(images): # Using tqdm for a progress bar
            # Process the image and send it to the GPU/CPU
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            # Get the model's output
            outputs = model(**inputs)
            
            # The feature vector is the last hidden state of the [CLS] token
            # We detach it from the graph, move to CPU, and convert to numpy
            features = outputs.last_hidden_state[:, 0].detach().cpu().numpy()
            feature_list.append(features)
            
    # Stack all features into a single (N, 768) numpy array
    return np.vstack(feature_list)


if __name__ == '__main__':
    # --- Configuration ---
    MODEL_FILE = 'data/your_model.obj' # ❗️ Make sure this path is correct
    CAMERA_DIR = 'data'
    OUTPUT_DIR = 'features'
    RENDER_WIDTH, RENDER_HEIGHT = 256, 256

    # --- Load Data ---
    main_model = load_model(MODEL_FILE)
    if main_model:
        cameras = load_camera_positions(CAMERA_DIR)

        if cameras.size > 0:
            scene = setup_scene(main_model)
            
            # --- Render Silhouettes ---
            print("\n🔄 Starting off-screen rendering...")
            renderer = pyrender.OffscreenRenderer(RENDER_WIDTH, RENDER_HEIGHT)
            camera_node = scene.get_nodes(name='camera')[0]
            light_node = scene.get_nodes(name='light')[0]
            rendered_images = []
            target, up = [0, 0, 0], [0, 1, 0]

            for i, pos in enumerate(cameras):
                cam_pose = create_view_matrix(pos, target, up)
                scene.set_pose(camera_node, cam_pose)
                scene.set_pose(light_node, cam_pose)
                color, depth = renderer.render(scene)
                rendered_images.append(Image.fromarray(color))

            renderer.delete()
            print(f"✅ Rendered {len(rendered_images)} images in memory.")

            # --- Load DinoV2 Model ---
            processor, model, device = load_dinov2_model()

            # ------------------------------------------------------------------
            # NEW SECTION: Extract features and save the .npy file
            # ------------------------------------------------------------------
            model_features = extract_features(rendered_images, processor, model, device)
            
            # Create the output directory if it doesn't exist
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Define the output filename based on the input model's name
            base_name = os.path.splitext(os.path.basename(MODEL_FILE))[0]
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}_dinov2.npy")
            
            # Save the final numpy array
            np.save(output_path, model_features)
            
            print("\n--- 🎉 All Done! ---")
            print(f"Saved features to: {output_path}")
            print(f"Array shape: {model_features.shape}")