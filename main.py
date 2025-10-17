"""
SyntheticFogGenerator
---------------------
Lightweight fog generation using OpenCV and NumPy.
No PyTorch required — works instantly with simplified or hybrid depth estimation.

Author: Hamim Ibne Nasim
"""

# ============================================
# REQUIREMENTS
# ============================================
# pip install opencv-python numpy matplotlib

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================
# SIMPLE FOG GENERATOR
# ============================================

class SimpleFogGenerator:
    """
    Fog generation using simplified depth estimation.
    Fast and dependency-free implementation.
    """

    def estimate_depth_simple(self, image):
        """Estimate depth using edge and gradient information."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge-based depth
        edges = cv2.Canny(gray, 50, 150).astype(np.float32)
        depth_edges = cv2.GaussianBlur(edges, (31, 31), 0)

        # Gradient-based depth
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
        depth_gradient = cv2.GaussianBlur(gradient.astype(np.float32), (31, 31), 0)

        # Combine and normalize
        depth = (depth_edges + depth_gradient) / 2
        depth = np.max(depth) - depth
        depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
        return depth

    def add_fog(self, image, depth, beta=4.0):
        """Apply atmospheric scattering model to add fog."""
        img_norm = image.astype(np.float64) / 255.0
        A = np.array([0.85, 0.85, 0.85])  # atmospheric light
        transmission = np.exp(-beta * depth)
        transmission_3ch = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)
        foggy = img_norm * transmission_3ch + A * (1 - transmission_3ch)
        return np.clip(foggy * 255, 0, 255).astype(np.uint8)


# ============================================
# HYBRID DEPTH FOG GENERATOR
# ============================================

class HybridDepthFog:
    """
    Hybrid fog generator combining multiple depth cues
    for more realistic fog synthesis.
    """

    def estimate_depth_hybrid(self, image):
        """Estimate depth using edge, brightness, and position cues."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 30, 100)
        depth_edges = np.max(cv2.GaussianBlur(edges.astype(np.float32), (31, 31), 0)) - edges

        depth_bright = cv2.GaussianBlur(255 - gray.astype(np.float32), (15, 15), 0)
        y_coords = np.linspace(0, 1, h).reshape(h, 1)
        depth_position = np.tile(y_coords, (1, w)) * 255

        depth = (0.4 * depth_edges + 0.4 * depth_bright + 0.2 * depth_position)
        return cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

    def add_fog(self, image, beta=4.0):
        """Add fog using hybrid depth estimation."""
        depth = self.estimate_depth_hybrid(image)
        img_norm = image.astype(np.float64) / 255.0
        A = np.array([0.85, 0.85, 0.85])
        transmission = np.exp(-beta * depth)
        transmission_3ch = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)
        foggy = img_norm * transmission_3ch + A * (1 - transmission_3ch)
        return np.clip(foggy * 255, 0, 255).astype(np.uint8), depth


# ============================================
# DATASET PROCESSING FUNCTIONS
# ============================================

def generate_fog_main_to_dataset(
    main_folder,
    dataset_folder,
    beta_values=[2, 3, 4, 5, 7],
    method="simple"
):
    """
    Generate foggy images from main_folder to dataset_folder.
    method: "simple" or "hybrid"
    """
    os.makedirs(dataset_folder, exist_ok=True)
    input_path = Path(main_folder)
    image_files = [f for ext in ['*.jpg', '*.jpeg', '*.png'] for f in input_path.glob(ext)]

    if not image_files:
        print(f"No images found in {main_folder}")
        return

    print(f"Processing {len(image_files)} images using {method} method...")

    fogger = HybridDepthFog() if method == "hybrid" else SimpleFogGenerator()

    for idx, img_file in enumerate(image_files, 1):
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"[{idx}] Skipped unreadable file: {img_file.name}")
            continue

        print(f"[{idx}/{len(image_files)}] {img_file.name}")
        depth = (
            fogger.estimate_depth_hybrid(image)
            if method == "hybrid"
            else fogger.estimate_depth_simple(image)
        )

        for beta in beta_values:
            foggy = (
                fogger.add_fog(image, beta)[0]
                if method == "hybrid"
                else fogger.add_fog(image, depth, beta)
            )
            output_name = f"{img_file.stem}_foggy_beta{beta}{img_file.suffix}"
            cv2.imwrite(str(Path(dataset_folder) / output_name), foggy)

    print(f"\n✓ Completed! Output saved in: {dataset_folder}")


# ============================================
# QUICK TEST FUNCTION
# ============================================

def quick_test_one_image(image_path, beta=4):
    """Run a quick test on a single image."""
    image = cv2.imread(image_path)
    fogger = HybridDepthFog()
    foggy, depth = fogger.add_fog(image, beta)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); axes[0].set_title('Original')
    axes[1].imshow(depth, cmap='plasma'); axes[1].set_title('Depth Map')
    axes[2].imshow(cv2.cvtColor(foggy, cv2.COLOR_BGR2RGB)); axes[2].set_title(f'Foggy (β={beta})')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show()

    cv2.imwrite('test_foggy.jpg', foggy)
    print("Saved as test_foggy.jpg")


# ============================================
# MAIN MESSAGE
# ============================================

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   SYNTHETIC FOG GENERATOR (NO PYTORCH REQUIRED)        ║
║   Lightweight OpenCV-based Image Fog Simulation        ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝

Usage Examples:
---------------
1. Process dataset:
   generate_fog_main_to_dataset("Main", "Dataset", method="hybrid")

2. Quick test:
   quick_test_one_image("sample.jpg", beta=4)
""")
