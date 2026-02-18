import numpy as np
import cv2
import os

# Create dummy or conditional import to bypass static analysis failure if package missing
try:
    from skimage.metrics import structural_similarity as ssim  # type: ignore
except ImportError:
    ssim = None

def compute_ssim(img1, img2):
    if ssim is None:
        return 0.0
    try:
        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray1, gray2, full=True)
        return score
    except Exception as e:
        print(f"SSIM Error: {e}")
        return 0.0

import lpips
import torch
import glob

# Initialize LPIPS model globally (optimization)
# Using alex net as it is faster and aligns well with human perception
loss_fn_alex = lpips.LPIPS(net='alex')

def calculate_metrics(comparison_dir):
    """
    Iterates through subfolders in comparisons_dir.
    Expects structure:
      comparison_dir/
        <Item_Name>/
          gt.png   (Ground Truth)
          image.png (Generated Image)
    
    Calculates L1 (Pixel Difference), SSIM, and LPIPS.
    """
    
    # Traverse all subdirectories
    subdirs = [d for d in os.listdir(comparison_dir) if os.path.isdir(os.path.join(comparison_dir, d))]
    subdirs.sort()

    print(f"{'Item Name':<50} | {'L1 Diff':<10} | {'SSIM':<10} | {'LPIPS':<10}")
    print("-" * 95)

    l1_scores = []
    ssim_scores = []
    lpips_scores = []

    for subdir in subdirs:
        subdir_path = os.path.join(comparison_dir, subdir)
        
        # File paths
        gt_path = os.path.join(subdir_path, "gt.png")
        img_path = os.path.join(subdir_path, "image.png")
        
        # Check if both files exist
        if not os.path.isfile(gt_path) or not os.path.isfile(img_path):
            print(f"Warning: Missing file in {subdir_path}, skipping...")
            continue
        
        # Read images
        gt = cv2.imread(gt_path)
        img = cv2.imread(img_path)
        
        if gt is None:
            print(f"Error: Could not read GT image at {gt_path}")
            return {}
        if img is None:
            print(f"Error: Could not read Generated image at {img_path}")
            return {}

        # Resize to match GT if necessary
        try:
            if gt.shape != img.shape:
                img = cv2.resize(img, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"Error resizing images: {e}")
            return {}

        # Define metrics
        metrics = {}
        
        # 1. L1 Distance (Pixel-wise absolute difference)
        try:
            l1_diff = np.mean(np.abs(gt - img))
            metrics['l1_diff'] = float(l1_diff)
        except Exception:
             metrics['l1_diff'] = -1.0
        
        # 2. SSIM (Structural Similarity Index)
        metrics['ssim'] = compute_ssim(gt, img)

        # 3. LPIPS (Learned Perceptual Image Patch Similarity)
        try:
            # Normalize images to [0, 1] for LPIPS
            gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
            # Calculate LPIPS
            lpips_value = loss_fn_alex(gt_tensor.unsqueeze(0), img_tensor.unsqueeze(0)).item()
            metrics['lpips'] = float(lpips_value)
        except Exception:
            metrics['lpips'] = -1.0

        print(f"{subdir:<50} | {metrics['l1_diff']:<10.4f} | {metrics['ssim']:<10.4f} | {metrics['lpips']:<10.4f}")
        
        l1_scores.append(metrics['l1_diff'])
        ssim_scores.append(metrics['ssim'])
        lpips_scores.append(metrics['lpips'])

    print("-" * 95)
    print(f"{'AVERAGE':<50} | {np.mean(l1_scores):<10.4f} | {np.mean(ssim_scores):<10.4f} | {np.mean(lpips_scores):<10.4f}")

if __name__ == "__main__":
    # Use relative path for portability
    COMPARISON_DIR = os.path.join(os.path.dirname(__file__), "comparison")
    
    if os.path.exists(COMPARISON_DIR):
        calculate_metrics(COMPARISON_DIR)
    else:
        print(f"Directory not found: {COMPARISON_DIR}")
