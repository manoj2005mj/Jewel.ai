import os
import torch
import numpy as np
import traceback
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running metrics on {DEVICE}...")

# Standard transforms for metrics
# FID and LPIPS expect normalized tensors (usually 0-1 or -1 to 1)
# Torchmetrics implementation details:
# - FID: expects uint8 [0, 255] tensor (N, C, H, W) OR float [0, 1] if normalize=True not set? 
#   Actually, torchmetrics FID update method: "The input images should be of type uint8 or float. If float, they are expected to be in [0, 1]."
#   "The images should be of shape (N, C, H, W)."
#
# - LPIPS: expects float tensors in range [-1, 1].

def load_and_preprocess_image(path, resize_dims=None):
    """
    Load an image and convert to tensor.
    Returns:
      - img_uint8: Tensor (C, H, W) in [0, 255] uint8 (for FID)
      - img_lpips: Tensor (1, C, H, W) in [-1, 1] float (for LPIPS)
    """
    try:
        img = Image.open(path).convert("RGB")
        if resize_dims:
            img = img.resize(resize_dims, Image.Resampling.LANCZOS)
            
        # Transform for FID (uint8 [0, 255])
        # We start with PIL image -> ToTensor gives float [0, 1]
        to_tensor = transforms.ToTensor()
        img_float = to_tensor(img) # C, H, W in [0, 1]
        
        # Convert to uint8 [0, 255] for FID (recommended for torchmetrics to avoid issues)
        img_uint8 = (img_float * 255).to(dtype=torch.uint8)
        
        # Transform for LPIPS (float [-1, 1])
        # [0, 1] -> [-1, 1] => x * 2 - 1
        img_lpips = (img_float * 2.0) - 1.0
        img_lpips = img_lpips.unsqueeze(0) # Add batch dimension: 1, C, H, W
        
        return img_uint8.to(DEVICE), img_lpips.to(DEVICE)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None

def calculate_metrics(comparison_root_dir):
    """
    Iterate through comparison folders, accumulate images, and calculate FID & LPIPS.
    
    Structure:
    comparison_root_dir/
       Folder1/
          gt.png
          image.png
       Folder2/
          gt.png
          image.png
       ...
    """
    
    # Initialize Metrics
    # feature=2048 is standard for FID
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    # net_type='alex' is standard for LPIPS, or 'vgg'
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(DEVICE)

    # Separate instance for per-image FID calculation
    fid_single = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    
    lpips_scores = []
    
    # Check root dir
    if not os.path.exists(comparison_root_dir):
        print(f"Directory not found: {comparison_root_dir}")
        return

    folders = [f for f in os.listdir(comparison_root_dir) if os.path.isdir(os.path.join(comparison_root_dir, f))]
    folders = sorted(folders)
    
    print(f"Found {len(folders)} comparison pairs.")
    
    count = 0
    
    # Buffers for batch processing if needed, but FID update takes batches of any size.
    # LPIPS needs pairs.
    
    for folder_name in folders:
        folder_path = os.path.join(comparison_root_dir, folder_name)
        
        # Identify files
        gt_path = os.path.join(folder_path, "gt.png")
        if not os.path.exists(gt_path): gt_path = os.path.join(folder_path, "gt.jpg")
        
        gen_path = os.path.join(folder_path, "image.png")
        if not os.path.exists(gen_path): gen_path = os.path.join(folder_path, "image.jpg")
        
        if os.path.exists(gt_path) and os.path.exists(gen_path):
            # Load images
            # Note: For FID, images should arguably be the same size if meaningful features are to be extracted, 
            # though Inception handles var sizes, resizing to standard (e.g. 299x299) is implicit in some implementations
            # or it handles spatial pooling.
            # To be safe and fair to LPIPS pixel-wise nature, we resize GT to Gen dimension like before.
            
            try:
                # Open just to check size
                with Image.open(gen_path) as gen_pil:
                     w, h = gen_pil.size
                     
                # Get tensors
                # Note: load_and_preprocess_image returns (uint8_tensor, lpips_tensor)
                # But looking at your load function:
                # return img_uint8.to(DEVICE), img_lpips.to(DEVICE)
                
                gt_uint8, gt_lpips_tensor = load_and_preprocess_image(gt_path, resize_dims=(w, h))
                gen_uint8, gen_lpips_tensor = load_and_preprocess_image(gen_path, resize_dims=(w, h))
                
                if gt_uint8 is None or gen_uint8 is None:
                    continue
                
                # --- UPDATE GLOBAL FID ---
                # update(imgs, real=True/False)
                # dims need to be (N, C, H, W)
                fid.update(gt_uint8.unsqueeze(0), real=True)
                fid.update(gen_uint8.unsqueeze(0), real=False)
                
                # --- CALCULATE LPIPS (Pairwise) ---
                # forward(img1, img2)
                score = lpips_metric(gt_lpips_tensor, gen_lpips_tensor)
                lpips_value = score.item()
                lpips_scores.append(lpips_value)
                
                # --- CALCULATE PER-IMAGE FID ---
                # FID normally requires a distribution (min 2 images).
                # Workaround: Duplicate images to form a batch of 2.
                try:
                    fid_single.reset()
                    # Stack to make batch size 2
                    gt_batch = torch.stack([gt_uint8, gt_uint8])
                    gen_batch = torch.stack([gen_uint8, gen_uint8])
                    
                    fid_single.update(gt_batch, real=True)
                    fid_single.update(gen_batch, real=False)
                    
                    single_fid = fid_single.compute()
                    fid_str = f"{single_fid.item():.4f}"
                except Exception as fid_e:
                    # Likely "Not enough samples" if duplication fails or matrix issues
                    fid_str = "Error"
                
                print(f"Processed {folder_name}: LPIPS = {lpips_value:.4f} | FID = {fid_str}")
                count += 1
                
            except Exception as e:
                print(f"Failed to process {folder_name}: {e}")
                traceback.print_exc()
        else:
            print(f"Skipping {folder_name}: Missing GT or Gen image.")

    print("\n" + "="*50)
    print(f"Total pairs processed: {count}")
    
    if count > 0:
        # Finalize Average LPIPS
        avg_lpips = sum(lpips_scores) / len(lpips_scores)
        print(f"Average LPIPS: {avg_lpips:.4f} (Lower is better)")
        
        # Finalize FID
        # Note: FID is typically calculated on a large dataset (>2048 images is recommended) 
        # to be statistically significant. With fewer images, it may be biased or unstable.
        print("Calculating FID (this may take a moment)...")
        try:
            fid_score = fid.compute()
            print(f"FID Score: {fid_score.item():.4f} (Lower is better)")
        except Exception as e:
            print(f"Error computing FID: {e}")
            print("Note: FID requires at least 2 images in each distribution (real and fake).")

    print("="*50 + "\n")

if __name__ == "__main__":
    COMPARISON_DIR = "/home/victus/Desktop/Scrape/comparison"
    calculate_metrics(COMPARISON_DIR)
