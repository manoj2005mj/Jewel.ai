import numpy as np
import torch
import cv2
import os
import traceback
from PIL import Image
from ultralytics.models.fastsam import FastSAM
import clip

# ==========================================
# 0.5 MONKEY PATCH TORCH LOAD
# ==========================================
# Fix for PyTorch 2.6+ causing "Weights only load failed"
# We need to allow loading complete objects for FastSAM/Ultralytics
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False if not specified to allow loading older models
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# ==========================================
# 1. INITIALIZE SEGMENTATION MODEL GLOBALLY
# ==========================================
# Loading globally prevents the model from reloading on every function call
print("Loading FastSAM model...")
# Path to FastSAM checkpoint
FASTSAM_CHECKPOINT_PATH = "./FastSAM-x.pt"  # Adjust as needed if unrelated to running dir
MODEL = FastSAM(FASTSAM_CHECKPOINT_PATH)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model loaded on {DEVICE}.")

# ==========================================
# 2. SEGMENTATION SETTINGS
# ==========================================
DEFAULT_PROMPT = "metal jewel"
DEFAULT_THRESHOLD = 0.5  # FastSAM usually gives binary or high conf masks


# ==========================================
# 3. MASK EXTRACTION (FastSAM + CLIP Filtering)
# ==========================================
# Reusing logic from run_fastsam.py which works correctly
REJECT_CLASSES = ["face", "skin", "hair", "neck", "person"]
MAX_OBJ_SIZE_RATIO = 0.05 

def extract_jewelry_mask(
    raw_image_path,
    prompt=None,
    threshold=None,
    morphology=True,
    save_mask=False,
    output_dir="debug_masks",
    mask_filename_prefix="",
):
    """
    Extract a binary segmentation mask for jewelry in an image using FastSAM + CLIP Filtering.
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    if prompt is None:
        prompt = DEFAULT_PROMPT
    
    # Check if raw_image_path is a path string or PIL Image
    if isinstance(raw_image_path, str):
        image_path = raw_image_path
        image = Image.open(raw_image_path).convert("RGB")
        filename = os.path.basename(raw_image_path)
    else:
        raise ValueError("extract_jewelry_mask expects a file path string. Pass the path, not a PIL image.")

    orig_w, orig_h = image.size
    
    # Needs CLIP model for filtering
    # Load CLIP local to function or globally if preferred, let's load globally or cached
    if not hasattr(extract_jewelry_mask, 'clip_model'):
        print("Loading CLIP for smart filtering...")
        extract_jewelry_mask.clip_model, extract_jewelry_mask.preprocess = clip.load("ViT-B/32", device=DEVICE)
    
    clip_model = extract_jewelry_mask.clip_model
    preprocess = extract_jewelry_mask.preprocess

    # --- Run FastSAM ---
    # retina_masks=True gives better quality mask
    results = MODEL(
        image_path, 
        device=DEVICE, 
        retina_masks=True, 
        imgsz=1024, 
        conf=0.2, 
        iou=0.9
    )
    
    # FastSAM usually returns a list [Results, ...]
    if isinstance(results, list) and len(results) > 0:
        result = results[0]
        if result.masks is None:
            return np.zeros((orig_h, orig_w), dtype=np.uint8)
    else:
         # Unexpected return type
         return np.zeros((orig_h, orig_w), dtype=np.uint8)

    masks = result.masks.data
    boxes = result.boxes.xyxy
    
    # Find {len(masks)} potential objects. Filtering...
    
    # Prepare text embeddings
    search_classes = [prompt] + REJECT_CLASSES
    text_inputs = clip.tokenize(search_classes).to(DEVICE)
    
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    final_masks = []
    
    # FIX: Always read original image via cv2.imread from the file path,
    # exactly like run_fastsam.py does, to ensure pixel dimensions match
    # the mask/box coordinates returned by FastSAM.
    original_img_np = cv2.imread(image_path)
    original_img_np = cv2.cvtColor(original_img_np, cv2.COLOR_BGR2RGB)
             
    h, w, _ = original_img_np.shape
    total_area = h * w

    for i, mask in enumerate(masks):
        mask_np = mask.cpu().numpy().astype(bool)
        mask_area = np.sum(mask_np)
        size_ratio = mask_area / total_area
        
        # Filter A: Size
        if size_ratio > MAX_OBJ_SIZE_RATIO:
            continue
        
        # Filter B: CLIP Semantic Check
        # Use mask-tight bounding box for a better crop (less background noise)
        mask_ys, mask_xs = np.where(mask_np)
        if len(mask_xs) == 0: continue
        x1 = int(np.min(mask_xs))
        y1 = int(np.min(mask_ys))
        x2 = int(np.max(mask_xs))
        y2 = int(np.max(mask_ys))
        margin = 10
        x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin); y2 = min(h, y2 + margin)
        
        # Create masked crop: white-out non-jewelry pixels so CLIP focuses on the object
        crop = original_img_np[y1:y2, x1:x2].copy()
        crop_mask = mask_np[y1:y2, x1:x2]
        crop[~crop_mask] = 255  # white background
        if crop.size == 0: continue
        
        crop_pil = Image.fromarray(crop)
        # Type check: preprocess usually returns a tensor.
        image_input = preprocess(crop_pil)
        if torch.is_tensor(image_input):
             image_input = image_input.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probs = similarity[0].cpu().numpy()

        target_score = probs[0]
        reject_score = np.max(probs[1:])
        
        if target_score > reject_score:
            final_masks.append(mask_np)

    if not final_masks:
        binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    else:
        # Combine valid masks
        combined_mask = np.any(np.array(final_masks), axis=0)
        binary_mask = (combined_mask.astype(np.uint8) * 255)
        
        # Resize if necessary
        if binary_mask.shape[:2] != (orig_h, orig_w):
             binary_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # --- Match run_fastsam.py logic EXACTLY: NO extra morphological cleanup ---
    # The original run_fastsam.py does NOT equate morphology.
    # It just combines masks and saves.
    
    # --- Save for debugging ---
    if save_mask:
        os.makedirs(output_dir, exist_ok=True)
        if mask_filename_prefix:
            save_path = os.path.join(output_dir, f"{mask_filename_prefix}_{filename}")
        else:
            save_path = os.path.join(output_dir, f"segmented_mask_{filename}")
        Image.fromarray(binary_mask).save(save_path)

    return binary_mask


# ==========================================
# 4. CUSTOM METRIC UTILITIES
# ==========================================
def get_centroid(mask):
    """
    Calculate the centroid of a binary mask.
    Returns (x, y) coordinates of the centroid, or None if mask is empty.
    """
    # mask > 0 returns (row_indices, col_indices)
    # row is y, col is x
    y_coords, x_coords = np.where(mask > 0)
    
    if len(x_coords) == 0:
        return None
        
    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)
    
    return (mean_x, mean_y)


def calculate_area_accuracy(mask_gt, mask_gen):
    """
    Calculate accuracy based on area of masks.
    Returns: accuracy (0-1), area_gt, area_gen
    """
    area_gt = np.sum(mask_gt > 0)
    area_gen = np.sum(mask_gen > 0)
    
    if area_gt == 0:
        return 0.0, area_gt, area_gen
        
    # Simple ratio: min/max
    if area_gen == 0:
        return 0.0, area_gt, area_gen
        
    accuracy = min(area_gt, area_gen) / max(area_gt, area_gen)
    return accuracy, area_gt, area_gen


def evaluate_tryon_images(
    raw_gt_path,
    raw_gen_path,
    item_prompt=None,
    threshold=None,
    debug_folder="debug_masks"
):
    """
    End-to-end pipeline: Extracts masks -> Aligns Centroids -> Calculates IoU and Area Accuracy.
    """
    try:
        # Get generated image dimensions for mask alignment
        gen_img = Image.open(raw_gen_path).convert("RGB")
        gen_w, gen_h = gen_img.size

        # Create subfolders for this comparison in debug_masks
        gt_debug_dir = os.path.join(debug_folder, "gt")
        gen_debug_dir = os.path.join(debug_folder, "gen")
        aligned_debug_dir = os.path.join(debug_folder, "aligned")

        os.makedirs(gt_debug_dir, exist_ok=True)
        os.makedirs(gen_debug_dir, exist_ok=True)
        os.makedirs(aligned_debug_dir, exist_ok=True)

        # Extract masks - pass FILE PATHS, not PIL images
        # FastSAM works correctly only with file paths (matching run_fastsam.py)
        mask_gt_original = extract_jewelry_mask(
            raw_gt_path,  # Pass file path, NOT PIL image
            prompt=item_prompt,
            threshold=threshold,
            save_mask=True,
            output_dir=gt_debug_dir,
            mask_filename_prefix="gt_original"
        )
        
        # Resize the GT mask to match Gen image dimensions for comparison
        if mask_gt_original.shape[:2] != (gen_h, gen_w):
             # cv2 uses (width, height)
             mask_gt = cv2.resize(mask_gt_original, (gen_w, gen_h), interpolation=cv2.INTER_NEAREST)
        else:
             mask_gt = mask_gt_original

        mask_gen = extract_jewelry_mask(
            raw_gen_path,  # Pass file path, NOT PIL image
            prompt=item_prompt,
            threshold=threshold,
            save_mask=True,
            output_dir=gen_debug_dir,
            mask_filename_prefix="gen"
        )

        # --- Calculate Area Accuracy ---
        area_acc, area_gt, area_gen = calculate_area_accuracy(mask_gt, mask_gen)
        print(f"Area Accuracy: {area_acc:.4f} (GT Area: {area_gt}, Gen Area: {area_gen})")

        # --- Centroid Alignment & IoU ---
        # Find centroids
        centroid_gt = get_centroid(mask_gt)
        centroid_gen = get_centroid(mask_gen)

        iou_score = 0.0
        aligned_mask_gen = None

        if centroid_gt is None or centroid_gen is None:
            print("Warning: One of the masks is empty. Cannot compute centroid alignment.")
        else:
            print(f"GT centroid:  {centroid_gt}")
            print(f"Gen centroid: {centroid_gen}")

            # Calculate shift to move gen mask to align with gt mask
            dx = centroid_gt[0] - centroid_gen[0]
            dy = centroid_gt[1] - centroid_gen[1]
            print(f"Shift: dx={dx:.1f}, dy={dy:.1f}")

            rows, cols = mask_gen.shape
            translation_matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            aligned_mask_gen = cv2.warpAffine(mask_gen, translation_matrix, (cols, rows))

            # Save aligned mask for debugging
            aligned_filename = f"aligned_mask_{os.path.basename(raw_gen_path)}"
            Image.fromarray(aligned_mask_gen).save(os.path.join(aligned_debug_dir, aligned_filename))
            # print(f"Saved aligned mask to {aligned_debug_dir}")

            # IoU
            intersection = np.logical_and(mask_gt > 0, aligned_mask_gen > 0)
            union = np.logical_or(mask_gt > 0, aligned_mask_gen > 0)

            union_sum = np.sum(union)
            if union_sum > 0:
                iou_score = np.sum(intersection) / union_sum

        return iou_score, area_acc

    except Exception as e:
        print(f"Error in evaluate_tryon_images: {e}")
        traceback.print_exc()
        return 0.0, 0.0


# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    comparison_root = "/home/victus/Desktop/Scrape/comparison"
    debug_root = "debug_masks"
    
    # Iterate through each folder in comparison
    if os.path.exists(comparison_root):
        folders = [f for f in os.listdir(comparison_root) if os.path.isdir(os.path.join(comparison_root, f))]
        
        iou_scores = []
        area_acc_scores = []

        for folder_name in sorted(folders):
            folder_path = os.path.join(comparison_root, folder_name)
            print(f"\nProcessing folder: {folder_name}")
            
            # Define file paths
            # Assuming standard naming inside comparison folders: gt.png and image.png
            # Adjust if filenames vary
            gt_path = os.path.join(folder_path, "gt.png")
            gen_path = os.path.join(folder_path, "image.png")
            
            if not os.path.exists(gt_path):
                # Try jpg if png doesn't exist
                gt_path = os.path.join(folder_path, "gt.jpg")
            
            if not os.path.exists(gen_path):
                 gen_path = os.path.join(folder_path, "image.jpg")

            if os.path.exists(gt_path) and os.path.exists(gen_path):
                # Create specific debug folder for this item
                item_debug_dir = os.path.join(debug_root, folder_name)
                
                iou, area_acc = evaluate_tryon_images(
                    gt_path,
                    gen_path,
                    debug_folder=item_debug_dir
                )
                print(f"Folder: {folder_name} | Centroid IoU: {iou:.4f} | Area Accuracy: {area_acc:.4f}")
                
                # Filter out low scores (less than 0.2) from the average
                if iou >= 0.2:
                    iou_scores.append(iou)
                if area_acc >= 0.2:
                    area_acc_scores.append(area_acc)

            else:
                print(f"Skipping {folder_name}: gt.png/jpg or image.png/jpg not found.")

        print("\n" + "="*50)
        if iou_scores:
            avg_iou = sum(iou_scores) / len(iou_scores)
            print(f"Average Centroid IoU (>= 0.2): {avg_iou:.4f}")
        else:
            print("No Centroid IoU scores >= 0.2 computed.")

        if area_acc_scores:
            avg_area_acc = sum(area_acc_scores) / len(area_acc_scores)
            print(f"Average Area Accuracy (>= 0.2): {avg_area_acc:.4f}")
        else:
            print("No Area Accuracy scores >= 0.2 computed.")
        print("="*50 + "\n")

    else:
        print(f"Comparison folder not found at {comparison_root}")