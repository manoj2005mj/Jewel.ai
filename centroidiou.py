import numpy as np
import torch
import cv2
import os
import traceback
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# ==========================================
# 1. INITIALIZE SEGMENTATION MODEL GLOBALLY
# ==========================================
# Loading globally prevents the model from reloading on every function call
print("Loading CLIPSeg model...")
PROCESSOR = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
MODEL = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL.to(torch.device(DEVICE))
print(f"Model loaded on {DEVICE}.")

# ==========================================
# 2. SEGMENTATION SETTINGS
# ==========================================
DEFAULT_PROMPT = "jewelry"
DEFAULT_THRESHOLD = 0.1


# ==========================================
# 3. MASK EXTRACTION (CLIPSeg)
# ==========================================
def _get_raw_mask_prob(image, prompt):
    """Run CLIPSeg on a single prompt and return the probability map (0-1 float numpy)."""
    inputs = PROCESSOR(
        text=[prompt],
        images=[image],
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = MODEL(**inputs)

    preds = outputs.logits
    if preds.dim() == 3:
        mask_prob = torch.sigmoid(preds[0]).cpu().numpy()
    else:
        mask_prob = torch.sigmoid(preds).cpu().numpy()
    return mask_prob


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
    Extract a binary segmentation mask for jewelry in an image.
    Resizes CLIPSeg 352×352 output back to original image dimensions.
    Keeps only the largest connected component to remove noise.
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    if prompt is None:
        prompt = DEFAULT_PROMPT
    
    # Check if raw_image_path is a path string or PIL Image
    if isinstance(raw_image_path, str):
        image = Image.open(raw_image_path).convert("RGB")
        filename = os.path.basename(raw_image_path)
    else:
        image = raw_image_path.convert("RGB")
        filename = "image.png"

    orig_w, orig_h = image.size

    # --- Get probability map (352×352 from CLIPSeg) ---
    mask_prob = _get_raw_mask_prob(image, prompt)

    # --- Resize prob map back to original image dimensions ---
    # Need to resize (352,352) -> (orig_w, orig_h)
    # mask_prob is (352,352)
    # cv2.resize expects (width, height)
    mask_prob_resized = cv2.resize(mask_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # --- Binarise ---
    binary_mask = (mask_prob_resized > threshold).astype(np.uint8) * 255

    # --- Keep only largest connected component ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels > 1:
        # label 0 is background, find largest foreground component
        # stats: [left, top, width, height, area]
        # We need to find index of max area skipping index 0
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # Create mask for largest component
        binary_mask = np.zeros_like(binary_mask)
        binary_mask[labels == largest_label] = 255

    # --- Optional morphological cleanup ---
    if morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Close gaps then remove noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

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
        # Resize GT image to match generated image dimensions
        gen_img = Image.open(raw_gen_path).convert("RGB")
        gen_w, gen_h = gen_img.size
        
        gt_img = Image.open(raw_gt_path).convert("RGB")
        if gt_img.size != (gen_w, gen_h):
            gt_img_resized = gt_img.resize((gen_w, gen_h), Image.Resampling.LANCZOS)
        else:
            gt_img_resized = gt_img

        # Create subfolders for this comparison in debug_masks
        gt_debug_dir = os.path.join(debug_folder, "gt")
        gen_debug_dir = os.path.join(debug_folder, "gen")
        aligned_debug_dir = os.path.join(debug_folder, "aligned")

        os.makedirs(gt_debug_dir, exist_ok=True)
        os.makedirs(gen_debug_dir, exist_ok=True)
        os.makedirs(aligned_debug_dir, exist_ok=True)

        # Extract masks
        mask_gt = extract_jewelry_mask(
            gt_img_resized, # Pass PIL image directly
            prompt=item_prompt,
            threshold=threshold,
            save_mask=True,
            output_dir=gt_debug_dir,
            mask_filename_prefix="gt"
        )
        
        mask_gen = extract_jewelry_mask(
            gen_img, # Pass PIL image directly
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
                
                if iou > 0:
                    iou_scores.append(iou)
                if area_acc > 0:
                    area_acc_scores.append(area_acc)

            else:
                print(f"Skipping {folder_name}: gt.png/jpg or image.png/jpg not found.")

        print("\n" + "="*50)
        if iou_scores:
            avg_iou = sum(iou_scores) / len(iou_scores)
            print(f"Average Centroid IoU (non-zero): {avg_iou:.4f}")
        else:
            print("No non-zero Centroid IoU scores computed.")

        if area_acc_scores:
            avg_area_acc = sum(area_acc_scores) / len(area_acc_scores)
            print(f"Average Area Accuracy (non-zero): {avg_area_acc:.4f}")
        else:
            print("No non-zero Area Accuracy scores computed.")
        print("="*50 + "\n")

    else:
        print(f"Comparison folder not found at {comparison_root}")