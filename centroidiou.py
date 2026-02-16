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
DEFAULT_THRESHOLD = 0.4


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

    image = Image.open(raw_image_path).convert("RGB")
    orig_w, orig_h = image.size

    # --- Get probability map (352×352 from CLIPSeg) ---
    mask_prob = _get_raw_mask_prob(image, prompt)

    # --- Resize prob map back to original image dimensions ---
    mask_prob_resized = cv2.resize(mask_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # --- Binarise ---
    binary_mask = (mask_prob_resized > threshold).astype(np.uint8) * 255

    # --- Keep only largest connected component ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels > 1:
        # label 0 is background, find largest foreground component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary_mask = ((labels == largest_label) * 255).astype(np.uint8)

    # --- Optional morphological cleanup ---
    if morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # --- Save for debugging ---
    if save_mask:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(raw_image_path)
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
    coords = np.column_stack(np.where(mask > 0))
    if coords.shape[0] == 0:
        return None
    centroid = coords.mean(axis=0)
    # np.where returns (row, col) -> (y, x). Format to (x, y).
    return centroid[::-1]


def evaluate_tryon_images(
    raw_gt_path,
    raw_gen_path,
    item_prompt=None,
    threshold=None,
):
    """
    End-to-end pipeline: Extracts masks -> Aligns Centroids -> Calculates IoU.
    """
    try:
        # Resize GT image to match generated image dimensions
        gen_img = Image.open(raw_gen_path)
        gen_w, gen_h = gen_img.size
        gt_img = Image.open(raw_gt_path).convert("RGB")
        if gt_img.size != (gen_w, gen_h):
            gt_img_resized = gt_img.resize((gen_w, gen_h), Image.LANCZOS)
            resized_gt_path = raw_gt_path + "_resized.jpg"
            gt_img_resized.save(resized_gt_path)
            print(f"Resized GT from {gt_img.size} to ({gen_w}, {gen_h})")
        else:
            resized_gt_path = raw_gt_path

        mask_gt = extract_jewelry_mask(
            resized_gt_path,
            prompt=item_prompt,
            threshold=threshold,
            save_mask=True,
            output_dir="debug_masks/gt",
        )
        mask_gen = extract_jewelry_mask(
            raw_gen_path,
            prompt=item_prompt,
            threshold=threshold,
            save_mask=True,
            output_dir="debug_masks/gen",
        )

        # Find centroids
        centroid_gt = get_centroid(mask_gt)
        centroid_gen = get_centroid(mask_gen)

        if centroid_gt is None or centroid_gen is None:
            print("Warning: One of the masks is empty. Cannot compute centroid alignment.")
            return 0.0

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
        os.makedirs("debug_masks/aligned", exist_ok=True)
        Image.fromarray(aligned_mask_gen).save("debug_masks/aligned/aligned_mask_gen.png")
        print("Saved aligned mask to debug_masks/aligned/aligned_mask_gen.png")

        # IoU
        intersection = np.logical_and(mask_gt > 0, aligned_mask_gen > 0)
        union = np.logical_or(mask_gt > 0, aligned_mask_gen > 0)

        union_sum = np.sum(union)
        if union_sum == 0:
            return 0.0

        iou_score = np.sum(intersection) / union_sum
        return iou_score

    except Exception as e:
        print(f"Error in evaluate_tryon_images: {e}")
        traceback.print_exc()
        return 0.0


# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    ground_truth_file = "/home/victus/Desktop/Scrape/WER0297-1.jpg"
    generated_file = "/home/victus/Desktop/Scrape/Images/Generated Image February 16, 2026 - 11_55AM.jpeg"

    score = evaluate_tryon_images(
        ground_truth_file,
        generated_file,
    )
    print(f"Translation-Invariant IoU Score (prompt='{DEFAULT_PROMPT}', thresh={DEFAULT_THRESHOLD}): {score:.4f}")