"""
Diagnostic script: Compare CLIPSeg segmentation quality across thresholds
and prompt variations to find the best configuration for earring detection.
"""
import numpy as np
import torch
import cv2
import os
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

print("Loading CLIPSeg model...")
PROCESSOR = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
MODEL = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL.to(torch.device(DEVICE))
print(f"Model loaded on {DEVICE}.")

GT_PATH = "/home/victus/Desktop/Scrape/WER0297-1.jpg"
GEN_PATH = "/home/victus/Desktop/Scrape/Images/Screenshot from 2026-02-15 01-20-50.png"

OUT_DIR = "debug_masks/diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# 1. Test different text prompts
# ==========================================
prompts = [
    "earring",
    "earrings",
    "a pair of earrings",
    "diamond earring",
    "jewelry earring on ear",
    "ear stud",
    "earring jewelry",
    "shiny earring accessory",
]

def get_raw_mask_prob(image_path, prompt):
    """Get the raw probability map (0-1 float) for a prompt."""
    image = Image.open(image_path).convert("RGB")
    inputs = PROCESSOR(text=[prompt], images=[image], padding="max_length", return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = MODEL(**inputs)
    preds = outputs.logits
    if preds.dim() == 3:
        mask_prob = torch.sigmoid(preds[0]).cpu().numpy()
    else:
        mask_prob = torch.sigmoid(preds).cpu().numpy()
    return mask_prob

print("\n" + "="*60)
print("PROMPT COMPARISON (Ground Truth Image)")
print("="*60)

best_prompt = None
best_score = -1

for prompt in prompts:
    prob = get_raw_mask_prob(GT_PATH, prompt)
    
    max_conf = prob.max()
    mean_conf = prob.mean()
    # Pixels above various thresholds
    pix_03 = (prob > 0.3).sum()
    pix_05 = (prob > 0.5).sum()
    pix_07 = (prob > 0.7).sum()
    
    # A good segmentation has HIGH max confidence and a reasonable number of pixels
    # Score: balance between confidence and coverage
    score = max_conf * (pix_05 / prob.size) * 1000  # arbitrary but useful ranking
    
    print(f"\nPrompt: '{prompt}'")
    print(f"  Max confidence: {max_conf:.4f}")
    print(f"  Mean confidence: {mean_conf:.4f}")
    print(f"  Pixels > 0.3: {pix_03:>6} ({pix_03/prob.size*100:.1f}%)")
    print(f"  Pixels > 0.5: {pix_05:>6} ({pix_05/prob.size*100:.1f}%)")
    print(f"  Pixels > 0.7: {pix_07:>6} ({pix_07/prob.size*100:.1f}%)")
    print(f"  Ranking score: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_prompt = prompt
    
    # Save probability heatmap for visual inspection
    heatmap = (prob * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    safe_name = prompt.replace(" ", "_")[:30]
    cv2.imwrite(os.path.join(OUT_DIR, f"gt_heatmap_{safe_name}.png"), heatmap_color)
    
    # Also save binary at 0.3 threshold
    binary = ((prob > 0.3).astype(np.uint8)) * 255
    Image.fromarray(binary).save(os.path.join(OUT_DIR, f"gt_binary03_{safe_name}.png"))

print(f"\n>>> Best prompt: '{best_prompt}' (score: {best_score:.4f})")

# ==========================================
# 2. Test Multi-prompt ensemble (average multiple prompts)
# ==========================================
print("\n" + "="*60)
print("MULTI-PROMPT ENSEMBLE")
print("="*60)

ensemble_prompts = ["jewelry earring on ear"]
probs_gt = []
probs_gen = []

for p in ensemble_prompts:
    probs_gt.append(get_raw_mask_prob(GT_PATH, p))
    probs_gen.append(get_raw_mask_prob(GEN_PATH, p))

ensemble_gt = np.mean(probs_gt, axis=0)
ensemble_gen = np.mean(probs_gen, axis=0)

print(f"Ensemble GT  - Max: {ensemble_gt.max():.4f}, Mean: {ensemble_gt.mean():.4f}, Pixels>0.3: {(ensemble_gt>0.3).sum()}, Pixels>0.5: {(ensemble_gt>0.5).sum()}")
print(f"Ensemble GEN - Max: {ensemble_gen.max():.4f}, Mean: {ensemble_gen.mean():.4f}, Pixels>0.3: {(ensemble_gen>0.3).sum()}, Pixels>0.5: {(ensemble_gen>0.5).sum()}")

# Save ensemble heatmaps
for name, ens in [("gt", ensemble_gt), ("gen", ensemble_gen)]:
    heatmap = (ens * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(OUT_DIR, f"{name}_ensemble_heatmap.png"), heatmap_color)
    for t in [0.2, 0.3, 0.4, 0.5]:
        binary = ((ens > t).astype(np.uint8)) * 255
        Image.fromarray(binary).save(os.path.join(OUT_DIR, f"{name}_ensemble_binary_{t}.png"))

# ==========================================
# 3. Compute IoU with ensemble at various thresholds
# ==========================================
print("\n" + "="*60)
print("IoU COMPARISON: Single-prompt vs Ensemble at various thresholds")
print("="*60)

def centroid_iou(mask_a, mask_b):
    """Compute centroid-aligned IoU between two binary masks."""
    coords_a = np.column_stack(np.where(mask_a > 0))
    coords_b = np.column_stack(np.where(mask_b > 0))
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return 0.0
    ca = coords_a.mean(axis=0)[::-1]  # (x, y)
    cb = coords_b.mean(axis=0)[::-1]
    dx, dy = float(ca[0] - cb[0]), float(ca[1] - cb[1])
    rows, cols = mask_b.shape
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    aligned_b = cv2.warpAffine(mask_b, M, (cols, rows))
    inter = np.logical_and(mask_a > 0, aligned_b > 0).sum()
    union = np.logical_or(mask_a > 0, aligned_b > 0).sum()
    return float(inter / union) if union > 0 else 0.0

# Single prompt at various thresholds
single_gt_prob = get_raw_mask_prob(GT_PATH, "earring")
single_gen_prob = get_raw_mask_prob(GEN_PATH, "earring")

print(f"\n{'Threshold':>10} | {'Single IoU':>12} | {'Ensemble IoU':>12} | {'Single GT px':>12} | {'Ens GT px':>12}")
print("-" * 70)

for thresh in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
    s_gt = ((single_gt_prob > thresh).astype(np.uint8)) * 255
    s_gen = ((single_gen_prob > thresh).astype(np.uint8)) * 255
    e_gt = ((ensemble_gt > thresh).astype(np.uint8)) * 255
    e_gen = ((ensemble_gen > thresh).astype(np.uint8)) * 255
    
    s_iou = centroid_iou(s_gt, s_gen)
    e_iou = centroid_iou(e_gt, e_gen)
    
    print(f"{thresh:>10.2f} | {s_iou:>12.4f} | {e_iou:>12.4f} | {(s_gt>0).sum():>12} | {(e_gt>0).sum():>12}")

print(f"\nDiagnostic images saved to: {OUT_DIR}/")
print("Open the heatmap images to visually inspect which prompt captures earrings best.")
