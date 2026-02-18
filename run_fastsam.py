import numpy as np
import matplotlib

# --- FIX 1: Force "Agg" backend (Headless/No GUI) ---
# This prevents "FigureCanvasTkAgg" errors and is faster.
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# --- FIX 2: Monkey-Patch 'tostring_rgb' ---
# We manually add the missing function back to Matplotlib
def tostring_rgb_patch(self):
    self.draw()
    # Get RGBA buffer and convert to RGB (drop alpha channel)
    return np.array(self.buffer_rgba())[:, :, :3].tobytes()

# Apply the patch
FigureCanvasAgg.tostring_rgb = tostring_rgb_patch
# ----------------------------------------------------

import torch
import os
import cv2
from PIL import Image
from ultralytics.models.fastsam import FastSAM
import clip 

# ==========================================
#   USER CONFIGURATION
# ==========================================
IMG_PATH = "/home/victus/Desktop/Scrape/comparison/The_Vivaciously_Designed_Huggie_Earrings/gt.png"
OUTPUT_PATH = "/home/victus/Desktop/Scrape/output_result.jpg"
MODEL_PATH = "/home/victus/Desktop/Scrape/FastSAM-x.pt"

TARGET_PROMPT = "metal"  # The object you want to segment (e.g., "a jewel", "an earring")
REJECT_CLASSES = ["face", "skin", "hair", "neck", "person"]
MAX_OBJ_SIZE_RATIO = 0.01
# ==========================================

def run_smart_segmentation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}...")

    # Load FastSAM
    model = FastSAM(MODEL_PATH)
    
    # Load CLIP
    print("Loading CLIP for smart filtering...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    print("Scanning image for all objects...")
    results = model(
        IMG_PATH, 
        device=device, 
        retina_masks=True, 
        imgsz=1024, 
        conf=0.1,
        iou=0.9
    )
    
    prompt_process = None  # FastSAMPrompt removed (incompatible with ultralytics 8.x)
    if results[0].masks is None:
        print("No objects detected at all.")
        return

    masks = results[0].masks.data
    boxes = results[0].boxes.xyxy
    
    print(f"Found {len(masks)} potential objects. Filtering...")
    
    # Prepare text embeddings
    search_classes = [TARGET_PROMPT] + REJECT_CLASSES
    text_inputs = clip.tokenize(search_classes).to(device)
    
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    final_masks = []
    
    original_img = cv2.imread(IMG_PATH)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w, _ = original_img.shape
    total_area = h * w

    for i, mask in enumerate(masks):
        mask_np = mask.cpu().numpy().astype(bool)
        mask_area = np.sum(mask_np)
        size_ratio = mask_area / total_area
        
        # Filter A: Size
        if size_ratio > MAX_OBJ_SIZE_RATIO:
            continue
        
        # Filter B: CLIP Semantic Check
        box = boxes[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        margin = 5
        x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin); y2 = min(h, y2 + margin)
        
        crop = original_img[y1:y2, x1:x2]
        if crop.size == 0: continue
        
        crop_pil = Image.fromarray(crop)
        image_input = preprocess(crop_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probs = similarity[0].cpu().numpy()

        target_score = probs[0]
        reject_score = np.max(probs[1:])
        
        if target_score > reject_score:
            print(f"  -> Found match! (Conf: {target_score:.2f})")
            final_masks.append(mask_np)

    # Save Result
    if len(final_masks) > 0:
        combined_mask = np.any(np.array(final_masks), axis=0)
        combined_mask_uint8 = combined_mask.astype(np.uint8) * 255

        print(f"Saving final result to {OUTPUT_PATH}")
        
        # Save mask overlaid on original image
        original_img_bgr = cv2.imread(IMG_PATH)
        mask_colored = cv2.applyColorMap(combined_mask_uint8, cv2.COLORMAP_JET)
        if mask_colored.shape[:2] != original_img_bgr.shape[:2]:
            mask_colored = cv2.resize(mask_colored, (original_img_bgr.shape[1], original_img_bgr.shape[0]))
        overlay = cv2.addWeighted(original_img_bgr, 0.6, mask_colored, 0.4, 0)
        cv2.imwrite(OUTPUT_PATH, overlay)
        print("Done!")
    else:
        print("No specific earrings found after filtering.")

if __name__ == "__main__":
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else original_load(*args, **kwargs)
    
    try:
        run_smart_segmentation()
    finally:
        torch.load = original_load