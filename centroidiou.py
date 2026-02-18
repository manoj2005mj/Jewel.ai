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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check relative path first, then look for FastSAM/FastSAM-x.pt relative to that
candidates = [
    "FastSAM-x.pt",
    "FastSAM/FastSAM-x.pt",
    "../FastSAM-x.pt"
]

FASTSAM_CHECKPOINT_PATH = None
for c in candidates:
    p = os.path.join(BASE_DIR, c)
    if os.path.exists(p):
        FASTSAM_CHECKPOINT_PATH = p
        break

if FASTSAM_CHECKPOINT_PATH is None:
    # try default absolute assumed path just in case
    # or raise clear error
    p = os.path.abspath("FastSAM-x.pt")
    if os.path.exists(p):
        FASTSAM_CHECKPOINT_PATH = p
    else:
        print("Warning: FastSAM-x.pt not found around script directory. Model loading may fail.")
        FASTSAM_CHECKPOINT_PATH = "FastSAM-x.pt" # Let library try to handle or fail

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
    if original_img_np is None:
        print(f"Error: Could not read image at {image_path}")
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

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
    
        # If using official CLIP, preprocess(image) returns a Tensor (3, 224, 224).
        # If using OpenCLIP or others, might differ, but generally it's a Tensor.
        if torch.is_tensor(image_input):
            image_input = image_input.unsqueeze(0).to(DEVICE)
