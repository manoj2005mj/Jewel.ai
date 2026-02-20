#!/usr/bin/env python3
"""
Jewelry Virtual Try-On Compositor

This script performs precision compositing of earrings onto reference images
using Gemini Pro for automatic earlobe detection and scale calculation.
"""

import os
import json
import glob
import numpy as np
from PIL import Image
import rembg
import dotenv
from typing import Tuple, Optional, Dict, Any


def parse_dimension_to_cm(dimension_str: str) -> float:
    """
    Parse a dimension string (e.g., '14.75 mm' or '1.5 cm') to centimeters.
    """
    dimension_str = dimension_str.strip().lower()
    value_str = dimension_str.split()[0]
    try:
        value = float(value_str)
    except ValueError:
        raise ValueError(f"Cannot parse numeric value from: {dimension_str}")

    if 'mm' in dimension_str:
        return value / 10.0
    elif 'cm' in dimension_str:
        return value
    else:
        raise ValueError(f"Unknown unit in dimension string: {dimension_str}")

def composite_earring(
    base_image_path: str,
    jewelry_image_path: str,
    earlobe_coords: Tuple[int, int],
    px_per_cm: float,
    jewelry_height_cm: float,
    jewelry_width_cm: float,
    output_dir: Optional[str] = None
) -> Optional[str]:
    """
    Composite an earring onto a base image with precise scaling and positioning.
    """
    try:
        # Load base image
        base_image = Image.open(base_image_path).convert("RGBA")

        # Load and remove background from jewelry image
        jewelry_image = Image.open(jewelry_image_path)
        jewelry_rgba = rembg.remove(jewelry_image)

        # Convert to numpy array for bounding box calculation
        jewelry_array = np.array(jewelry_rgba)

        # Find bounding box of non-transparent pixels
        if jewelry_array.shape[2] == 4:
            alpha_channel = jewelry_array[:, :, 3]
            non_transparent_mask = alpha_channel > 0

            if not non_transparent_mask.any():
                print("Warning: Jewelry image is fully transparent after background removal")
                return None

            rows = np.any(non_transparent_mask, axis=1)
            cols = np.any(non_transparent_mask, axis=0)
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]

            if len(y_indices) == 0 or len(x_indices) == 0:
                print("Warning: Could not find valid bounding box")
                return None

            y_min, y_max = y_indices[0], y_indices[-1]
            x_min, x_max = x_indices[0], x_indices[-1]

            # Crop to bounding box
            jewelry_cropped = jewelry_rgba.crop((x_min, y_min, x_max + 1, y_max + 1))
        else:
            print("Warning: Jewelry image doesn't have alpha channel")
            jewelry_cropped = jewelry_rgba

        # Calculate target dimensions with 1:1 scale (physically accurate)
        visibility_multiplier = 1
        target_height = int(jewelry_height_cm * px_per_cm * visibility_multiplier)
        target_width = int(jewelry_width_cm * px_per_cm * visibility_multiplier)

        print(f"Resizing jewelry: {target_width} x {target_height} pixels")
        print(f"  Scale: {px_per_cm:.2f} px/cm | Multiplier: {visibility_multiplier}x")
        print(f"  Physical size: {jewelry_width_cm:.2f} x {jewelry_height_cm:.2f} cm")

        # Resize using LANCZOS
        jewelry_resized = jewelry_cropped.resize(
            (target_width, target_height),
            Image.Resampling.LANCZOS
        )

        # Calculate paste position (top-center anchored at earlobe)
        paste_x = earlobe_coords[0] - (target_width // 2)
        paste_y = earlobe_coords[1]

        # Handle out-of-bounds
        base_width, base_height = base_image.size

        if paste_x < 0:
            paste_x = 0
        elif paste_x + target_width > base_width:
            paste_x = base_width - target_width

        if paste_y < 0:
            paste_y = 0
        elif paste_y + target_height > base_height:
            paste_y = base_height - target_height

        print(f"Pasting at: ({paste_x}, {paste_y})")

        # Paste with alpha channel
        base_image.paste(jewelry_resized, (paste_x, paste_y), jewelry_resized)

        # Convert to RGB and save
        output_image = base_image.convert("RGB")

        if output_dir is None:
            output_dir = os.path.dirname(base_image_path)
        output_path = os.path.join(output_dir, "stickerimg.png")
        output_image.save(output_path)

        print(f"Saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error during compositing: {e}")
        import traceback
        traceback.print_exc()
        return None


def read_dimensions_json(json_path: str) -> Dict[str, float]:
    """Read dimensions from JSON file and convert to cm."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    height_str = data.get('height')
    width_str = data.get('breadth')

    if not height_str or not width_str:
        raise ValueError("dimensions.json must contain 'height' and 'breadth' fields")

    return {
        'height': parse_dimension_to_cm(height_str),
        'width': parse_dimension_to_cm(width_str)
    }


def main() -> None:
    """
    Main execution block for jewelry compositing using hardcoded dimensions.
    """
    # Load environment variables
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv.load_dotenv(os.path.join(base_dir, ".env"))

    # ============================================================
    # HARDCODED DIMENSIONS CONFIGURATION
    # ============================================================
    # User-specified dimensions for compositing
    # Modify these values as needed for different reference images

    # Earlobe coordinates (x, y) for earring placement
    EARLOBE_COORDS = (503, 523)  # Right earlobe position

    # Scale: pixels per centimeter
    PX_PER_CM = 54.70  # Physical scale factor

    # ============================================================

    # Configuration
    BASE_DIR = os.path.join(base_dir, "Intialtest")
    REFERENCE_IMAGE_PATH = os.path.join(base_dir, "reference.jpeg")

    print("=" * 60)
    print("Jewelry Virtual Try-On Compositor")
    print("=" * 60)

    # Check reference image
    if not os.path.exists(REFERENCE_IMAGE_PATH):
        print(f"Error: Reference image '{REFERENCE_IMAGE_PATH}' not found")
        return

    # Check Intialtest folder
    base_intialtest = os.path.join(base_dir, "Intialtest")
    if not os.path.exists(base_intialtest):
        print(f"Error: 'Intialtest' folder not found")
        return

    # Use the correct base directory
    BASE_DIR = base_intialtest

    # Use hardcoded dimensions
    px_per_cm = PX_PER_CM
    earlobe_coords = EARLOBE_COORDS

    print(f"\nUsing Hardcoded Dimensions:")
    print(f"  Earlobe Coordinates: {earlobe_coords}")
    print(f"  Scale: {px_per_cm:.2f} pixels per cm")

    # Get product folders
    product_folders = [
        os.path.join(BASE_DIR, d)
        for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ]

    if not product_folders:
        print(f"Error: No product folders found in '{BASE_DIR}'")
        return

    # Sort product folders alphabetically for consistent processing
    product_folders.sort()

    print(f"\nFound {len(product_folders)} product folders to process")
    print("=" * 60)

    # Process each product folder
    success_count = 0
    fail_count = 0

    for idx, product_folder in enumerate(product_folders, 1):
        product_name = os.path.basename(product_folder)
        print(f"\n[{idx}/{len(product_folders)}] Processing: {product_name}")
        print("-" * 60)

        # Check dimensions.json
        dimensions_path = os.path.join(product_folder, "dimensions.json")
        if not os.path.exists(dimensions_path):
            print(f"  SKIP: No dimensions.json found")
            fail_count += 1
            continue

        # Read dimensions
        try:
            dimensions = read_dimensions_json(dimensions_path)
            jewelry_height_cm = dimensions['height']
            jewelry_width_cm = dimensions['width']
            print(f"  Jewelry: {jewelry_height_cm:.2f} cm (H) x {jewelry_width_cm:.2f} cm (W)")
        except Exception as e:
            print(f"  SKIP: Error reading dimensions: {e}")
            fail_count += 1
            continue

        # Get jewelry images
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(product_folder, ext)))

        if not image_files:
            print(f"  SKIP: No images found")
            fail_count += 1
            continue

        image_files.sort()

        if len(image_files) < 4:
            print(f"  SKIP: Need at least 4 images, found {len(image_files)}")
            fail_count += 1
            continue

        jewelry_image_path = image_files[3]
        print(f"  Jewelry image: {os.path.basename(jewelry_image_path)}")

        # Composite earring
        output_path = composite_earring(
            base_image_path=REFERENCE_IMAGE_PATH,
            jewelry_image_path=jewelry_image_path,
            earlobe_coords=earlobe_coords,
            px_per_cm=px_per_cm,
            jewelry_height_cm=jewelry_height_cm,
            jewelry_width_cm=jewelry_width_cm,
            output_dir=product_folder
        )

        if output_path:
            print(f"  SUCCESS: {output_path}")
            success_count += 1
        else:
            print(f"  FAILED: Compositing error")
            fail_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total folders: {len(product_folders)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
