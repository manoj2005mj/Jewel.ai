import os
import json
import shutil
import mimetypes
from PIL import Image
from google import genai
from google.genai import types
import dotenv

# --- Configuration ---
INPUT_DIR = "Intialtest"
OUTPUT_DIR = "output_jewels"
REFERENCE_GIRL_IMAGE = "reference.jpeg"

dotenv.load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    API_KEY = "AIzaSyCMHHecQE9SFW1gnZCFFvqv-3F04rRLNjQ"

MODEL_NAME = "gemini-3-pro-image-preview"


def setup_client():
    return genai.Client(api_key=API_KEY)


def process_folder(folder_path, client, output_base):
    folder_name = os.path.basename(folder_path)
    print(f"\nProcessing: {folder_name}")

    valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    all_files = os.listdir(folder_path)
    image_files = sorted([f for f in all_files if os.path.splitext(f)[1].lower() in valid_exts])

    if not image_files:
        print(f"  No images found, skipping.")
        return

    # Ground Truth = alphabetically first image (do NOT send to model)
    gt_filename = image_files[0]
    gt_path = os.path.join(folder_path, gt_filename)

    # Remaining = jewelry reference images (send only 1 to avoid IMAGE_RECITATION)
    ref_filenames = image_files[1:2]

    # Read dimensions.json
    json_path = os.path.join(folder_path, "dimensions.json")
    dimensions_info = ""
    if os.path.exists(json_path):
        with open(json_path, 'r') as jf:
            data = json.load(jf)
            if isinstance(data, dict):
                parts = [f"{k}: {v}" for k, v in data.items() if v]
                dimensions_info = ", ".join(parts)

    # Prompt — describe the task without asking to "copy" anything
    prompt_text = (
        f"Here is a photo of a girl and a photo of an earring design. "
        f"The earring has dimensions: {dimensions_info}. "
        f"Generate a new realistic photo of this girl wearing this earring on her ear. "
        f"Keep the same pose, lighting, and skin tone from the girl photo. "
        f"Make the earring look natural and properly sized on her ear."
    )

    # Build contents: prompt + girl image + jewelry images
    contents = [prompt_text]
    opened_images = []

    # Girl reference
    girl_img_path = os.path.abspath(REFERENCE_GIRL_IMAGE)
    girl_img = Image.open(girl_img_path)
    opened_images.append(girl_img)
    contents.append(girl_img)

    # Jewelry references
    for ref_name in ref_filenames:
        img = Image.open(os.path.join(folder_path, ref_name))
        opened_images.append(img)
        contents.append(img)

    print(f"  Sending: 1 girl + {len(ref_filenames)} jewelry refs")

    # Save GT
    save_folder = os.path.join(output_base, folder_name)
    os.makedirs(save_folder, exist_ok=True)
    gt_ext = os.path.splitext(gt_filename)[1]
    shutil.copy2(gt_path, os.path.join(save_folder, f"gt{gt_ext}"))

    # Call API using streaming (from Google AI Studio template)
    try:
        generated_count = 0
        for chunk in client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE', 'TEXT'],
                image_config=types.ImageConfig(
                    aspect_ratio="1:1",
                    image_size="1K"
                ),
            )
        ):
            if chunk.parts is None:
                continue
            for part in chunk.parts:
                if part.inline_data and part.inline_data.data:
                    file_extension = mimetypes.guess_extension(part.inline_data.mime_type) or ".png"
                    save_path = os.path.join(save_folder, f"generated_{generated_count}{file_extension}")
                    with open(save_path, "wb") as f:
                        f.write(part.inline_data.data)
                    print(f"  Saved: {save_path}")
                    generated_count += 1
                elif part.text:
                    print(f"  Model text: {part.text}")

        if generated_count == 0:
            print("  WARNING: No image generated.")

    except Exception as e:
        print(f"  API Error: {e}")
    finally:
        for img in opened_images:
            img.close()


def main():
    base_path = os.path.abspath(INPUT_DIR)
    output_base = os.path.abspath(OUTPUT_DIR)
    os.makedirs(output_base, exist_ok=True)

    client = setup_client()

    folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
    print(f"Found {len(folders)} folders to process.")

    for folder_name in folders:
        process_folder(os.path.join(base_path, folder_name), client, output_base)

    print("\nAll done!")


if __name__ == "__main__":
    main()
