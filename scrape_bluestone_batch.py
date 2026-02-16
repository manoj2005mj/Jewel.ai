import os
import time
import requests
import concurrent.futures
from urllib.parse import urlparse, unquote
from playwright.sync_api import sync_playwright
import collections
import csv
import random
import json
import re

# Configuration
CSV_FILE = "bluestone.csv"
DOWNLOAD_ROOT = "downloaded_images_bluestone"
MAX_LINKS = 100  # Process first 100 links

def upgrade_url_quality(url):
    """
    Attempt to upgrade image URL quality by modifying transformation parameters.
    Specifically targets patterns like 'w_176' -> 'w_2400'.
    """
    # Pattern 1: Path segments like /w_176/ or ,w_176,
    new_url = re.sub(r'w_\d+', 'w_2400', url)
    # Pattern 2: Query parameters ?width=100
    new_url = re.sub(r'width=\d+', 'width=2400', new_url)
    return new_url

def sanitize_filename(name):
    """Clean string for use as a folder or filename."""
    return "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')

def download_image(url, folder):
    """Download a single image to the specified folder."""
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        response.raise_for_status()

        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            ext = '.jpg'

        # Generate filename from URL
        filename = os.path.basename(parsed.path)
        filename = unquote(filename)
        
        if not filename:
            filename = f"image_{int(time.time()*1000)}{ext}"
        
        filepath = os.path.join(folder, filename)
        
        # Avoid overwriting
        counter = 1
        base_filepath = filepath
        while os.path.exists(filepath):
            name, ext = os.path.splitext(base_filepath)
            filepath = f"{name}_{counter}{ext}"
            counter += 1

        with open(filepath, 'wb') as f:
            f.write(response.content)
            
        print(f"  ✓ Saved: {os.path.basename(filepath)}")
        return True
    except Exception as e:
        print(f"  ✗ Failed {url}: {e}")
        return False

def scrape_product(url, product_name, p_browser):
    """Scrapes a single product URL."""
    print(f"\nProcessing Product: {product_name}")
    print(f"URL: {url}")

    # Create folder named after product name
    folder_name = sanitize_filename(product_name)[:100]
    save_folder = os.path.join(DOWNLOAD_ROOT, folder_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    context = p_browser.new_context(
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        viewport={'width': 1920, 'height': 1080}
    )
    page = context.new_page()

    try:
        page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["font", "stylesheet"] else route.continue_())
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        
        # Scroll
        for _ in range(3):
            page.mouse.wheel(0, 1000)
            time.sleep(0.3)
        
        # Extract Images
        image_elements = page.query_selector_all('img')
        img_urls = set()
        potential_urls = []
        
        for img in image_elements:
            # Upgrade quality logic
            src = img.get_attribute('data-high-res') or img.get_attribute('data-zoom-image') or img.get_attribute('data-full-src') or img.get_attribute('src') or img.get_attribute('data-src')
            
            srcset = img.get_attribute('srcset')
            if srcset:
                try:
                    candidates = []
                    for entry in srcset.split(','):
                        entry = entry.strip()
                        parts = entry.split(' ')
                        url_part = parts[0]
                        width = 0
                        if len(parts) > 1 and 'w' in parts[1]:
                            width = int(parts[1].replace('w', ''))
                        candidates.append((width, url_part))
                    best_candidate = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
                    if not src:
                        src = best_candidate
                    elif len(candidates) > 0:
                         if sorted(candidates, key=lambda x: x[0], reverse=True)[0][0] > 500:
                             src = best_candidate
                except:
                    pass
            
            if src:
                if ' ' in src: src = src.split(' ')[0]
                if src.startswith('//'): src = 'https:' + src
                elif src.startswith('/'):
                    parsed_base = urlparse(url)
                    src = f"{parsed_base.scheme}://{parsed_base.netloc}{src}"
                
                if any(x in src.lower() for x in ['jpg', 'jpeg', 'png', 'webp']):
                    potential_urls.append(src)

        # SKU Detection
        detected_skus = []
        for u in potential_urls:
            path = urlparse(u).path
            filename = os.path.basename(path)
            if '_' in filename:
                parts = filename.split('_')
                candidate = parts[0]
                if len(candidate) > 4 and all(c.isalnum() or c == '-' for c in candidate):
                    detected_skus.append(candidate)

        if detected_skus:
            target_sku = collections.Counter(detected_skus).most_common(1)[0][0]
            print(f"  Detected likely SKU: '{target_sku}'")
            for u in potential_urls:
                if target_sku in u:
                    img_urls.add(upgrade_url_quality(u))
        else:
            print("  No obvious SKU pattern found. Downloading all candidates.")
            for u in potential_urls:
                img_urls.add(upgrade_url_quality(u))
        
        print(f"  Found {len(img_urls)} images.")

        # Extract Dimensions
        print("  Extracting dimensions...")
        dimensions = {}
        try:
            full_text = page.inner_text("body")
            def search_dim(keywords, text):
                for keyword in keywords:
                    pattern = re.compile(rf"{keyword}[^0-9]{{0,100}}?(\d+(?:\.\d+)?)\s*(mm|cm)", re.IGNORECASE | re.DOTALL)
                    match = pattern.search(text)
                    if match: return f"{match.group(1)} {match.group(2)}"
                return None

            dimensions['height'] = search_dim(['Height', 'Length'], full_text)
            dimensions['breadth'] = search_dim(['Width', 'Breadth'], full_text)
            dimensions['depth'] = search_dim(['Depth', 'Thickness'], full_text)
            
            print(f"  Dimensions: {dimensions}")
            with open(os.path.join(save_folder, "dimensions.json"), 'w') as f:
                json.dump(dimensions, f, indent=4)
        except Exception as e:
            print(f"  Error dimensions: {e}")

        # Download Images
        print(f"  Downloading {len(img_urls)} images...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(download_image, u, save_folder) for u in img_urls]
            concurrent.futures.wait(futures)

    except Exception as e:
        print(f"  Error processing {url}: {e}")
    finally:
        context.close()

def main():
    if not os.path.exists(DOWNLOAD_ROOT):
        os.makedirs(DOWNLOAD_ROOT)
        
    links_to_process = []
    
    # Read CSV
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader) # Skip header
        
        # Based on user description: "3 colum" -> index 2 (0-based) contains href?
        # Looking at file: "p-name href" is indeed the 3rd column (index 2).
        # "p-name" is index 1.
        
        count = 0
        for row in reader:
            if count >= MAX_LINKS:
                break
            
            if len(row) > 2:
                p_name = row[1]
                p_url = row[2]
                
                if p_url and p_url.startswith('http'):
                    links_to_process.append((p_name, p_url))
                    count += 1
    
    print(f"Found {len(links_to_process)} links to process.")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for name, link in links_to_process:
            scrape_product(link, name, browser)
        browser.close()

if __name__ == "__main__":
    main()
