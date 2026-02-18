import os
import time
import requests
import concurrent.futures
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
import collections
import csv
import random
import json
import re

# Configuration
URLS = [
    "https://www.bluestone.com/earrings/the-elif-multi-pierced-stud-earrings~77272.html?impEvent=browseclick&posEvent=3&sortbyEvent=mostpopular&tagEvent="
    # Add more URLs here
]
DOWNLOAD_ROOT = os.path.join(os.path.dirname(__file__), "downloaded_bluestone")

def upgrade_url_quality(url):
    """
    Attempt to upgrade image URL quality by modifying transformation parameters.
    Specifically targets patterns like 'w_176' -> 'w_2400'.
    """
    # Pattern 1: Path segments like /w_176/ or ,w_176,
    # match w_ followed by digits
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

        # Determine extension
        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            ext = '.jpg'

        # Generate filename from URL
        filename = os.path.basename(parsed.path)
        # Unquote to handle URL encoded characters if any
        from urllib.parse import unquote
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

def scrape_and_download(url, output_folder=None):
    """Scrapes a single product URL and downloads images to output_folder."""
    print(f"\nProcessing: {url}")
    
    with sync_playwright() as p:
        # Launch options to appear more like a real user
        browser = p.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu'
            ]
        )
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()

        try:
            # Navigate with a generous timeout
            # Optimization: Block unnecessary resources if possible to speed up loading
            # But be careful not to block images we need to detect
            page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["font", "stylesheet"] else route.continue_())

            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            
            # Scroll to trigger lazy loading - Optimized
            # Reduced wait time and iterations, increased scroll amount
            for _ in range(3):
                page.mouse.wheel(0, 1000)
                time.sleep(0.3)
            
            # Extract Title for folder name
            title_el = page.query_selector('h1')
            if title_el:
                title = title_el.inner_text().strip()
            else:
                # Fallback to URL part if no H1 found
                title = urlparse(url).path.strip('/').split('/')[-1] or "unknown_product"
            
            # Use pre-determined output_folder if provided, else use title
            if output_folder:
                folder_name = output_folder
            else:
                folder_name = sanitize_filename(title)[:50]
                
            save_folder = os.path.join(DOWNLOAD_ROOT, folder_name)
            
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            
            print(f"  Target Folder: {save_folder}")

            # Extract Images
            # Logic: Get all images, filter for decent size/quality usually product images
            image_elements = page.query_selector_all('img')
            img_urls = set()
            
            potential_urls = []
            for img in image_elements:
                # Prefer high-res attributes often found in e-commerce sites
                src = img.get_attribute('data-high-res') or img.get_attribute('data-zoom-image') or img.get_attribute('data-full-src') or img.get_attribute('src') or img.get_attribute('data-src')
                
                # Check srcset for largest image
                srcset = img.get_attribute('srcset')
                if srcset:
                    # Parse srcset to find the URL with the highest width (e.g. "img.jpg 1024w")
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
                        # Pick the one with largest width
                        best_candidate = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
                        if not src: # If no other src found, or to override? 
                            # Usually srcset provides better quality than default src
                            src = best_candidate
                        elif len(candidates) > 0:
                             # If we found a high-res candidate in srcset, use it prefers over basic src
                             if sorted(candidates, key=lambda x: x[0], reverse=True)[0][0] > 500: # Arbitrary threshold
                                 src = best_candidate
                    except:
                        pass
                
                if src:
                    # Clean up src (sometimes srcset has multiple)
                    if ' ' in src: 
                        src = src.split(' ')[0]
                    
                    # Fix relative URLs
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        parsed_base = urlparse(url)
                        src = f"{parsed_base.scheme}://{parsed_base.netloc}{src}"
                    
                    if any(x in src.lower() for x in ['jpg', 'jpeg', 'png', 'webp']):
                        potential_urls.append(src)

            # Strategy: Identify the product SKU from the image filenames and extract all matching images.
            # User pattern: BISP0076D08_... -> SKU is BISP0076D08
            
            detected_skus = []
            for u in potential_urls:
                path = urlparse(u).path
                filename = os.path.basename(path)
                
                # Check for pattern: SKU is usually the prefix before '_'
                if '_' in filename:
                    parts = filename.split('_')
                    candidate = parts[0]
                    # Filter out obvious non-SKUs (too short, or common UI names)
                    # Allow alphanumeric and hyphens for SKUs
                    if len(candidate) > 4 and all(c.isalnum() or c == '-' for c in candidate):
                        detected_skus.append(candidate)

            if detected_skus:
                # The most frequent prefix is likely our Product SKU
                target_sku = collections.Counter(detected_skus).most_common(1)[0][0]
                print(f"  Detected likely SKU: '{target_sku}'")
                
                # Filter: Keep any image URL that contains this SKU
                for u in potential_urls:
                    if target_sku in u:
                        # Upgrade quality before adding
                        hq_url = upgrade_url_quality(u)
                        img_urls.add(hq_url)
            else:
                print("  No obvious SKU pattern found (prefix before '_'). Downloading all candidates.")
                for u in potential_urls:
                     img_urls.add(upgrade_url_quality(u))
            
            print(f"  Found {len(img_urls)} images matching the SKU pattern.")
            
            # Extract Dimensions
            print("  Extracting dimensions...")
            # Initialize with None or explicit types so type checker is happy if we were using static checking, 
            # but here it's just a dict. The linter error suggests python type inference or similar got confused 
            # because I initialized with values as None.
            dimensions = {}
            
            try:
                # Extract all text from the page to scan for dimensions
                # User requested to search all content to fix null issues
                full_text = page.inner_text("body")
                
                # Helper to find dimension using regex
                def search_dim(keywords, text):
                    for keyword in keywords:
                        # Improved regex:
                        # 1. Matches keyword (case insensitive)
                        # 2. Matches up to 100 non-digit characters (allows for newlines, labels, colons etc.)
                        # 3. Captures the number (integer or decimal)
                        # 4. Captures the unit (mm or cm)
                        pattern = re.compile(
                            rf"{keyword}[^0-9]{{0,100}}?(\d+(?:\.\d+)?)\s*(mm|cm)", 
                            re.IGNORECASE | re.DOTALL
                        )
                        match = pattern.search(text)
                        if match:
                            return f"{match.group(1)} {match.group(2)}"
                    return None

                dimensions['height'] = search_dim(['Height', 'Length'], full_text)
                dimensions['breadth'] = search_dim(['Width', 'Breadth'], full_text)
                dimensions['depth'] = search_dim(['Depth', 'Thickness'], full_text)
                
                # Filter out None values for cleaner JSON? User said "if available".
                # We will keep keys but value as null if not found, or remove them.
                # Let's keep keys to match the requested structure.
                
                print(f"  Dimensions found: {dimensions}")
                
                # Save to JSON
                json_path = os.path.join(save_folder, "dimensions.json")
                with open(json_path, 'w') as f:
                    json.dump(dimensions, f, indent=4)
                    
            except Exception as e:
                print(f"  Error extracting dimensions: {e}")
            
            # Download
            count = 0
            # Optimized: Use ThreadPoolExecutor for concurrent downloads
            print(f"  Starting concurrent download of {len(img_urls)} images...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(download_image, u, save_folder) for u in img_urls]
                for future in concurrent.futures.as_completed(futures):
                    if future.result():
                        count += 1
            
            print(f"  Completed. {count} images downloaded to '{os.path.basename(save_folder)}'.")

        except Exception as e:
            print(f"  Error scraping {url}: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    if not os.path.exists(DOWNLOAD_ROOT):
        os.makedirs(DOWNLOAD_ROOT)
        
    for link in URLS:
        scrape_and_download(link)
