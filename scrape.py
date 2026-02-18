import os
import time
import requests
import concurrent.futures
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
import collections
import csv
import random

# Configuration
URLS = [
    "https://jewelbox.co.in/mini-round-distinct-diamond-ear-studs/",
    "https://jewelbox.co.in/briget-ribbon-knot-diamond-earrings/"
    # Add more URLs here
]
DOWNLOAD_ROOT = os.path.join(os.path.dirname(__file__), "downloaded_images")

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

        # Generate filename from URL hash or counter (handled by caller ideally, checking duplication here basic)
        filename = os.path.basename(parsed.path)
        if not filename or len(filename) > 30:
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
                src = img.get_attribute('src') or img.get_attribute('data-src') or img.get_attribute('srcset')
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

            # Strategy: Find common prefixes among images to identify the "hero" set
            

            # 1. Extract filenames
            filenames = [os.path.basename(urlparse(u).path) for u in potential_urls]
            
            # 2. Find the most common prefix of sufficient length (e.g. > 4 chars)
            # We'll look for chunks of characters that appear in multiple images
            
            def get_longest_prefix(s1, s2):
                min_len = min(len(s1), len(s2))
                for i in range(min_len):
                    if s1[i] != s2[i]:
                        return s1[:i]
                return s1[:min_len]

            # Compare adjacent sorted filenames to find clusters
            sorted_files = sorted(filenames)
            prefixes = []
            for i in range(len(sorted_files) - 1):
                p = get_longest_prefix(sorted_files[i], sorted_files[i+1])
                # Filter out obvious UI elements or common non-product prefixes
                if len(p) > 5 and not any(x in p.lower() for x in ['login', 'slide', 'icon', 'logo', 'button', 'banner', 'popup']): 
                    prefixes.append(p)
            
            if prefixes:
                # The most common prefix is likely our product series
                most_common_prefix = collections.Counter(prefixes).most_common(1)[0][0]
                print(f"  Detected pattern prefix: '{most_common_prefix}'")
                
                # Filter URLs that match this prefix
                for u in potential_urls:
                    fname = os.path.basename(urlparse(u).path)
                    if fname.startswith(most_common_prefix):
                        img_urls.add(u)
                
                # SPECIAL HANDLING FOR "DIMM" / ALL VARIATIONS OF THE SKU
                # Previous logic tried to be smart about "dimm".
                # User now requests: "pick all the source urls where everer id is same"
                # e.g. if ID is WER0125, grab EVERYTHING containing WER0125.
                
                # 1. Extract the core SKU from the most_common_prefix
                # Heuristic: Remove common separators and trailing numbers
                # e.g. "1759829997_PER0730" -> "PER0730"
                # e.g. "WER0297" -> "WER0297"
                
                clean_prefix = most_common_prefix.strip('-_')
                parts = clean_prefix.split('_')
                
                # Attempt to find the part that looks like an SKU (alphanumeric, often uppercase)
                # If patterns are like 12345_SKU, usually the last part is the distinct SKU
                sku_part = parts[-1] if len(parts) > 1 else clean_prefix
                
                # Remove common trailing markers like 'front', 'side', 'back' if they accidentally got into the prefix
                for suffix in ['front', 'side', 'back', 'top', 'dimm']:
                    if sku_part.lower().endswith(suffix):
                        sku_part = sku_part[:-len(suffix)]
                
                sku_part = sku_part.strip('-_')

                if len(sku_part) > 3:
                    print(f"  Broad search for any image containing SKU: '{sku_part}'")
                    for u in potential_urls:
                        # Check if the filename contains the SKU
                        # We use basename to avoid matching random parts of the path structure
                        fname = os.path.basename(urlparse(u).path).lower()
                        if sku_part.lower() in fname:
                             img_urls.add(u)
                        else:
                            # Sometimes the SKU might be embedded in the path too? 
                            # User said "where everer id is same"
                            pass

            else:
                # Fallback to grabbing everything if no pattern found
                print("  No obvious naming pattern found, downloading all valid images.")
                img_urls.update(potential_urls)
            
            print(f"  Found {len(img_urls)} images matching the hero section pattern.")
            
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
