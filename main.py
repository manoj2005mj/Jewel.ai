import csv
import time
import random
import os
from scrape import scrape_and_download, sanitize_filename, DOWNLOAD_ROOT

# Configuration
CSV_FILE = os.path.join(os.path.dirname(__file__), "jewelbox (1).csv")
MAX_LINKS = 150

def main():
    if not os.path.exists(DOWNLOAD_ROOT):
        os.makedirs(DOWNLOAD_ROOT)
    
    # Check if CSV exists
    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file '{CSV_FILE}' not found.")
        return

    processed_count = 0
    summary = []

    print(f"Reading URLs from {CSV_FILE}...")
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            if not rows:
                print("CSV is empty.")
                return
                
            # Determine start index (skip header if present)
            start_idx = 0
            if rows[0][0].startswith('list_') or 'href' in rows[0][0]:
                start_idx = 1
                
            print(f"Processing first {MAX_LINKS} links...")
            
            # Process links
            for row in rows[start_idx:]:
                if processed_count >= MAX_LINKS:
                    break
                
                # Extract URL
                # Priority 1: First column (list_hover href)
                url = row[0].strip()
                
                # Priority 2: Third column (d-flex href) if first is empty
                if not url and len(row) > 2:
                    url = row[2].strip()
                
                # Validation
                if not url or not url.startswith('http'):
                    continue

                # Extract Product Name from 5th column "card__text"
                product_name = ""
                if len(row) > 4:
                    product_name = row[4].strip()
                
                # Fallback to URL part if name is empty
                if not product_name:
                    product_name = url.split('/')[-2] if url.endswith('/') else url.split('/')[-1]

                # Create specific folder name for this product
                safe_name = sanitize_filename(product_name)[:50]
                
                try:
                    # Pass the desired folder name to the scraper
                    scrape_and_download(url, output_folder=safe_name)
                    processed_count += 1
                    summary.append((product_name, "Processed"))
                except Exception as e:
                    print(f"Error processing {url}: {e}")
                    summary.append((product_name, "Failed"))
                
                # Random delay
                if processed_count < MAX_LINKS:
                    delay = random.uniform(2, 5)
                    print(f"Waiting {delay:.1f} seconds...")
                    time.sleep(delay)

    except Exception as e:
        print(f"Error reading CSV: {e}")

    print("\n--- Summary ---")
    for name, status in summary:
        print(f"Product: {name} | Status: {status}")

if __name__ == "__main__":
    main()