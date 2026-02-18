# Jewelry Scraper and Virtual Try-On Pipeline

This repository hosts a comprehensive automated pipeline designed to bridge the gap between e-commerce product imagery and realistic virtual try-on experiences. The system integrates  web scraping, computer vision for segmentation, and Generative AI for image synthesis.

## Project Overview

The project consists of three main robust modules:

### 1. Intelligent Scraping Module
Automates the extraction of high-quality product assets from jewelry e-commerce platforms (currently optimized for JewelBox and BlueStone).
- **Dynamic Content Handling**: Uses **Playwright** to navigate complex, lazy-loaded web pages.
- **Smart Asset Extraction**: Automatically identifies high-resolution image URLs, filters out UI elements, and detects product SKUs to organize files.
- **Metadata Retrieval**: Scrapes product dimensions (height, width) to inform the scaling of virtual try-ons.

### 2. Advanced Segmentation (FastSAM + CLIP)
Isolates jewelry items from white-background product photos with high precision.
- **FastSAM Integration**: Uses the Fast Segment Anything Model for robust object detection.
- **Semantic Filtering**: Integrates **OpenAI CLIP** to semantically filter masks, ensuring only "jewelry" or "metal/diamond" objects are selected, rejecting noise like mannequin necks or stands.
- **Quality Metrics**: tools to calculate IoU (Intersection over Union) and centroid alignment to validate segmentation quality.

### 3. Generative Virtual Try-On (Google Gemini)
Synthesizes realistic images of models wearing the scraped jewelry.
- **GenAI Pipeline**: Sends reference model images and segmented jewelry assets to **Google Gemini 1.5 Pro**.
- **Context-Aware Synthesis**: The conceptual prompting system instructs the AI to respect lighting, skin tone, and physical dimensions extracted during the scraping phase for hyper-realistic results.

---

## Prerequisites

Before running any scripts, ensure you have the following installed on your machine:

1.  **Python 3.10+** (Python 3.12 is recommended)
2.  **uv** (Fast Python package installer) or standard `pip`
3.  **Git** (for installing dependencies from repositories)
4.  **Google API Key** (for virtual try-on features)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Scrape
    ```

2.  **Set up a virtual environment (Recommended):**
    Using `uv`:
    ```bash
    uv venv
    source .venv/bin/activate
    ```
    Or using standard python:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Run the provided installation script to install FastSAM, CLIP, and other core libraries:
    ```bash
    ./installsam.sh
    ```
    
    Additionally, install the project's direct dependencies:
    ```bash
    uv add playwright google-genai python-dotenv scikit-image requests
    # Or with pip:
    # pip install playwright google-genai python-dotenv scikit-image requests
    ```

4.  **Install Playwright Browsers:**
    Required for the scraping scripts to work:
    ```bash
    playwright install chromium
    ```

5.  **Model Checkpoints:**
    Ensure `FastSAM-x.pt` is present in the root directory. If not, download it from the [FastSAM repository](https://github.com/CASIA-LMC-Lab/FastSAM) or check if it was downloaded by the install script.

6.  **Environment Variables:**
    Create a `.env` file in the root directory and add your Google API Key:
    ```
    GOOGLE_API_KEY=your_actual_api_key_here
    ```

## Usage

### 1. Scraping Jewelry Images
There are several scripts for different sources:

*   **Scrape JewelBox (Batch):**
    Reads from `jewelbox (1).csv` and downloads images.
    ```bash
    python main.py
    ```

*   **Scrape BlueStone:**
    Run the BlueStone specific scraper.
    ```bash
    python Scrapebs.py
    ```

### 2. Segmentation & Analysis
Tools for segmenting jewelry from images using FastSAM and CLIP.

*   **Run FastSAM on a single image:**
    Segments earrings from a specific image.
    ```bash
    python run_fastsam.py
    ```

*   **Evaluate Segmentation (Head-to-Head):**
    Compares different segmentation approaches.
    ```bash
    python test_headtohead.py
    ```

### 3. Virtual Try-On Generation
Generates a realistic try-on image using Google GenAI.

*   **Process Jewels:**
    Reads images from `Intialtest`, sends them to Gemeni along with a reference girl image (`reference.jpeg`), and saves the result in `output_jewels`.
    ```bash
    python process_jewels.py
    ```

## Project Structure

*   `main.py` - specific usage for scrapping jewelbox urls from csv.
*   `Scrapebs.py` - Scraper for BlueStone.
*   `scrape.py` - General scraping utility functions.
*   `centroidiou.py` - Core logic for mask extraction and IoU calculation.
*   `process_jewels.py` - AI Try-On generation pipeline.
*   `downloaded_images/` - Output folder for scraped images.
*   `output_jewels/` - Output folder for generated try-on images.

## Troubleshooting

*   **"Weights only load failed"**: This project includes a monkey-patch for `torch.load` to handle older model checkpoints on newer PyTorch versions.
*   **Playwright errors**: Make sure you ran `playwright install chromium`.
*   **Missing FastSAM**: If `FastSAM-x.pt` is missing, the scripts will warn you. Download it manually if needed.
