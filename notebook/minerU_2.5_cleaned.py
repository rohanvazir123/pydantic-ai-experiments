#!/usr/bin/env python
"""
MinerU 2.5 Document Extraction Script.

Extracts text, tables, and layout information from document images
using the MinerU2.5-2509-1.2B vision-language model.
"""

import base64
import io
import logging
import os
import time
import traceback
import zipfile

from huggingface_hub import scan_cache_dir
from mineru_vl_utils import MinerUClient
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        logger.info(f"[PROFILE] {self.name}: {self.elapsed:.2f}s")


def check_gpu() -> bool:
    """Check if CUDA GPU is available."""
    import torch

    if torch.cuda.is_available():
        logger.info(
            f"CUDA is available. Current device: {torch.cuda.get_device_name(0)}"
        )
        return True
    else:
        logger.warning("CUDA is not available. Using CPU.")
        return False


def configure_magic_pdf() -> str:
    """Configure Magic PDF for GPU usage."""
    import json

    config_path = os.path.expanduser("~/.config/magic-pdf.json")

    if os.path.exists(config_path):
        logger.info(f"Config file already exists at: {config_path}")
        return config_path

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    config_data = {
        "device-mode": "cuda",
        "models-dir": "C:\\Users\\rohan\\.cache\\huggingface\\hub\\models--opendatalab--MinerU2.5-2509-1.2B",
    }

    with open(config_path, "w") as f:
        json.dump(config_data, f)

    logger.info(f"Created config file at: {config_path}")
    return config_path


def unzip_folder(zip_file_name: str, destination_dir: str):
    """Extract zip file to destination directory."""
    if os.path.exists(destination_dir) and os.listdir(destination_dir):
        logger.info(
            f"Directory '{destination_dir}' already exists. Skipping extraction."
        )
        return

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    try:
        with Timer("Unzip"):
            with zipfile.ZipFile(zip_file_name, "r") as zf:
                zf.extractall(destination_dir)
        logger.info(f"Successfully extracted all files to '{destination_dir}'")
    except zipfile.BadZipFile:
        logger.error(
            f"Error: '{zip_file_name}' is not a valid zip file or is corrupted."
        )
    except FileNotFoundError:
        logger.error(f"Error: '{zip_file_name}' not found.")


def scan_mineru_cache() -> bool:
    """Scan cache for MinerU model."""
    cache_info = scan_cache_dir()
    if not cache_info.repos:
        logger.warning("No cached models found.")
        return False
    for repo in cache_info.repos:
        if "MinerU2.5-2509-1.2B" in repo.repo_id:
            logger.info(f"Model ID: {repo.repo_id}")
            logger.info(f"Local Path: {repo.repo_path}")
            return True
    return False


def display_extraction(image_path: str, layout_data: list):
    """Process extraction results and save to HTML file."""
    color_map = {
        "header": "#FF5733",
        "title": "#C70039",
        "text": "#2E86C1",
        "list": "#8E44AD",
        "table": "#27AE60",
        "table_caption": "#16A085",
        "figure": "#F39C12",
        "default": "#34495E",
    }

    try:
        with Timer(f"Display extraction for {os.path.basename(image_path)}"):
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                draw = ImageDraw.Draw(img)
                width, height = img.size

                right_pane_html = ""

                for item in layout_data:
                    item_type = item.get("type", "default")
                    content = item.get("content")
                    bbox = item.get("bbox")

                    if bbox:
                        x1, y1, x2, y2 = bbox
                        rect_coords = [x1 * width, y1 * height, x2 * width, y2 * height]
                        color = color_map.get(item_type, color_map["default"])
                        draw.rectangle(rect_coords, outline=color, width=3)
                        draw.text(
                            (rect_coords[0], rect_coords[1] - 10), item_type, fill=color
                        )

                    if content:
                        clean_content = (
                            str(content)
                            .replace(r"\(", "")
                            .replace(r"\)", "")
                            .replace(r"\%", "%")
                        )

                        if item_type == "title":
                            right_pane_html += f"<h2>{clean_content}</h2>"
                        elif item_type == "header":
                            right_pane_html += (
                                f"<h4 style='margin-bottom:0;'>{clean_content}</h4>"
                            )
                        elif item_type == "table":
                            right_pane_html += f"<div style='overflow-x:auto; margin: 10px 0;'>{clean_content}</div>"
                        elif item_type == "table_caption":
                            right_pane_html += (
                                f"<p><strong>{clean_content}</strong></p>"
                            )
                        else:
                            right_pane_html += f"<p>{clean_content}</p>"

                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_uri = f"data:image/png;base64,{img_str}"

    except FileNotFoundError:
        logger.error(f"Image file not found at '{image_path}'")
        return
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return

    html_template = f"""
    <style>
        .container {{
            display: flex;
            align-items: flex-start;
            width: 100%;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        }}
        .pane {{
            flex: 1;
            padding: 15px;
            max-width: 50%;
            box-sizing: border-box;
        }}
        .pane-left {{
            background-color: #f9f9f9;
            border-right: 1px solid #e0e0e0;
            text-align: center;
        }}
        .pane img {{
            width: 100%;
            height: auto;
            object-fit: contain;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .markdown-body {{
            font-size: 14px;
            line-height: 1.6;
            height: 80vh;
            overflow-y: auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 12px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 6px;
            text-align: left;
        }}
        th {{
            font-weight: bold;
        }}
    </style>

    <div class="container">
      <div class="pane pane-left">
        <img src="{image_uri}">
      </div>
      <div class="pane markdown-body">
        {right_pane_html}
      </div>
    </div>
    """

    output_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"output_{output_name}.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{output_name}</title></head><body>{html_template}</body></html>"
        )
    logger.info(f"Saved extraction to: {output_path}")


def process_image(client: MinerUClient, img_path: str):
    """Process a single image and display extraction results."""
    logger.info(f"Processing: {img_path}")

    with Timer(f"Extraction for {os.path.basename(img_path)}"):
        extracted_blocks = client.two_step_extract(Image.open(img_path))

    logger.info(
        f"Extracted {len(extracted_blocks)} blocks from {os.path.basename(img_path)}"
    )
    display_extraction(img_path, extracted_blocks)


if __name__ == "__main__":
    total_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("MinerU 2.5 Document Extraction")
    logger.info("=" * 60)

    # Check GPU availability
    if not check_gpu():
        logger.error("GPU not available or not properly configured.")
        exit(1)

    # Configure Magic PDF for GPU usage
    if not configure_magic_pdf():
        logger.error("Failed to configure Magic PDF.")
        exit(1)

    # Scan MinerU cache
    if not scan_mineru_cache():
        logger.error(
            "MinerU model not found in cache. Please download the model first."
        )
        exit(1)

    # Unzip OCR documents
    unzip_folder("./ocr-documents.zip", "ocr-documents")

    # Load model and processor
    with Timer("Load model"):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "opendatalab/MinerU2.5-2509-1.2B", dtype="auto", device_map="auto"
        )
    if model is None:
        logger.error("Failed to load MinerU model.")
        exit(1)

    with Timer("Load processor"):
        processor = AutoProcessor.from_pretrained(
            "opendatalab/MinerU2.5-2509-1.2B", use_fast=True
        )
    if processor is None:
        logger.error("Failed to load MinerU processor.")
        exit(1)

    # Initialize MinerU client
    with Timer("Initialize client"):
        client = MinerUClient(backend="transformers", model=model, processor=processor)
    if not client:
        logger.error("Failed to initialize MinerU client.")
        exit(1)

    # Test on sample images
    try:
        test_images = [
            "ocr-documents/nvidia-first-page.jpg",
            "ocr-documents/nvidia-inner-page.jpg",
            "ocr-documents/receipt.jpg",
            "ocr-documents/id-card.png",
        ]

        for img_path in test_images:
            if os.path.exists(img_path):
                process_image(client, img_path)
            else:
                logger.warning(f"Image not found: {img_path}")

    except Exception as e:
        logger.error(f"An error occurred during extraction or display: {e}")
        traceback.print_exc()

    total_elapsed = time.perf_counter() - total_start
    logger.info("=" * 60)
    logger.info(f"[PROFILE] Total execution time: {total_elapsed:.2f}s")
    logger.info("=" * 60)
