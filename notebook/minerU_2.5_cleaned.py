#!/usr/bin/env python
"""
MinerU 2.5 Document Extraction Script.

Extracts text, tables, and layout information from document images and PDFs
using the MinerU2.5-2509-1.2B vision-language model.

Features:
- PDF and image processing
- Automatic figure/diagram description using VLM
- HTML output with bounding boxes and extracted content
"""

import base64
import io
import logging
import os
import time
import traceback
import zipfile
from dataclasses import dataclass

import pypdfium2 as pdfium
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


@dataclass
class ExtractionContext:
    """Context for document extraction with model references."""

    client: MinerUClient
    model: Qwen2VLForConditionalGeneration
    processor: AutoProcessor
    describe_figures: bool = True


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


def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
    """Convert PDF pages to PIL Images using pypdfium2."""
    images = []
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        n_pages = len(pdf)
        logger.info(f"PDF has {n_pages} pages")

        for i in range(n_pages):
            page = pdf[i]
            scale = dpi / 72
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            images.append(pil_image)
            logger.info(f"Converted page {i + 1}/{n_pages} to image")

        pdf.close()
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")

    return images


def describe_figure(
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    prompt: str = "Describe this diagram or figure in detail. Include any text, labels, arrows, and the relationships between elements.",
) -> str:
    """Use VLM to describe a figure/diagram.

    Args:
        model: Qwen2VL model
        processor: AutoProcessor for the model
        image: PIL Image of the figure
        prompt: Prompt for description

    Returns:
        Text description of the figure
    """
    try:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0] if output_text else ""
    except Exception as e:
        logger.error(f"Error describing figure: {e}")
        return ""


def crop_figure(
    image: Image.Image, bbox: list[float], padding: int = 10
) -> Image.Image:
    """Crop a figure from an image based on normalized bbox coordinates.

    Args:
        image: Source PIL Image
        bbox: Normalized coordinates [x1, y1, x2, y2] in range 0-1
        padding: Pixels to add around the crop

    Returns:
        Cropped PIL Image
    """
    width, height = image.size
    x1, y1, x2, y2 = bbox

    # Convert normalized to pixel coordinates
    left = max(0, int(x1 * width) - padding)
    top = max(0, int(y1 * height) - padding)
    right = min(width, int(x2 * width) + padding)
    bottom = min(height, int(y2 * height) + padding)

    return image.crop((left, top, right, bottom))


def display_extraction(
    image_or_path: str | Image.Image,
    layout_data: list,
    output_name: str | None = None,
    ctx: ExtractionContext | None = None,
):
    """Process extraction results and save to HTML file.

    Args:
        image_or_path: Either a file path string or a PIL Image object
        layout_data: List of extracted blocks with type, content, bbox
        output_name: Optional output filename (without extension)
        ctx: ExtractionContext with model/processor for figure description
    """
    color_map = {
        "header": "#FF5733",
        "title": "#C70039",
        "text": "#2E86C1",
        "list": "#8E44AD",
        "table": "#27AE60",
        "table_caption": "#16A085",
        "figure": "#F39C12",
        "figure_caption": "#E67E22",
        "image": "#9B59B6",  # Purple for images/diagrams
        "default": "#34495E",
    }

    try:
        # Determine output name and load image
        if isinstance(image_or_path, str):
            display_name = os.path.basename(image_or_path)
            output_name = output_name or os.path.splitext(display_name)[0]
            img = Image.open(image_or_path).convert("RGB")
        else:
            display_name = output_name or "image"
            output_name = output_name or "output"
            img = image_or_path.convert("RGB")

        with Timer(f"Display extraction for {display_name}"):
            # Create a copy for drawing
            img_draw = img.copy()
            draw = ImageDraw.Draw(img_draw)
            width, height = img.size

            right_pane_html = ""
            figure_count = 0

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

                # Handle figure and image blocks - describe them using VLM
                # MinerU classifies: "figure" for charts/graphs, "image" for diagrams/photos
                if item_type in ("figure", "image") and bbox and ctx and ctx.describe_figures:
                    figure_count += 1
                    logger.info(f"Describing figure {figure_count}...")

                    with Timer(f"Describe figure {figure_count}"):
                        # Crop the figure region
                        figure_img = crop_figure(img, bbox)

                        # Generate description using VLM
                        description = describe_figure(
                            ctx.model, ctx.processor, figure_img
                        )

                    if description:
                        # Create figure HTML with embedded image and description
                        fig_buffered = io.BytesIO()
                        figure_img.save(fig_buffered, format="PNG")
                        fig_str = base64.b64encode(fig_buffered.getvalue()).decode(
                            "utf-8"
                        )
                        fig_uri = f"data:image/png;base64,{fig_str}"

                        right_pane_html += f"""
                        <div style='border: 2px solid {color_map["figure"]}; padding: 10px; margin: 10px 0; border-radius: 8px;'>
                            <h4 style='color: {color_map["figure"]}; margin-top: 0;'>Figure {figure_count}</h4>
                            <img src='{fig_uri}' style='max-width: 100%; height: auto; margin-bottom: 10px;'>
                            <p style='background: #fff8e1; padding: 10px; border-radius: 4px;'><strong>Description:</strong> {description}</p>
                        </div>
                        """
                    continue  # Skip normal content handling for figures

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
                        right_pane_html += f"<p><strong>{clean_content}</strong></p>"
                    elif item_type == "figure_caption":
                        right_pane_html += f"<p style='font-style: italic; color: #666;'>{clean_content}</p>"
                    else:
                        right_pane_html += f"<p>{clean_content}</p>"

            buffered = io.BytesIO()
            img_draw.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_uri = f"data:image/png;base64,{img_str}"

    except FileNotFoundError:
        logger.error(f"Image file not found at '{image_or_path}'")
        return
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()
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

    output_path = f"output_{output_name}.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{output_name}</title></head><body>{html_template}</body></html>"
        )
    logger.info(f"Saved extraction to: {output_path}")


def process_image(ctx: ExtractionContext, img_path: str):
    """Process a single image and display extraction results."""
    logger.info(f"Processing image: {img_path}")

    with Timer(f"Extraction for {os.path.basename(img_path)}"):
        extracted_blocks = ctx.client.two_step_extract(Image.open(img_path))

    logger.info(
        f"Extracted {len(extracted_blocks)} blocks from {os.path.basename(img_path)}"
    )

    # Count figures
    figure_count = sum(1 for b in extracted_blocks if b.get("type") == "figure")
    if figure_count > 0:
        logger.info(f"Found {figure_count} figures to describe")

    display_extraction(img_path, extracted_blocks, ctx=ctx)


def process_pdf(ctx: ExtractionContext, pdf_path: str, dpi: int = 200):
    """Process a PDF file by extracting each page.

    Args:
        ctx: ExtractionContext with client and model
        pdf_path: Path to the PDF file
        dpi: DPI for rendering PDF pages (default 200)
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    logger.info(f"Processing PDF: {pdf_path}")

    with Timer(f"PDF to images for {pdf_name}"):
        images = pdf_to_images(pdf_path, dpi=dpi)

    if not images:
        logger.error(f"No images extracted from PDF: {pdf_path}")
        return

    total_blocks = 0
    total_figures = 0

    for i, img in enumerate(images):
        page_num = i + 1
        page_name = f"{pdf_name}_page{page_num}"
        logger.info(f"Processing page {page_num}/{len(images)}")

        with Timer(f"Extraction for {page_name}"):
            extracted_blocks = ctx.client.two_step_extract(img)

        total_blocks += len(extracted_blocks)
        # Count both "figure" and "image" blocks for VLM description
        figure_count = sum(1 for b in extracted_blocks if b.get("type") in ("figure", "image"))
        total_figures += figure_count

        # Log all block types found for debugging
        block_types = {}
        for b in extracted_blocks:
            btype = b.get("type", "unknown")
            block_types[btype] = block_types.get(btype, 0) + 1
        logger.info(f"Block types on page {page_num}: {block_types}")

        logger.info(
            f"Extracted {len(extracted_blocks)} blocks ({figure_count} figures) from page {page_num}"
        )

        display_extraction(img, extracted_blocks, output_name=page_name, ctx=ctx)

    logger.info(
        f"PDF {pdf_name}: {len(images)} pages, {total_blocks} blocks, {total_figures} figures"
    )


def process_file(ctx: ExtractionContext, file_path: str):
    """Process a file (image or PDF) and extract content.

    Args:
        ctx: ExtractionContext with client and model
        file_path: Path to the file (image or PDF)
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        process_pdf(ctx, file_path)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]:
        process_image(ctx, file_path)
    else:
        logger.warning(f"Unsupported file type: {ext} for {file_path}")


if __name__ == "__main__":
    total_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("MinerU 2.5 Document Extraction with Figure Description")
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

    # Create extraction context
    ctx = ExtractionContext(
        client=client,
        model=model,
        processor=processor,
        describe_figures=True,  # Enable figure description
    )

    # Test on sample files (images and PDFs)
    try:
        test_files = [
            "ocr-documents/graph_basics.pdf",
            "ocr-documents/loss_cheat_sheet.png",
            "ocr-documents/nvidia-first-page.jpg",
            "ocr-documents/nvidia-inner-page.jpg",
            "ocr-documents/receipt.jpg",
            "ocr-documents/id-card.png",
        ]

        for file_path in test_files:
            if os.path.exists(file_path):
                process_file(ctx, file_path)
            else:
                logger.warning(f"File not found: {file_path}")

    except Exception as e:
        logger.error(f"An error occurred during extraction or display: {e}")
        traceback.print_exc()

    total_elapsed = time.perf_counter() - total_start
    logger.info("=" * 60)
    logger.info(f"[PROFILE] Total execution time: {total_elapsed:.2f}s")
    logger.info("=" * 60)
