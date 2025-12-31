#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


get_ipython().run_line_magic('pip', 'install unzip')


# In[ ]:


get_ipython().system('pip install -U "mineru[all]"')


# In[ ]:


get_ipython().run_line_magic('unzip', '')


# In[ ]:


import os, zipfile


# In[ ]:


def unzip_folder(zip_file_name: str, destination_dir: str):
    # Ensure the destination directory exists (optional)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # Open the zip file in read mode and extract all contents
    try:
        with zipfile.ZipFile(zip_file_name, 'r') as zf:
            zf.extractall(destination_dir)
        print(f"Successfully extracted all files to '{destination_dir}'")
    except zipfile.BadZipFile:
        print(f"Error: '{zip_file_name}' is not a valid zip file or is corrupted.")
    except FileNotFoundError:
        print(f"Error: '{zip_file_name}' not found.")


# In[ ]:


get_ipython().system('gdown -qqq 18jwp7X3wFoVTq75Y4ARjXbgFEVPUwOHm')
get_ipython().run_line_magic('pwd', '')


# In[ ]:


unzip_folder("./ocr-documents.zip", "ocr-documents")


# In[ ]:


get_ipython().run_line_magic('pip', 'install "mineru-vl-utils[transformers]"')


# In[2]:


from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from mineru_vl_utils import MinerUClient
import base64
import io
from PIL import Image, ImageDraw
from IPython.display import display, HTML


# In[3]:


model = Qwen2VLForConditionalGeneration.from_pretrained(
    "opendatalab/MinerU2.5-2509-1.2B", dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    "opendatalab/MinerU2.5-2509-1.2B", use_fast=True
)

client = MinerUClient(backend="transformers", model=model, processor=processor)


# In[4]:


import os
from huggingface_hub import scan_cache_dir

# Scans the cache for the specific model
cache_info = scan_cache_dir()
for repo in cache_info.repos:
    if "MinerU2.5-2509-1.2B" in repo.repo_id:
        # Use repo_path instead of cache_dir
        print(f"Model ID: {repo.repo_id}")
        print(f"Local Path: {repo.repo_path}")


# In[5]:


# @title Compare extraction
def display_extraction(image_path: str, layout_data: list):
    # 1. Define colors for different block types
    color_map = {
        "header": "#FF5733",  # Red-Orange
        "title": "#C70039",  # Dark Red
        "text": "#2E86C1",  # Blue
        "list": "#8E44AD",  # Purple
        "table": "#27AE60",  # Green
        "table_caption": "#16A085",  # Teal
        "figure": "#F39C12",  # Orange
        "default": "#34495E",  # Dark Grey
    }

    try:
        # 2. Load and process the image
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            width, height = img.size

            # Construct the HTML content for the right pane simultaneously
            right_pane_html = ""

            for item in layout_data:
                item_type = item.get("type", "default")
                content = item.get("content")
                bbox = item.get("bbox")  # [x1, y1, x2, y2] normalized

                # --- Draw Bounding Box ---
                if bbox:
                    # Convert normalized coordinates (0-1) to pixel coordinates
                    x1, y1, x2, y2 = bbox
                    rect_coords = [x1 * width, y1 * height, x2 * width, y2 * height]

                    # Get color
                    color = color_map.get(item_type, color_map["default"])

                    # Draw rectangle (outline)
                    draw.rectangle(rect_coords, outline=color, width=3)

                    # Optional: Draw type label slightly above the box
                    # (Simple text drawing, might overlap on dense documents)
                    draw.text(
                        (rect_coords[0], rect_coords[1] - 10), item_type, fill=color
                    )

                # --- Build HTML Content ---
                if content:
                    # Clean escaped dollar signs for display
                    # clean_content = str(content).replace(r"\$", "$")
                    clean_content = (
                        str(content)
                        .replace(r"\(", "")
                        .replace(r"\)", "")
                        .replace(r"\%", "%")
                    )

                    # clean_content = content

                    if item_type == "title":
                        right_pane_html += f"<h2>{clean_content}</h2>"
                    elif item_type == "header":
                        right_pane_html += (
                            f"<h4 style='margin-bottom:0;'>{clean_content}</h4>"
                        )
                    elif item_type == "table":
                        # Tables usually come as raw HTML in these models
                        right_pane_html += f"<div style='overflow-x:auto; margin: 10px 0;'>{clean_content}</div>"
                    elif item_type == "table_caption":
                        right_pane_html += f"<p><strong>{clean_content}</strong></p>"
                    else:
                        # Standard text
                        right_pane_html += f"<p>{clean_content}</p>"

            # 3. Convert processed PIL image to Base64 string
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_uri = f"data:image/png;base64,{img_str}"

    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # 4. Construct HTML Template
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
            height: 80vh; /* Fixed height with scroll for text pane */
            overflow-y: auto;
        }}
        /* Table Styling */
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
      <!-- Left Pane: Image with Bounding Boxes -->
      <div class="pane pane-left">
        <img src="{image_uri}">
      </div>

      <!-- Right Pane: Extracted Text/HTML -->
      <div class="pane markdown-body">
        {right_pane_html}
      </div>
    </div>
    """

    display(HTML(html_template))


# In[6]:


from PIL import Image


# In[7]:


img_path = "ocr-documents/nvidia-first-page.jpg"
extracted_blocks = client.two_step_extract(Image.open(img_path))


# In[8]:


display_extraction(img_path, extracted_blocks)


# In[9]:


img_path = "ocr-documents/nvidia-inner-page.jpg"
extracted_blocks = client.two_step_extract(Image.open(img_path))


# In[10]:


display_extraction(img_path, extracted_blocks)


# In[11]:


img_path = "ocr-documents/receipt.jpg"
extracted_blocks = client.two_step_extract(Image.open(img_path))


# In[16]:


import json
import os

# Define the config directory (MinerU looks here by default)
config_path = os.path.expanduser("~/.config/magic-pdf.json")
os.makedirs(os.path.dirname(config_path), exist_ok=True)

# Set device-mode to "cuda"
config_data = {
    "device-mode": "cuda",
    "models-dir": "C:\Users\rohan\.cache\huggingface\hub\models--opendatalab--MinerU2.5-2509-1.2B  # Ensure this points to where you downloaded MinerU models
}

with open(config_path, "w") as f:
    json.dump(config_data, f)


# In[12]:


display_extraction(img_path, extracted_blocks)


# In[13]:


img_path = "ocr-documents/id-card.png"
extracted_blocks = client.two_step_extract(Image.open(img_path))


# In[14]:


display_extraction(img_path, extracted_blocks)


# 
# References:
# 
# - Model: https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B
# 

# In[15]:


get_ipython().system('pip install -U "mineru[all]"')


# In[17]:


import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Current Device: {torch.cuda.get_device_name(0)}")


# In[ ]:


# Process a PDF using the GPU
get_ipython().system('CUDA_VISIBLE_DEVICES=0 mineru -p /path/to/input.pdf -o /path/to/output/')


# In[ ]:




