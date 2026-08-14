# pip install transformers accelerate pillow torch

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# 1. Initialize the device and pre-trained VQA pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Salesforce/blip-vqa-base"

# Processor handles both image resizing and text tokenization
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForQuestionAnswering.from_pretrained(model_id).to(device)

# 2. Prepare raw inputs (Image and Question)
# Replace with your local image path
image_path = "sample_room.jpg"  
raw_image = Image.open(image_path).convert("RGB")
question = "What color is the sofa in the room?"

# 3. Preprocess inputs into joint multimodal tensors
inputs = processor(images=raw_image, text=question, return_tensors="pt").to(device)

# 4. Generate answer tokens using the VLM decoder
with torch.no_grad():
    out = model.generate(**inputs)

# 5. Decode back to human language
answer = processor.decode(out[0], skip_special_tokens=True)
print(f"Question: {question}")
print(f"Answer: {answer}")
