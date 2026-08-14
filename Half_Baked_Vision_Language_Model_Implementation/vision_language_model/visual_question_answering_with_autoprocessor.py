# pip install torch torchvision transformers pillow accelerate

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

print("Loading model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print("Model loaded.")

def visual_question_answering(image_path, question):

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

    answer = processor.batch_decode(
        outputs,
        skip_special_tokens=True,
    )[0]

    return answer

from google.colab import files

uploaded = files.upload()

if __name__ == "__main__":

    image_path = "cat_sample_image.jpg"

    question = "What breed of cat is shown?"

    answer = visual_question_answering(image_path, question)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)