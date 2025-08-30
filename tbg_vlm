

from math import gcd
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from pygments.lexer import default
import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_MODEL_ID = "apple/FastVLM-7B"
IMAGE_TOKEN_INDEX = -200

_tokenizer = None
_model = None


def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    # Convert tensor of shape [batch, height, width, channels] at the batch_index to PIL Image
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img

def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    #if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
    #    image = image.unsqueeze(-1)
    return image


def get_models_dir():
    """Return the active ComfyUI models directory."""
    return os.getenv(
        "COMFYUI_MODELS_PATH",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")),
    )


def get_fastvlm_dir():
    """Return the dedicated FastVLM storage path inside models/."""
    return os.path.join(get_models_dir(), "LLM", "FastVLM")


def ensure_model_installed():
    """Ensure the FastVLM model is available inside ComfyUI models folder."""
    target_dir = get_fastvlm_dir()
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def load_fastvlm():
    """Load tokenizer and model, caching inside ComfyUI/models/LLM/FastVLM"""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        local_model_path = ensure_model_installed()

        print(f"[FastVLM7BNode] Loading {HF_MODEL_ID} into {local_model_path}")

        _tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_ID, trust_remote_code=True, cache_dir=local_model_path
        )
        _model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=local_model_path,
        )
    return _tokenizer, _model


def run_fastvlm7b(pil_img: Image.Image, instruction: str, max_new_tokens: int = 128) -> str:
    tokenizer, model = load_fastvlm()

    # Build prompt
    messages = [{"role": "user", "content": f"<image>\n{instruction}"}]
    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    pre, post = rendered.split("<image>", 1)

    # Encode text
    pre_ids = tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids

    # Insert image token
    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    # Preprocess image
    vision_tower = model.get_vision_tower()
    px = vision_tower.image_processor(images=pil_img, return_tensors="pt")["pixel_values"]
    px = px.to(model.device, dtype=model.dtype)

    # Generate response
    with torch.no_grad():
        output = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=px,
            max_new_tokens=max_new_tokens,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# --------- ComfyUI Node Definition ----------
class FastVLM7BNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "instruction": ("STRING", {"multiline": True}),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 1024}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "AI/FastVLM"

    def inference(self, image, instruction, max_new_tokens):
        # Convert ComfyUI tensor image -> PIL
        pil_img = tensor_to_pil(image, 0)
        #if isinstance(image, list):
        #    image = image[0]
        #pil_img = Image.fromarray((image.cpu().numpy() * 255).astype("uint8"))

        response = run_fastvlm7b(pil_img, instruction, max_new_tokens)
        return (response,)





NODE_CLASS_MAPPINGS["FastVLM7BNode"] = FastVLM7BNode
NODE_DISPLAY_NAME_MAPPINGS["FastVLM7BNode"] = "FastVLM 7B (Apple)"

# Node mapping
