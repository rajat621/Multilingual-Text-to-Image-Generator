# #  Step 1: Install dependencies (run this once before running the rest)
# !pip install -q diffusers transformers accelerate langdetect sentencepiece gradio
# !pip install -q git+https://github.com/TencentARC/GFPGAN.git basicsr facexlib

# !sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/import torchvision.transforms.functional as TF\nrgb_to_grayscale = TF.rgb_to_grayscale/' /usr/local/lib/python*/dist-packages/basicsr/data/degradations.py

import os
import numpy as np
import torch
from PIL import Image
from datetime import datetime
import gradio as gr
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from gfpgan import GFPGANer

#  Configuration
class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    steps = 25
    size = (1024, 1024)
    guidance = 8
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    save_dir = "outputs"
    os.makedirs(save_dir, exist_ok=True)

LANG_CODE_MAP = {
    "kn": "kan_Knda", "hi": "hin_Deva", "ta": "tam_Taml",
    "te": "tel_Telu", "ml": "mal_Mlym", "bn": "ben_Beng",
    "en": "eng_Latn"
}

#  Load NLLB model
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_model.to(CFG.device)

def get_translation(text, target_lang="eng_Latn"):
    detected_lang_code = detect(text)
    src_lang = LANG_CODE_MAP.get(detected_lang_code)
    if not src_lang:
        raise ValueError(f"Unsupported language: {detected_lang_code}")
    translation_pipeline = pipeline(
        "translation",
        model=nllb_model,
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=target_lang,
        max_length=400,
        device=0 if CFG.device == "cuda" else -1
    )
    result = translation_pipeline(text)
    return result[0]["translation_text"], detected_lang_code

#  Load Stable Diffusion XL
def load_model():
    scheduler = EulerDiscreteScheduler.from_pretrained(CFG.model_id, subfolder="scheduler")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        CFG.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16 if CFG.device == "cuda" else torch.float32,
        safety_checker=None,
        use_safetensors=True
    )
    pipe.to(CFG.device)
    return pipe

sd_model = load_model()

#  Face Restoration
def restore_face(img):
    restorer = GFPGANer(
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
        device=CFG.device
    )
    _, _, output = restorer.enhance(np.array(img), has_aligned=False, only_center_face=False, paste_back=True)
    return Image.fromarray(output)

#  Full pipeline
def run_pipeline(user_text, restore_faces=True):
    try:
        translated_prompt, lang = get_translation(user_text)
    except Exception as e:
        return f"Translation failed: {str(e)}", None

    torch.manual_seed(CFG.seed)
    image = sd_model(
        prompt=translated_prompt,
        num_inference_steps=CFG.steps,
        guidance_scale=CFG.guidance
    ).images[0]

    image = image.resize(CFG.size)
    if restore_faces:
        image = restore_face(image)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    save_path = os.path.join(CFG.save_dir, filename)
    image.save(save_path)
    return f"Language: {lang} | Translated Prompt: {translated_prompt}", image

#  Gradio UI
def interface(prompt, restore_faces):
    return run_pipeline(prompt, restore_faces)

gr.Interface(
    fn=interface,
    inputs=[
        gr.Textbox(label="Enter prompt (any language)", placeholder="e.g., ನದಿಯ ದಡದಲ್ಲಿ ನಿಂತಿರುವ ಹುಡುಗ"),
        gr.Checkbox(label="Enhance faces with GFPGAN", value=True)
    ],
    outputs=[
        gr.Textbox(label="Translation Info"),
        gr.Image(label="Generated Image")
    ],
    title="🌍 Multilingual Text-to-Image Generator",
    description="Enter a prompt in any supported language (e.g., Kannada, Hindi, Tamil). It will be translated, rendered with Stable Diffusion XL, and optionally enhanced using GFPGAN."
).launch()
