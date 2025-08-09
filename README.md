# 🌍 Multilingual Text-to-Image Generator

A multilingual text-to-image generation pipeline powered by **Stable Diffusion XL** and **GFPGAN** for face restoration.  
This project supports prompts in multiple Indian languages (**Kannada, Hindi, Tamil, Telugu, Malayalam, Bengali**) as well as English, automatically translating them into English for better image generation.

---

## ✨ Features
- **Automatic Language Detection** – Detects the language of your prompt using `langdetect`.
- **NLLB Translation** – Translates non-English prompts into English using Meta's **NLLB-200** model.
- **Stable Diffusion XL** – Generates high-quality, realistic images from translated prompts.
- **Face Enhancement** – Optionally restores and enhances faces in generated images using **GFPGAN**.
- **Gradio Web UI** – Simple, interactive interface for prompt input and image preview.
- **Multilingual Support** – Works with Kannada, Hindi, Tamil, Telugu, Malayalam, Bengali, and English.

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rajat621/Multilingual-Text-to-Image-Generator.git
cd multilingual-text2image
```
2️⃣ Install Dependencies
Run the following commands once to set up the environment:
```
pip install -q diffusers transformers accelerate langdetect sentencepiece gradio
pip install -q git+https://github.com/TencentARC/GFPGAN.git basicsr facexlib
```
3️⃣ Patch basicsr (GFPGAN requirement)
```
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/import torchvision.transforms.functional as TF\nrgb_to_grayscale = TF.rgb_to_grayscale/' \
/usr/local/lib/python*/dist-packages/basicsr/data/degradations.py
```
🚀 Usage
Run the script:
```
python app.py

```
After launching, Gradio will provide a local and/or public URL in the terminal.<br>
Open it in your browser and start generating images.

## 🛠 How It Works

- **Prompt Input – Enter a prompt in any supported language.
- **Language Detection – The langdetect library identifies the language.
- **Translation – Uses NLLB-200 to translate into English.
- **Image Generation – Stable Diffusion XL generates the image.
- **Optional Face Restoration – GFPGAN enhances face details.
- **Output – Image is displayed in the UI and saved to the outputs/ folder.

## ⚙ Configuration
You can adjust settings in the CFG class inside app.py:
```
steps = 25          # Inference steps
size = (1024, 1024) # Output resolution
guidance = 8        # Guidance scale for Stable Diffusion
seed = 42           # Random seed for reproducibility

```
## 📜 License
This project is for educational and research purposes only.
Follow the licenses of:

- **Stable Diffusion XL
- **NLLB-200
- **GFPGAN
