# KyZer AI Image Generator — Lite

Ultra-light local Windows text-to-image app.

## Target

- No API key
- No cloud generation credits
- Unlimited local generation
- CPU-only
- Small model download
- Windows desktop app

## Model

This build uses `IlyasMoutawwakil/tiny-stable-diffusion-onnx`. The model repository is about 9.33 MB and contains a tiny ONNX Stable Diffusion pipeline. It is intentionally much smaller than normal Stable Diffusion models.

Because the model is tiny, image quality is experimental/low compared with full Stable Diffusion. The app generates at 512x512 internally and can upscale the result to the selected output size such as 1280x720.

## First generation

The EXE does not contain the model. On the first generation, the required model files are downloaded from Hugging Face into:

`%USERPROFILE%\\.kyzer_ai\\tiny_model`

The model download is only around 9 MB. After that, generation is local/offline.

## Build

Run `build.bat`.

The GitHub Actions workflow builds a Windows package automatically.
