# KyZer AI Image Generator

A Windows desktop text-to-image app using a local Stable Diffusion model.

## First run
The first generation downloads the model from Hugging Face and stores it under `%USERPROFILE%\\.kyzer_ai\\model`. Later generations can run without downloading the model again.

## Important
This app runs inference locally on the user's CPU/GPU. It does not provide a cloud generation quota. Performance depends on the PC hardware.

The included GitHub Actions workflow builds a Windows executable package.
