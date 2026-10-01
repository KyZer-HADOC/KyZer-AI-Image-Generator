import os
import threading
import time
import urllib.request
import json
import logging
import platform
import sys
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
from PIL import Image, ImageTk

MODEL_ID = "IlyasMoutawwakil/tiny-stable-diffusion-onnx"
APP_DIR = os.path.join(os.path.expanduser("~"), ".kyzer_ai")
MODEL_DIR = os.path.join(APP_DIR, "tiny_model")
LOG_DIR = os.path.join(APP_DIR, "logs")
OUT_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "KyZer AI")
HF_BASE = f"https://huggingface.co/{MODEL_ID}/resolve/main/"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"kyzer_ai_{time.strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger("KyZerAI")


def log_exception(context, exc=None):
    if exc is not None:
        logger.error("%s: %s", context, exc, exc_info=True)
    else:
        logger.error(context, exc_info=True)


def global_exception_handler(exc_type, exc_value, exc_traceback):
    if exc_type is KeyboardInterrupt:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical(
        "UNHANDLED APPLICATION EXCEPTION",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


sys.excepthook = global_exception_handler

logger.info("KyZer AI started")
logger.info("Log file: %s", LOG_FILE)
logger.info("Python: %s", sys.version.replace("\\n", " "))
logger.info("Platform: %s", platform.platform())
logger.info("Executable: %s", sys.executable)
logger.info("App directory: %s", APP_DIR)


MODEL_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.onnx",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "unet/config.json",
    "unet/model.onnx",
    "vae_decoder/config.json",
    "vae_decoder/model.onnx",
]


def download_model(progress=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    total = len(MODEL_FILES)
    for i, rel in enumerate(MODEL_FILES, 1):
        dst = os.path.join(MODEL_DIR, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            logger.info("Model file already exists: %s (%d bytes)", rel, os.path.getsize(dst))
            if progress:
                progress(i, total, f"Model files: {i}/{total}")
            continue
        url = HF_BASE + rel
        tmp = dst + ".part"
        logger.info("Downloading model file: %s", url)
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dst)
        logger.info("Downloaded: %s (%d bytes)", rel, os.path.getsize(dst))
        if progress:
            progress(i, total, f"Model files: {i}/{total}")


def model_ready():
    return all(os.path.exists(os.path.join(MODEL_DIR, f.replace("/", os.sep))) for f in MODEL_FILES)


class TinyDiffusion:
    def __init__(self, progress=None):
        import onnxruntime as ort
        from transformers import CLIPTokenizer

        self.progress = progress
        logger.info("Loading ONNX Runtime %s", ort.__version__)
        logger.info("Available ONNX providers: %s", ort.get_available_providers())
        download_model(progress)

        self.ort = ort
        self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(MODEL_DIR, "tokenizer"))

        self.text = self._load_session("text_encoder/model.onnx", "text_encoder")
        self.unet = self._load_session("unet/model.onnx", "unet")
        self.vae = self._load_session("vae_decoder/model.onnx", "vae_decoder")
        self.scheduler = self._scheduler()

    def _load_session(self, relative_path, name):
        path = os.path.join(MODEL_DIR, relative_path.replace("/", os.sep))
        logger.info("Loading %s ONNX model: %s", name, path)
        session_options = self.ort.SessionOptions()
        session_options.log_severity_level = 3
        session = self.ort.InferenceSession(
            path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        logger.info(
            "%s inputs: %s",
            name,
            [
                {"name": x.name, "type": x.type, "shape": x.shape}
                for x in session.get_inputs()
            ],
        )
        logger.info(
            "%s outputs: %s",
            name,
            [
                {"name": x.name, "type": x.type, "shape": x.shape}
                for x in session.get_outputs()
            ],
        )
        return session

    def _scheduler(self):
        with open(os.path.join(MODEL_DIR, "scheduler", "scheduler_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        n = int(cfg.get("num_train_timesteps", 1000))
        beta_start = float(cfg.get("beta_start", 0.00085))
        beta_end = float(cfg.get("beta_end", 0.012))
        betas = np.linspace(np.sqrt(beta_start), np.sqrt(beta_end), n, dtype=np.float32) ** 2
        alphas = 1.0 - betas
        return np.cumprod(alphas).astype(np.float32)

    @staticmethod
    def _input_name(session, candidates):
        names = [x.name for x in session.get_inputs()]
        logger.debug("Finding input name from candidates %s in %s", candidates, names)
        for c in candidates:
            for n in names:
                if c in n.lower():
                    return n
        if not names:
            raise RuntimeError("ONNX model has no inputs.")
        return names[0]

    def _encode(self, prompt):
        logger.debug("Encoding prompt (length=%d)", len(prompt))
        ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )["input_ids"].astype(np.int64)
        name = self._input_name(self.text, ["input_ids"])
        return self.text.run(None, {name: ids})[0].astype(np.float32)

    def _unet(self, latents, timestep, hidden):
        inputs = self.unet.get_inputs()
        feed = {}
        for x in inputs:
            n = x.name.lower()
            if "sample" in n:
                feed[x.name] = latents.astype(np.float32)
            elif "timestep" in n:
                feed[x.name] = np.array([timestep], dtype=np.float32)
            elif "encoder_hidden_states" in n or "hidden" in n:
                feed[x.name] = hidden.astype(np.float32)

        logger.debug("UNet feed keys: %s", list(feed.keys()))
        missing = [x.name for x in inputs if x.name not in feed]
        if missing:
            raise RuntimeError(f"UNet input(s) not mapped: {missing}")
        return self.unet.run(None, feed)[0].astype(np.float32)

    def _decode(self, latents):
        name = self._input_name(self.vae, ["latent_sample", "latents", "sample"])
        scaled = (latents / 0.18215).astype(np.float32)
        logger.debug("VAE decode input=%s shape=%s", name, scaled.shape)
        image = self.vae.run(None, {name: scaled})[0]
        image = (image / 2.0 + 0.5).clip(0, 1)
        image = (image[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)
        return Image.fromarray(image, "RGB")

    def generate(self, prompt, width, height, steps, guidance, seed):
        logger.info(
            "Generation started | prompt=%r | size=%sx%s | steps=%s | guidance=%s | seed=%s",
            prompt, width, height, steps, guidance, seed
        )
        native_w = 512
        native_h = 512
        latent_h, latent_w = native_h // 8, native_w // 8

        if self.progress:
            self.progress(0, steps, "Encoding prompt...")
        cond = self._encode(prompt)
        uncond = self._encode("")
        rng = np.random.default_rng(seed)
        latents = rng.standard_normal((1, 4, latent_h, latent_w), dtype=np.float32)

        timesteps = np.linspace(999, 1, steps, dtype=np.int64)
        for i, t in enumerate(timesteps):
            a_t = float(self.scheduler[t])
            prev_t = 0 if i == steps - 1 else int(timesteps[i + 1])
            a_prev = float(self.scheduler[prev_t])

            latent_in = np.concatenate([latents, latents], axis=0)
            hidden = np.concatenate([uncond, cond], axis=0)
            noise = self._unet(latent_in, int(t), hidden)
            eps_uncond, eps_cond = noise[0:1], noise[1:2]
            eps = eps_uncond + guidance * (eps_cond - eps_uncond)

            pred_x0 = (latents - np.sqrt(1.0 - a_t) * eps) / np.sqrt(max(a_t, 1e-6))
            pred_x0 = np.clip(pred_x0, -4.0, 4.0)
            direction = np.sqrt(max(1.0 - a_prev, 0.0)) * eps
            latents = np.sqrt(max(a_prev, 1e-6)) * pred_x0 + direction

            if self.progress:
                self.progress(i + 1, steps, f"Generating {i + 1}/{steps}...")

        image = self._decode(latents)
        if (width, height) != image.size:
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        logger.info("Generation completed successfully")
        return image


class App:
    def __init__(self, root):
        self.root = root
        root.title("KyZer AI Image Generator — Lite")
        root.geometry("1050x760")
        root.minsize(900, 650)
        self.image = None
        self.tkimg = None
        self.engine = None

        frm = ttk.Frame(root, padding=18)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="KyZer AI Image Generator", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        ttk.Label(frm, text="Ultra Lite • Local CPU AI • No API • No credits").pack(anchor="w", pady=(0, 15))

        ttk.Label(frm, text="Prompt").pack(anchor="w")
        self.prompt = tk.Text(frm, height=5, font=("Segoe UI", 11), wrap="word")
        self.prompt.pack(fill="x", pady=(5, 12))

        opts = ttk.Frame(frm)
        opts.pack(fill="x")
        ttk.Label(opts, text="Width").pack(side="left")
        self.w = tk.StringVar(value="1280")
        ttk.Combobox(opts, textvariable=self.w, values=["512", "768", "1024", "1280"], width=7, state="readonly").pack(side="left", padx=5)

        ttk.Label(opts, text="Height").pack(side="left", padx=(15, 0))
        self.h = tk.StringVar(value="720")
        ttk.Combobox(opts, textvariable=self.h, values=["512", "576", "720", "768"], width=7, state="readonly").pack(side="left", padx=5)

        ttk.Label(opts, text="Steps").pack(side="left", padx=(15, 0))
        self.steps = tk.StringVar(value="15")
        ttk.Combobox(opts, textvariable=self.steps, values=["8", "10", "12", "15", "20"], width=7, state="readonly").pack(side="left", padx=5)

        self.btn = ttk.Button(opts, text="GENERATE", command=self.generate)
        self.btn.pack(side="left", padx=25)
        ttk.Button(opts, text="SAVE IMAGE", command=self.save).pack(side="left")

        self.status = tk.StringVar(value="Ready • Tiny AI model is only about 9 MB.")
        ttk.Label(frm, textvariable=self.status).pack(anchor="w", pady=10)
        self.preview = ttk.Label(frm, text="Generated image will appear here", anchor="center")
        self.preview.pack(fill="both", expand=True)

    def set_progress(self, current, total, msg):
        self.root.after(0, lambda: self.status.set(msg))

    def generate(self):
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt:
            messagebox.showwarning("Prompt required", "Enter an image prompt.")
            return
        try:
            width = int(self.w.get())
            height = int(self.h.get())
            steps = int(self.steps.get())
        except ValueError:
            messagebox.showerror("Invalid settings", "Choose valid image settings.")
            return

        logger.info("Generate button clicked")
        self.btn.config(state="disabled")
        self.status.set("Loading tiny local AI...")
        threading.Thread(
            target=self._generate,
            args=(prompt, width, height, steps),
            daemon=True,
        ).start()

    def _generate(self, prompt, width, height, steps):
        try:
            if self.engine is None:
                self.engine = TinyDiffusion(progress=self.set_progress)
            seed = int(time.time_ns() & 0xFFFFFFFF)
            image = self.engine.generate(prompt, width, height, steps, 7.5, seed)
            self.image = image
            path = os.path.join(OUT_DIR, f"kyzer_{time.time_ns()}.png")
            image.save(path)
            logger.info("Image saved: %s", path)
            self.root.after(0, lambda: self.show(image, path))
        except Exception as e:
            log_exception("IMAGE GENERATION FAILED", e)
            error_text = f"{type(e).__name__}: {e}\n\nFull error log saved to:\n{LOG_FILE}"
            self.root.after(0, lambda: messagebox.showerror("Generation failed", error_text))
            self.root.after(0, lambda: self.status.set(f"Generation failed • Log: {LOG_FILE}"))
        finally:
            self.root.after(0, lambda: self.btn.config(state="normal"))

    def show(self, image, path):
        preview = image.copy()
        preview.thumbnail((780, 500))
        self.tkimg = ImageTk.PhotoImage(preview)
        self.preview.config(image=self.tkimg, text="")
        self.status.set(f"Done • Saved automatically: {path}")

    def save(self):
        if self.image is None:
            messagebox.showinfo("No image", "Generate an image first.")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if p:
            self.image.save(p)
            logger.info("Image manually saved: %s", p)
            self.status.set("Saved: " + p)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
