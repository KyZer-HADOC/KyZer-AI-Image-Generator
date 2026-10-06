import os
import threading
import time
import urllib.request
import json
import logging
import platform
import sys
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
    logger.error("%s: %s", context, exc or "", exc_info=True)


def global_exception_handler(exc_type, exc_value, exc_traceback):
    if exc_type is KeyboardInterrupt:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("UNHANDLED APPLICATION EXCEPTION", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = global_exception_handler
logger.info("KyZer AI started | Python=%s | Platform=%s | EXE=%s", sys.version.replace("\\n", " "), platform.platform(), sys.executable)

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
            logger.info("Model exists: %s (%d bytes)", rel, os.path.getsize(dst))
        else:
            url = HF_BASE + rel
            tmp = dst + ".part"
            logger.info("Downloading: %s", url)
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dst)
            logger.info("Downloaded: %s (%d bytes)", rel, os.path.getsize(dst))
        if progress:
            progress(i, total, f"Model files {i}/{total}")


class PNDMLite:
    """Small NumPy implementation of the model's PNDMScheduler/PLMS path.
    The model config explicitly uses PNDMScheduler with skip_prk_steps=true.
    """

    def __init__(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        n = int(cfg.get("num_train_timesteps", 1000))
        beta_start = float(cfg.get("beta_start", 0.00085))
        beta_end = float(cfg.get("beta_end", 0.012))
        beta_schedule = cfg.get("beta_schedule", "scaled_linear")
        self.steps_offset = int(cfg.get("steps_offset", 1))
        self.set_alpha_to_one = bool(cfg.get("set_alpha_to_one", False))
        if beta_schedule == "scaled_linear":
            betas = np.linspace(np.sqrt(beta_start), np.sqrt(beta_end), n, dtype=np.float32) ** 2
        else:
            betas = np.linspace(beta_start, beta_end, n, dtype=np.float32)
        self.alphas_cumprod = np.cumprod(1.0 - betas).astype(np.float32)
        self.final_alpha_cumprod = np.float32(1.0 if self.set_alpha_to_one else self.alphas_cumprod[0])
        self.num_train_timesteps = n
        self.ets = []
        self.counter = 0
        self.cur_sample = None
        self.num_inference_steps = None
        self.plms_timesteps = None

    def set_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        base = (np.arange(0, num_inference_steps) * step_ratio).round().astype(np.int64)
        base += self.steps_offset
        self.plms_timesteps = np.concatenate([base[:-1], base[-2:-1], base[-1:]])[::-1].copy()
        self.ets = []
        self.counter = 0
        self.cur_sample = None
        return self.plms_timesteps

    def _get_prev_sample(self, sample, timestep, prev_timestep, model_output):
        a_t = self.alphas_cumprod[int(timestep)]
        a_prev = self.alphas_cumprod[int(prev_timestep)] if prev_timestep >= 0 else self.final_alpha_cumprod
        b_t = 1.0 - a_t
        b_prev = 1.0 - a_prev
        sample_coeff = np.sqrt(a_prev / a_t)
        denom = a_t * np.sqrt(b_prev) + np.sqrt(a_t * b_t * a_prev)
        return sample_coeff * sample - ((a_prev - a_t) * model_output / max(denom, 1e-12))

    def step(self, model_output, timestep, sample):
        step_size = self.num_train_timesteps // self.num_inference_steps
        prev_timestep = int(timestep) - step_size

        if self.counter != 1:
            self.ets = self.ets[-3:]
            self.ets.append(model_output.copy())
        else:
            prev_timestep = int(timestep)
            timestep = int(timestep) + step_size

        if len(self.ets) == 1 and self.counter == 0:
            output = model_output
            self.cur_sample = sample.copy()
        elif len(self.ets) == 1 and self.counter == 1:
            output = (model_output + self.ets[-1]) / 2.0
            sample = self.cur_sample
            self.cur_sample = None
        elif len(self.ets) == 2:
            output = (3.0 * self.ets[-1] - self.ets[-2]) / 2.0
        elif len(self.ets) == 3:
            output = (23.0 * self.ets[-1] - 16.0 * self.ets[-2] + 5.0 * self.ets[-3]) / 12.0
        else:
            output = (55.0 * self.ets[-1] - 59.0 * self.ets[-2] + 37.0 * self.ets[-3] - 9.0 * self.ets[-4]) / 24.0

        prev = self._get_prev_sample(sample, int(timestep), int(prev_timestep), output)
        self.counter += 1
        return prev.astype(np.float32)


class TinyDiffusion:
    def __init__(self, progress=None):
        import onnxruntime as ort
        from transformers import CLIPTokenizer

        self.progress = progress
        logger.info("ONNX Runtime %s | providers=%s", ort.__version__, ort.get_available_providers())
        download_model(progress)

        self.ort = ort
        self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(MODEL_DIR, "tokenizer"))
        self.text = self._load_session("text_encoder/model.onnx", "text_encoder")
        self.unet = self._load_session("unet/model.onnx", "unet")
        self.vae = self._load_session("vae_decoder/model.onnx", "vae_decoder")
        self.scheduler = PNDMLite(os.path.join(MODEL_DIR, "scheduler", "scheduler_config.json"))
        logger.info("PNDM scheduler loaded: skip_prk_steps=true | exact PLMS timestep path")

    def _load_session(self, relative_path, name):
        path = os.path.join(MODEL_DIR, relative_path.replace("/", os.sep))
        opts = self.ort.SessionOptions()
        opts.log_severity_level = 3
        session = self.ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        logger.info("%s inputs=%s outputs=%s", name,
                    [{"name": x.name, "type": x.type, "shape": x.shape} for x in session.get_inputs()],
                    [{"name": x.name, "type": x.type, "shape": x.shape} for x in session.get_outputs()])
        return session

    @staticmethod
    def _input_name(session, candidates):
        names = [x.name for x in session.get_inputs()]
        for c in candidates:
            for n in names:
                if c in n.lower():
                    return n
        if not names:
            raise RuntimeError("ONNX model has no inputs.")
        return names[0]

    def _encode(self, prompt):
        ids = self.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np")["input_ids"].astype(np.int64)
        name = self._input_name(self.text, ["input_ids"])
        return self.text.run(None, {name: ids})[0].astype(np.float32)

    def _unet(self, latents, timestep, hidden):
        feed = {}
        for x in self.unet.get_inputs():
            n = x.name.lower()
            if "sample" in n:
                feed[x.name] = latents.astype(np.float32)
            elif "timestep" in n:
                feed[x.name] = np.asarray(timestep, dtype=np.float32)
            elif "encoder_hidden_states" in n or "hidden" in n:
                feed[x.name] = hidden.astype(np.float32)
        missing = [x.name for x in self.unet.get_inputs() if x.name not in feed]
        if missing:
            raise RuntimeError(f"UNet input(s) not mapped: {missing}")
        return self.unet.run(None, feed)[0].astype(np.float32)

    def _decode(self, latents):
        name = self._input_name(self.vae, ["latent_sample", "latents", "sample"])
        scaled = (latents / 0.18215).astype(np.float32)
        image = self.vae.run(None, {name: scaled})[0]
        image = (image / 2.0 + 0.5).clip(0, 1)
        image = (image[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)
        return Image.fromarray(image, "RGB")

    def generate(self, prompt, width, height, steps, guidance, seed):
        logger.info("Generation started | prompt=%r | output=%sx%s | steps=%s | seed=%s", prompt, width, height, steps, seed)
        # This ONNX export's UNet is configured for a 64x64 latent grid.
        # With the VAE's 4x spatial factor, that corresponds to 256x256
        # native images. This matches the tiny Stable Diffusion pipeline's
        # training/export geometry.
        latent_size = 64
        native = latent_size * 4
        logger.info("Tiny pipeline geometry | native=%s | latent=%sx%s", native, latent_size, latent_size)

        cond = self._encode(prompt)
        uncond = self._encode("")
        rng = np.random.default_rng(seed)
        latents = rng.standard_normal((1, 4, latent_size, latent_size), dtype=np.float32)

        timesteps = self.scheduler.set_timesteps(steps)
        # PNDM/PLMS intentionally returns steps + 1 timesteps because one
        # timestep is repeated to bootstrap the linear multistep method.
        # Do NOT truncate it to "steps" or the final denoising update is lost.
        total_calls = len(timesteps)
        for i, t in enumerate(timesteps):
            latent_in = np.concatenate([latents, latents], axis=0)
            hidden = np.concatenate([uncond, cond], axis=0)
            noise = self._unet(latent_in, int(t), hidden)
            eps_uncond, eps_cond = noise[0:1], noise[1:2]
            eps = eps_uncond + guidance * (eps_cond - eps_uncond)
            latents = self.scheduler.step(eps, int(t), latents)

            if self.progress:
                pct = int((i + 1) / max(total_calls, 1) * 100)
                display_step = min(i + 1, steps)
                self.progress(i + 1, total_calls, f"Generating {display_step}/{steps} • {pct}%")

        image = self._decode(latents)

        # Preserve the requested aspect ratio instead of stretching the
        # tiny model's native square output.
        target_ratio = width / max(height, 1)
        src_ratio = image.width / max(image.height, 1)
        if abs(target_ratio - src_ratio) > 0.01:
            if target_ratio > src_ratio:
                crop_h = max(1, int(image.width / target_ratio))
                top = max(0, (image.height - crop_h) // 2)
                image = image.crop((0, top, image.width, top + crop_h))
            else:
                crop_w = max(1, int(image.height * target_ratio))
                left = max(0, (image.width - crop_w) // 2)
                image = image.crop((left, 0, left + crop_w, image.height))

        if (width, height) != image.size:
            image = image.resize((width, height), Image.Resampling.LANCZOS)

        logger.info("Generation completed successfully | native=%sx%s | latent=%sx%s | final=%sx%s",
                    native, native, latent_size, latent_size, width, height)
        return image


class App:
    BG = "#0b0d12"
    PANEL = "#11151d"
    PANEL2 = "#171c26"
    TEXT = "#f5f7fb"
    MUTED = "#8f9bad"
    ACCENT = "#ff3d71"
    ACCENT2 = "#7c5cff"
    BORDER = "#252c38"

    def __init__(self, root):
        self.root = root
        root.title("KyZer AI")
        root.geometry("1240x820")
        root.minsize(1040, 720)
        root.configure(bg=self.BG)
        self.image = None
        self.tkimg = None
        self.engine = None
        self._setup_icon()
        self._setup_style()
        self._build_ui()

    def _setup_icon(self):
        try:
            base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
            icon = os.path.join(base, "icon.ico")
            if os.path.exists(icon):
                self.root.iconbitmap(icon)
        except Exception as e:
            logger.warning("Could not set application icon: %s", e)

    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background=self.BG, foreground=self.TEXT, font=("Segoe UI", 10))
        style.configure("TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("TLabel", background=self.BG, foreground=self.TEXT)
        style.configure("Muted.TLabel", background=self.BG, foreground=self.MUTED)
        style.configure("Panel.TLabel", background=self.PANEL, foreground=self.TEXT)
        style.configure("PanelMuted.TLabel", background=self.PANEL, foreground=self.MUTED)
        style.configure("TButton", background=self.PANEL2, foreground=self.TEXT, borderwidth=0, padding=(14, 9))
        style.map("TButton", background=[("active", self.ACCENT2)])
        style.configure("Accent.TButton", background=self.ACCENT, foreground="white", padding=(18, 10), font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#ff5a86"), ("disabled", "#4d2634")])
        style.configure("TCombobox", fieldbackground=self.PANEL2, background=self.PANEL2, foreground=self.TEXT, arrowcolor=self.TEXT, padding=6)
        style.map("TCombobox", fieldbackground=[("readonly", self.PANEL2)], foreground=[("readonly", self.TEXT)])
        style.configure("TProgressbar", troughcolor=self.PANEL2, background=self.ACCENT, borderwidth=0, thickness=5)

    def _build_ui(self):
        outer = tk.Frame(self.root, bg=self.BG)
        outer.pack(fill="both", expand=True, padx=24, pady=22)

        header = tk.Frame(outer, bg=self.BG)
        header.pack(fill="x", pady=(0, 16))

        logo = tk.Canvas(header, width=60, height=60, bg=self.BG, highlightthickness=0)
        logo.pack(side="left")
        logo.create_oval(3, 3, 57, 57, fill=self.ACCENT2, outline="")
        logo.create_oval(9, 9, 51, 51, fill=self.ACCENT, outline="")
        logo.create_text(30, 31, text="K", fill="white", font=("Segoe UI", 24, "bold"))
        logo.create_oval(43, 8, 50, 15, fill="#ffffff", outline="")

        title_box = tk.Frame(header, bg=self.BG)
        title_box.pack(side="left", padx=14)
        tk.Label(title_box, text="KyZer AI", bg=self.BG, fg=self.TEXT, font=("Segoe UI", 23, "bold")).pack(anchor="w")
        tk.Label(title_box, text="LOCAL IMAGE GENERATOR  •  PRIVATE  •  UNLIMITED", bg=self.BG, fg=self.MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w")

        status_pill = tk.Frame(header, bg="#14221b", highlightthickness=1, highlightbackground="#254b37")
        status_pill.pack(side="right", pady=12)
        tk.Label(status_pill, text="  ●  LOCAL CPU  ", bg="#14221b", fg="#69e89a", font=("Segoe UI", 9, "bold")).pack(padx=2, pady=6)

        content = tk.Frame(outer, bg=self.BG)
        content.pack(fill="both", expand=True)

        left = tk.Frame(content, bg=self.PANEL, width=380, highlightthickness=1, highlightbackground=self.BORDER)
        left.pack(side="left", fill="y", padx=(0, 14))
        left.pack_propagate(False)

        tk.Label(left, text="Create something amazing", bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=22, pady=(22, 3))
        tk.Label(left, text="Your prompt stays on this PC. No API • No credits • No limits.", bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=22, pady=(0, 16))

        tk.Label(left, text="PROMPT", bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=20)
        prompt_wrap = tk.Frame(left, bg=self.PANEL2, highlightthickness=1, highlightbackground=self.BORDER)
        prompt_wrap.pack(fill="x", padx=22, pady=(6, 15))
        self.prompt = tk.Text(prompt_wrap, height=8, bg=self.PANEL2, fg=self.TEXT, insertbackground=self.TEXT, relief="flat", bd=0, wrap="word", font=("Segoe UI", 10), padx=10, pady=9)
        self.prompt.pack(fill="both", expand=True)

        self._label(left, "WIDTH", 22, 0)
        self.w = tk.StringVar(value="1280")
        ttk.Combobox(left, textvariable=self.w, values=["512", "768", "1024", "1280"], state="readonly").pack(fill="x", padx=22, pady=(5, 9))

        self._label(left, "HEIGHT", 22, 0)
        self.h = tk.StringVar(value="720")
        ttk.Combobox(left, textvariable=self.h, values=["512", "576", "720", "768"], state="readonly").pack(fill="x", padx=22, pady=(5, 9))

        self._label(left, "QUALITY / STEPS", 22, 0)
        self.steps = tk.StringVar(value="15")
        ttk.Combobox(left, textvariable=self.steps, values=["8", "10", "12", "15", "20"], state="readonly").pack(fill="x", padx=22, pady=(5, 15))

        self.btn = ttk.Button(left, text="✦  GENERATE IMAGE", style="Accent.TButton", command=self.generate)
        self.btn.pack(fill="x", padx=22, ipady=2)

        ttk.Button(left, text="SAVE CURRENT IMAGE", command=self.save).pack(fill="x", padx=22, pady=9)

        self.status = tk.StringVar(value="Ready • First run downloads the tiny local model (~9 MB).")
        tk.Label(left, text="TIP  •  Keep prompts simple and concrete for this ultra-light model.", bg=self.PANEL, fg="#6f7890", wraplength=330, justify="left", font=("Segoe UI", 8)).pack(anchor="w", padx=22, pady=(5, 5))
        tk.Label(left, textvariable=self.status, bg=self.PANEL, fg=self.MUTED, wraplength=330, justify="left", font=("Segoe UI", 8)).pack(anchor="w", padx=22, pady=(6, 8))

        right = tk.Frame(content, bg=self.PANEL, highlightthickness=1, highlightbackground=self.BORDER)
        right.pack(side="left", fill="both", expand=True)

        top = tk.Frame(right, bg=self.PANEL)
        top.pack(fill="x", padx=18, pady=16)
        tk.Label(top, text="Preview", bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 15, "bold")).pack(side="left")
        tk.Label(top, text="AUTO-SAVED  •  PICTURES / KYZER AI", bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 8, "bold")).pack(side="right")

        preview_bg = tk.Frame(right, bg="#080a0f", highlightthickness=1, highlightbackground="#2b3140")
        preview_bg.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        self.preview = tk.Label(preview_bg, text="✦\n\nYour generated artwork\nwill appear here", bg="#080a0f", fg="#596273", font=("Segoe UI", 15, "bold"), justify="center")
        self.preview.pack(fill="both", expand=True, padx=20, pady=20)

        self.progress = ttk.Progressbar(right, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=18, pady=(0, 5))
        self.progress["value"] = 0

    def _label(self, parent, text, padx, pady):
        tk.Label(parent, text=text, bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=padx, pady=(pady, 0))

    def set_progress(self, current, total, msg):
        pct = int(current / max(total, 1) * 100)
        self.root.after(0, lambda: (self.status.set(msg), self.progress.configure(value=pct)))

    def generate(self):
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt:
            messagebox.showwarning("Prompt required", "Enter a description first.")
            return
        try:
            width, height, steps = int(self.w.get()), int(self.h.get()), int(self.steps.get())
        except ValueError:
            messagebox.showerror("Invalid settings", "Choose valid image settings.")
            return

        logger.info("Generate clicked | prompt=%r | size=%sx%s | steps=%s", prompt, width, height, steps)
        self.btn.config(state="disabled")
        self.progress["value"] = 0
        self.status.set("Loading local AI...")
        threading.Thread(target=self._generate, args=(prompt, width, height, steps), daemon=True).start()

    def _generate(self, prompt, width, height, steps):
        try:
            if self.engine is None:
                self.engine = TinyDiffusion(progress=self.set_progress)
            seed = int(time.time_ns() & 0xFFFFFFFF)
            image = self.engine.generate(prompt, width, height, steps, 7.5, seed)
            self.image = image
            path = os.path.join(OUT_DIR, f"kyzer_{time.strftime('%Y%m%d_%H%M%S')}_{seed}.png")
            image.save(path)
            logger.info("Image saved: %s", path)
            self.root.after(0, lambda: self.show(image, path))
        except Exception as e:
            log_exception("IMAGE GENERATION FAILED", e)
            text = f"{type(e).__name__}: {e}\n\nFull log:\n{LOG_FILE}"
            self.root.after(0, lambda: messagebox.showerror("Generation failed", text))
            self.root.after(0, lambda: self.status.set("Generation failed • check the log file."))
        finally:
            self.root.after(0, lambda: self.btn.config(state="normal"))

    def show(self, image, path):
        preview = image.copy()
        preview.thumbnail((760, 560))
        self.tkimg = ImageTk.PhotoImage(preview)
        self.preview.config(image=self.tkimg, text="")
        self.progress["value"] = 100
        self.status.set(f"Done • Saved: {path}")

    def save(self):
        if self.image is None:
            messagebox.showinfo("No image", "Generate an image first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if p:
            self.image.save(p)
            logger.info("Image manually saved: %s", p)
            self.status.set("Saved: " + p)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
