import os, threading, tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_DIR = os.path.join(os.path.expanduser("~"), ".kyzer_ai", "model")
OUT_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "KyZer AI")
os.makedirs(OUT_DIR, exist_ok=True)

class App:
    def __init__(self, root):
        self.root=root
        root.title("KyZer AI Image Generator")
        root.geometry("1050x760")
        root.minsize(900,650)
        self.image=None
        self.tkimg=None
        frm=ttk.Frame(root,padding=18); frm.pack(fill="both",expand=True)
        ttk.Label(frm,text="KyZer AI Image Generator",font=("Segoe UI",22,"bold")).pack(anchor="w")
        ttk.Label(frm,text="Local • Offline after model download • No generation quota").pack(anchor="w",pady=(0,15))
        ttk.Label(frm,text="Prompt").pack(anchor="w")
        self.prompt=tk.Text(frm,height=5,font=("Segoe UI",11),wrap="word"); self.prompt.pack(fill="x",pady=(5,12))
        opts=ttk.Frame(frm); opts.pack(fill="x")
        ttk.Label(opts,text="Width").pack(side="left"); self.w=tk.StringVar(value="512"); ttk.Combobox(opts,textvariable=self.w,values=["384","512","640"],width=7,state="readonly").pack(side="left",padx=5)
        ttk.Label(opts,text="Height").pack(side="left",padx=(15,0)); self.h=tk.StringVar(value="512"); ttk.Combobox(opts,textvariable=self.h,values=["384","512","640"],width=7,state="readonly").pack(side="left",padx=5)
        ttk.Label(opts,text="Steps").pack(side="left",padx=(15,0)); self.steps=tk.StringVar(value="20"); ttk.Combobox(opts,textvariable=self.steps,values=["10","15","20","25","30"],width=7,state="readonly").pack(side="left",padx=5)
        self.btn=ttk.Button(opts,text="GENERATE",command=self.generate); self.btn.pack(side="left",padx=25)
        ttk.Button(opts,text="SAVE IMAGE",command=self.save).pack(side="left")
        self.status=tk.StringVar(value="Ready. First generation downloads the AI model (~4 GB).")
        ttk.Label(frm,textvariable=self.status).pack(anchor="w",pady=10)
        self.preview=ttk.Label(frm,text="Your generated image will appear here",anchor="center")
        self.preview.pack(fill="both",expand=True)

    def generate(self):
        p=self.prompt.get("1.0","end").strip()
        if not p: messagebox.showwarning("Prompt required","Enter an image prompt."); return
        self.btn.config(state="disabled"); self.status.set("Loading local AI model / generating...")
        threading.Thread(target=self._generate,args=(p,),daemon=True).start()

    def _generate(self,prompt):
        try:
            import torch
            from diffusers import StableDiffusionPipeline
            if not hasattr(self,"pipe"):
                self.pipe=StableDiffusionPipeline.from_pretrained(
                    MODEL_ID, cache_dir=MODEL_DIR, torch_dtype=torch.float32,
                    safety_checker=None, requires_safety_checker=False
                )
                self.pipe.to("cpu")
            img=self.pipe(prompt,width=int(self.w.get()),height=int(self.h.get()),
                          num_inference_steps=int(self.steps.get())).images[0]
            self.image=img
            path=os.path.join(OUT_DIR,"kyzer_"+str(__import__("time").time_ns())+".png")
            img.save(path)
            self.root.after(0,lambda:self.show(img,path))
        except Exception as e:
            self.root.after(0,lambda: messagebox.showerror("Generation failed",str(e)))
            self.root.after(0,lambda:self.status.set("Generation failed."))
        finally:
            self.root.after(0,lambda:self.btn.config(state="normal"))

    def show(self,img,path):
        preview=img.copy(); preview.thumbnail((780,500))
        self.tkimg=ImageTk.PhotoImage(preview)
        self.preview.config(image=self.tkimg,text="")
        self.status.set("Done • Saved automatically to: "+path)

    def save(self):
        if self.image is None: messagebox.showinfo("No image","Generate an image first."); return
        p=filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if p: self.image.save(p); self.status.set("Saved: "+p)

if __name__=="__main__":
    root=tk.Tk(); App(root); root.mainloop()
