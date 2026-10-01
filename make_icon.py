from PIL import Image, ImageDraw, ImageFont

S = 256
img = Image.new("RGBA", (S, S), (11, 13, 18, 255))
d = ImageDraw.Draw(img)

# Premium neon-style KyZer K mark
d.rounded_rectangle((12, 12, 244, 244), radius=58, fill=(20, 24, 34, 255), outline=(124, 92, 255, 255), width=6)
d.ellipse((42, 42, 214, 214), fill=(255, 61, 113, 255))
d.ellipse((56, 56, 200, 200), fill=(124, 92, 255, 255))

try:
    font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 118)
except Exception:
    font = ImageFont.load_default()

text = "K"
box = d.textbbox((0, 0), text, font=font)
tw, th = box[2] - box[0], box[3] - box[1]
d.text(((S - tw) / 2 - 2, (S - th) / 2 - 13), text, font=font, fill="white")

# Tiny AI spark
d.polygon([(194, 48), (199, 61), (212, 66), (199, 71), (194, 84), (189, 71), (176, 66), (189, 61)], fill="white")

img.save("icon.png")
img.save("icon.ico", sizes=[(256,256), (128,128), (64,64), (48,48), (32,32), (16,16)])
