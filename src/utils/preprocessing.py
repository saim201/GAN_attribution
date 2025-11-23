import torch
import torchvision.transforms as T
from PIL import Image, ImageOps


# ImageNet normalization
IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
IMG_SIZE = 256


def resize_crop_256(img: Image.Image, img_size=IMG_SIZE):
    s = random.uniform(0.95, 1.05)
    w, h = img.size
    img = img.resize((max(8, int(w*s)), max(8, int(h*s))), resample=Image.BICUBIC)
    w, h = img.size
    
    if w < img_size or h < img_size:
        pad_w = max(0, img_size - w)
        pad_h = max(0, img_size - h)
        img = ImageOps.expand(img, border=(pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2), fill=0)
    
    w, h = img.size
    left = max(0, (w - img_size)//2)
    top = max(0, (h - img_size)//2)
    img = img.crop((left, top, left + img_size, top + img_size))
    return img


def normalize_residual(r, alpha=0.20):
    return (r / (alpha + 1e-8) * 0.5 + 0.5)


def clamp_img(x):
    return torch.clamp(x, -3.0, 3.0)
