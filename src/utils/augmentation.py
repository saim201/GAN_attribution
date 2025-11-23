import io
import random
from PIL import Image, ImageFilter, ImageOps


class RandomDegrade:
    
    def __init__(self, apply_prob=1.0):
        self.apply_prob = apply_prob
        self.kernels = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.BOX, Image.LANCZOS]

    def _jpeg(self, img):
        q = random.randint(50, 100)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=q, optimize=True)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    def _tiny_rescale(self, img):
        s = random.uniform(0.9, 1.1)
        w, h = img.size
        tw, th = max(8, int(w*s)), max(8, int(h*s))
        img = img.resize((tw, th), resample=random.choice(self.kernels))
        return img.resize((w, h), resample=random.choice(self.kernels))

    def _blur_or_sharpen(self, img):
        if random.random() < 0.5:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))
        else:
            return img.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))

    def __call__(self, img):
        if random.random() > self.apply_prob:
            return img
        img = self._jpeg(img)
        img = self._tiny_rescale(img)
        img = self._blur_or_sharpen(img)
        return img


class CarrierBuilder:
    
    def __init__(self):
        self.kernels = [Image.BILINEAR, Image.BICUBIC, Image.BOX, Image.LANCZOS]

    def __call__(self, img):
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.2)))
        w, h = img.size
        s = random.uniform(0.5, 0.85)
        tw, th = max(8, int(w*s)), max(8, int(h*s))
        img = img.resize((tw, th), resample=random.choice(self.kernels))
        img = img.resize((w, h), resample=random.choice(self.kernels))
        return img
