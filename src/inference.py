
import argparse
import torch
from pathlib import Path
from PIL import Image, ImageOps

from .models import UNetRes, EncoderClassifier, RealFakeClassifier
from .utils import normalize_residual, IMAGENET_NORM
import torchvision.transforms as T


class DeepFakeInference:

    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_rf = None
        self.model_g = None
        self.model_c = None
        self.to_tensor = T.ToTensor()

    def load_models(self, model_dir="model_data/runs"):
        model_dir = Path(model_dir)

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Load Stage 1: Real/Fake detector
        rf_path = model_dir / "best_real_fake_model.pt"
        if rf_path.exists():
            self.model_rf = RealFakeClassifier(num_classes=2).to(self.device)
            self.model_rf.load_state_dict(torch.load(rf_path, map_location=self.device))
            self.model_rf.eval()
            print(f"Loaded Real/Fake model from {rf_path}")
        else:
            print(f"Real/Fake model not found at {rf_path}")

        # Load Stage 2: GAN attribution
        gen_path = model_dir / "best_gan_generator.pt"
        clf_path = model_dir / "best_gan_classifier.pt"
        if gen_path.exists() and clf_path.exists():
            self.model_g = UNetRes(scale=0.20).to(self.device)
            self.model_c = EncoderClassifier(num_classes=3).to(self.device)
            self.model_g.load_state_dict(torch.load(gen_path, map_location=self.device))
            self.model_c.load_state_dict(torch.load(clf_path, map_location=self.device))
            self.model_g.eval()
            self.model_c.eval()
            print(f"Loaded GAN models from {gen_path} and {clf_path}")
        else:
            print(f"GAN models not found")

    def preprocess_image(self, img_path, img_size=256):
        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img).convert('RGB')
            img = img.resize((img_size, img_size), Image.BILINEAR)
            return img

    @torch.no_grad()
    def predict(self, img_path):
        if not Path(img_path).exists():
            return {"error": f"Image not found: {img_path}"}

        img = self.preprocess_image(img_path)
        x = IMAGENET_NORM(self.to_tensor(img)).unsqueeze(0).to(self.device)

        result = {"image": str(img_path)}

        # Stage 1: Real vs Fake
        if self.model_rf is None:
            result["real_fake"] = "error - model not loaded"
        else:
            with torch.no_grad():
                logits = self.model_rf(x)
                pred = logits.argmax(1).item()
                conf = torch.softmax(logits, dim=1).max().item()

                real_fake = ["Real", "Fake"][pred]
                result["real_fake"] = real_fake
                result["real_fake_confidence"] = f"{conf:.2%}"

                # Stage 2: GAN attribution (only if fake)
                if pred == 1 and self.model_g is not None:
                    r = self.model_g(x)
                    r_img = normalize_residual(r)
                    r_img = IMAGENET_NORM(r_img)
                    logits_gan = self.model_c(r_img)
                    gan_pred = logits_gan.argmax(1).item()
                    gan_conf = torch.softmax(logits_gan, dim=1).max().item()

                    gan_names = ["ProGAN", "StyleGAN2", "BigGAN"]
                    result["gan_type"] = gan_names[gan_pred]
                    result["gan_confidence"] = f"{gan_conf:.2%}"

        return result


def main():
    parser = argparse.ArgumentParser(description='DeepFake Detection Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to image or directory')
    parser.add_argument('--model-dir', type=str, default='model_data/runs', help='Directory with trained models')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device: cuda or cpu')

    args = parser.parse_args()

    print("\n --- DEEPFAKE DETECTION - INFERENCE")

    inference = DeepFakeInference(device=args.device)
    inference.load_models(args.model_dir)

    img_path = Path(args.image)

    if img_path.is_file():
        print(f"\nRunning inference on: {args.image}")
        result = inference.predict(str(img_path))
        print("\nResult:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    elif img_path.is_dir():
        print(f"\nRunning inference on images in: {args.image}")
        for img_file in img_path.glob("*.jpg"):
            result = inference.predict(str(img_file))
            print(f"{img_file.name}: {result['real_fake']}", end="")
            if "gan_type" in result:
                print(f" ({result['gan_type']})", end="")
            print()
    else:
        print(f"Error: {args.image} is not a file or directory")



if __name__ == "__main__":
    main()
