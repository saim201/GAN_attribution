
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

from .models import UNetRes, EncoderClassifier, RealFakeClassifier, PatchDiscriminator
from .datasets import GFDTwoViewDataset, RealFakeDataset
from .utils import normalize_residual, plot_cm, IMAGENET_NORM
from .config import NUM_EPOCHS, BATCH_SIZE, LR_GC, RF_NUM_EPOCHS, RF_BATCH_SIZE, RF_LR

warnings.filterwarnings('ignore')


class DeepFakeDetector:

    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.output_dir = Path("model_data/runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._set_seed(42)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_data(self, data_root="data/GANAttributes"):
        print(f"\nLoading data from {data_root}...")

        df_list = []
        data_path = Path(data_root)

        for class_name in ["ProGAN", "StyleGAN2", "BigGAN", "RealPhotos"]:
            class_dir = data_path / class_name
            if class_dir.exists():
                for img_file in class_dir.glob("*.jpg"):
                    df_list.append({
                        "path": str(img_file),
                        "label": class_name,
                        "is_fake": 0 if class_name == "RealPhotos" else 1
                    })

        df = pd.DataFrame(df_list)
        print(f"Loaded {len(df)} images")
        print(df["label"].value_counts())

        return df


    def split_data(self, df, test_size=0.2, val_size=0.1):
        print("\nSplitting data...")

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = list(splitter.split(df, df['label']))[0]

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        val_idx = np.random.choice(train_df.index, size=int(len(train_df)*val_size), replace=False)
        val_df = train_df.loc[val_idx].copy()
        train_df = train_df.drop(val_idx).copy()

        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'

        df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return df

    def train_stage1(self, df):
        print("\n --- STAGE 1: Real vs Fake Detection")

        model = RealFakeClassifier(num_classes=2).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=RF_LR)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        patience, patience_left = 3, 3

        for epoch in range(RF_NUM_EPOCHS):
            model.train()
            train_loss = 0

            train_loader = DataLoader(
                RealFakeDataset(df[df['split']=='train'], 'train'),
                batch_size=RF_BATCH_SIZE,
                shuffle=True
            )

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{RF_NUM_EPOCHS}"):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            val_loader = DataLoader(
                RealFakeDataset(df[df['split']=='val'], 'val'),
                batch_size=RF_BATCH_SIZE
            )

            model.eval()
            with torch.no_grad():
                val_preds, val_true = [], []
                for images, labels in val_loader:
                    images = images.to(self.device)
                    outputs = model(images)
                    preds = outputs.argmax(1)
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(labels.numpy())

                val_acc = np.mean(np.array(val_preds) == np.array(val_true)) * 100

            print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_left = patience
                torch.save(model.state_dict(), self.output_dir / "best_real_fake_model.pt")
                print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("  Early stopping triggered")
                    break

        return model


    def train_stage2(self, df):
        print("\n --- STAGE 2: GAN Attribution")

        model_g = UNetRes(scale=0.20).to(self.device)
        model_c = EncoderClassifier(num_classes=3).to(self.device)
        model_d = PatchDiscriminator().to(self.device)

        opt_g = optim.Adam(model_g.parameters(), lr=LR_GC)
        opt_c = optim.Adam(model_c.parameters(), lr=LR_GC)

        criterion_cls = nn.CrossEntropyLoss()

        best_val_acc = 0
        patience, patience_left = 3, 3

        for epoch in range(NUM_EPOCHS):
            model_g.train()
            model_c.train()
            model_d.train()

            train_loss = 0

            train_loader = DataLoader(
                GFDTwoViewDataset(df[df['split']=='train'], 'train', num_classes=3),
                batch_size=BATCH_SIZE,
                shuffle=True
            )

            for x, c, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                x, c, y = x.to(self.device), c.to(self.device), y.to(self.device)

                r = model_g(x)
                r_img = normalize_residual(r)
                r_img = IMAGENET_NORM(r_img)

                logits_c = model_c(r_img)
                loss_c = criterion_cls(logits_c, y)

                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()

                opt_g.zero_grad()
                loss_g = criterion_cls(logits_c, y)
                loss_g.backward()
                opt_g.step()

                train_loss += loss_g.item()

            val_loader = DataLoader(
                GFDTwoViewDataset(df[df['split']=='val'], 'val', num_classes=3),
                batch_size=BATCH_SIZE
            )

            model_g.eval()
            model_c.eval()
            with torch.no_grad():
                val_preds, val_true = [], []
                for x, _, y in val_loader:
                    x = x.to(self.device)
                    r = model_g(x)
                    r_img = normalize_residual(r)
                    r_img = IMAGENET_NORM(r_img)
                    logits = model_c(r_img)
                    preds = logits.argmax(1)
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(y.numpy())

                val_acc = np.mean(np.array(val_preds) == np.array(val_true)) * 100

            print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_left = patience
                torch.save(model_g.state_dict(), self.output_dir / "best_gan_generator.pt")
                torch.save(model_c.state_dict(), self.output_dir / "best_gan_classifier.pt")
                print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("  Early stopping triggered")
                    break

        return model_g, model_c

    def evaluate(self, model_rf, model_g, model_c, df):
        print("\n --- EVALUATION ON TEST SET")


        # Stage 1 evaluation
        test_loader = DataLoader(
            RealFakeDataset(df[df['split']=='test'], 'test'),
            batch_size=RF_BATCH_SIZE
        )

        model_rf.eval()
        with torch.no_grad():
            test_preds, test_true = [], []
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = model_rf(images)
                preds = outputs.argmax(1)
                test_preds.extend(preds.cpu().numpy())
                test_true.extend(labels.numpy())

            test_acc = np.mean(np.array(test_preds) == np.array(test_true)) * 100
            cm = confusion_matrix(test_true, test_preds)

            print(f"\nStage 1 - Real/Fake Detection")
            print(f"Test Accuracy: {test_acc:.2f}%")
            print(classification_report(test_true, test_preds, target_names=["Real", "Fake"]))
            plot_cm(cm, "Real/Fake Test CM", str(self.output_dir / "rf_test_cm.png"), ["Real", "Fake"])

        # Stage 2 evaluation
        test_loader = DataLoader(
            GFDTwoViewDataset(df[df['split']=='test'], 'test', num_classes=3),
            batch_size=BATCH_SIZE
        )

        model_g.eval()
        model_c.eval()
        with torch.no_grad():
            test_preds, test_true = [], []
            for x, _, y in test_loader:
                x = x.to(self.device)
                r = model_g(x)
                r_img = normalize_residual(r)
                r_img = IMAGENET_NORM(r_img)
                logits = model_c(r_img)
                preds = logits.argmax(1)
                test_preds.extend(preds.cpu().numpy())
                test_true.extend(y.numpy())

            test_acc = np.mean(np.array(test_preds) == np.array(test_true)) * 100
            cm = confusion_matrix(test_true, test_preds)

            print(f"\nStage 2 - GAN Attribution")
            print(f"Test Accuracy: {test_acc:.2f}%")
            print(classification_report(test_true, test_preds, target_names=["ProGAN", "StyleGAN2", "BigGAN"]))
            plot_cm(cm, "GAN Test CM", str(self.output_dir / "gan_test_cm.png"), ["ProGAN", "StyleGAN2", "BigGAN"])

    def run(self, data_root="data/GANAttributes"):

        try:
            df = self.load_data(data_root)
            df = self.split_data(df)

            model_rf = self.train_stage1(df)
            model_g, model_c = self.train_stage2(df)

            self.evaluate(model_rf, model_g, model_c, df)

            print("\n --- PIPELINE COMPLETE ✓")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    detector = DeepFakeDetector(device='cuda')
    detector.run(data_root="data/GANAttributes")
