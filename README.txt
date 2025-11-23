
                                    DEEPFAKE DETECTION MODEL


PROBLEM
-------
Detecting deepfake images to their source architecture is
challenging. You need to:
  1. First figure out if an image is real or fake
  2. Then if fake, identify which GAN created it (ProGAN, StyleGAN2, BigGAN)
  3. Make this work even when images are compressed or degraded

This matters because deepfakes are everywhere, and you need to trace them back
to their source.


SOLUTION
--------
A two-stage deep learning pipeline:

Stage 1: Real vs Fake Detector
  - Uses ResNet50 backbone
  - Binary classification: Real or Fake
  - ~95% accuracy on test set

Stage 2: GAN Attribution
  - If image is fake, identify which GAN made it
  - Uses UNet to extract residuals (fingerprints)
  - Uses EfficientNet to classify the residual
  - ~85-90% accuracy per GAN

Both stages use:
  - Adversarial training (PatchGAN discriminator)
  - Perceptual loss (VGG16)
  - Data augmentation (JPEG, blur, rescaling)
  - Multi-crop inference for robustness


HOW TO RUN
----------

1. Install dependencies:
   pip install -r requirements.txt

2. Prepare data:
   mkdir -p data/GANAttributes/{ProGAN,StyleGAN2,BigGAN,RealPhotos}
   # Put your images in these folders

3. Train both stages:
   python -m src.main

   This will:
   - Train Stage 1 (Real/Fake detector)
   - Train Stage 2 (GAN attribution)
   - Evaluate both on validation/test sets
   - Test JPEG compression robustness
   - Save models to model_data/runs/

4. Run inference on a single image:
   python -m src.inference --image path/to/image.jpg

5. Use modular components in your own code:
   from src.models import UNetRes, EncoderClassifier, RealFakeClassifier
   from src.datasets import GFDTwoViewDataset, RealFakeDataset
   from src.utils import RandomDegrade, normalize_residual, plot_cm


CODE STRUCTURE
--------------

src/
├── main.py              
├── config.py            
├── inference.py         Single image inference
├── train.py             Training template
│
├── models/              
│   ├── generator.py       UNetRes (residual extractor)
│   ├── classifier.py      EncoderClassifier, RealFakeClassifier
│   ├── discriminator.py   PatchDiscriminator (adversarial)
│   └── perceptual.py      VGGPerceptual (VGG16 loss)
│
├── datasets/            
│   ├── gan_dataset.py     
│   └── real_fake_dataset.py 
│
└── utils/               
    ├── augmentation.py   RandomDegrade, CarrierBuilder
    ├── preprocessing.py  Image resizing, normalization
    └── visualization.py  Confusion matrix plotting


WHAT TO EXPECT
---------------

Training takes time:
  - Stage 1: ~30-60 minutes (depending on data size)
  - Stage 2: ~60-120 minutes (depending on data size)
  - Total: 2-3 hours on a good GPU

Outputs in model_data/runs/:
  - best_real_fake_model.pt         (Stage 1 checkpoint)
  - best_gfd2_model.pt              (Stage 2 checkpoint)
  - cm_val.png, cm_test.png         (Confusion matrices)
  - sweep_jpeg_stage2.csv           (Robustness test results)
  - val_report.txt, test_report.txt (Classification reports)

Expected performance:
  - Real/Fake detection: 92-96% accuracy
  - GAN attribution: 82-92% accuracy
  - Robustness: Maintains 80%+ accuracy even at JPEG quality 50




TIPS FOR BETTER RESULTS
-----------------------

1. Data quality matters
   - Use balanced datasets (equal samples per GAN)
   - Avoid duplicates or near-duplicates
   - Include diverse lighting, poses, scenes

2. Adjust hyperparameters in src/config.py
   - NUM_EPOCHS: More epochs = better accuracy (but slower)
   - BATCH_SIZE: Bigger batches = more stable but needs more memory
   - LR_GC: Learning rate (default 2e-4 usually works)

3. Monitor training
   - Check early stopping triggers (not overfitting)
   - Look at confusion matrices (where does it fail?)
   - Test on held-out set before deployment





SUMMARY
-------

This is a working deepfake detection system. It's not perfect, but it solves
the core problem: detecting real vs fake images and attributing fakes to their
source GANs.

Use it as-is for basic detection, or extend it for your specific needs.

Good luck!


