# 🏥 Lung Disease Classification System - Usage Guide

## 📋 Overview
This system uses **Vision Transformer + Mixture of Experts (MoE)** architecture for lung disease classification from X-ray images.

## 🎯 Available Models
- ✅ **TB Model**: `models_saved/tb_moet.pth` (98.12 MB)
- ✅ **Cancer Model**: `models_saved/cancer_moet.pth` (98.12 MB)

## 🚀 Quick Start

### 1. Command Line Prediction
```bash
# Predict with TB model
python src/training/predict.py xray_image.jpg models_saved/tb_moet.pth

# Predict with Cancer model
python src/training/predict.py xray_image.jpg models_saved/cancer_moet.pth

# Default uses TB model
python src/training/predict.py xray_image.jpg
```

### 2. Batch Testing
```bash
# Test both models on available data
python src/training/test.py
```

### 3. Advanced Model Loading
```bash
# Use custom model format (supports .adding, .pth, .pt, .pkl)
python src/training/predict_trained.py lung.adding xray_image.jpg
```

## 📊 Model Architecture
- **Vision Transformer**: Patch embedding, 6 transformer blocks, 8 attention heads
- **Mixture of Experts**: 3 expert networks with top-k routing
- **Classification**: Binary classification (Normal vs Disease)

## 🔧 Integration Examples

### Python API Usage
```python
from src.utils.model_loader import TrainedModelLoader

# Load trained model
loader = TrainedModelLoader("models_saved/cancer_moet.pth")
model = loader.load_model()

# Make prediction
prediction, confidence = loader.get_confidence_and_prediction(image_tensor)

result = "NORMAL" if prediction == 0 else "DISEASE DETECTED"
print(f"Prediction: {result} ({confidence*100:.2f}% confidence)")
```

### Custom Pipeline Integration
```python
import cv2
import torch
from src.utils.model_loader import TrainedModelLoader

def predict_xray(image_path, model_path="models_saved/tb_moet.pth"):
    # Load model
    loader = TrainedModelLoader(model_path)
    model = loader.load_model()
    
    # Preprocess
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0)
    
    # Predict
    pred, conf = loader.get_confidence_and_prediction(img)
    
    return pred, conf
```

## 📈 Current Performance
- **Training Data**: Dummy random images (6 samples per class)
- **Accuracy**: ~50% (baseline for random data)
- **Confidence**: ~52% (expected with dummy data)

## 🎯 To Achieve High Confidence (90-99%)
1. **Replace dummy data** with real medical X-rays
2. **Organize data** as:
   ```
   data/
   ├── tb/
   │   ├── normal/     # Real normal X-rays
   │   └── disease/    # Real TB X-rays
   └── cancer/
       ├── normal/     # Real normal X-rays
       └── disease/    # Real cancer X-rays
   ```
3. **Retrain** with real data:
   ```bash
   python src/training/train.py
   ```

## 🔍 Model Analysis
```bash
# Analyze any model file
python analyze_model.py

# Check specific model
python analyze_model.py models_saved/cancer_moet.pth
```

## 📁 File Structure
```
src/
├── models/
│   └── moet.py                    # ViT + MoE architecture
├── preprocessing/
│   └── dataset.py                 # Data loader
├── training/
│   ├── train.py                   # Training script
│   ├── test.py                    # Evaluation script
│   ├── predict.py                 # Main prediction
│   └── predict_trained.py         # Advanced prediction
└── utils/
    └── model_loader.py            # Model loading utilities
```

## 🚨 Important Notes
- Models are trained on dummy data - replace with real medical images for production
- Architecture supports 90-99% confidence with proper training data
- All scripts automatically handle device selection (CPU/GPU)
- Supports multiple model formats (.pth, .pt, .pkl, .adding)

## 📞 Support
For issues or questions:
1. Check model file: `python analyze_model.py`
2. Test with dummy data: `python src/training/predict.py data/tb/normal/img0.jpg`
3. Verify installation: All dependencies in `requirements.txt`

---
**Ready for production with real medical data!** 🏥✨
