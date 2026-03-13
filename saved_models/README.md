# Saved Models Directory

This directory will contain trained PyTorch model checkpoints after running the training notebook.

## Expected Files After Training

After running the notebook, you should see files like:
- `trained_modelr18.h5` - ResNet18 model
- `trained_modelr50.h5` - ResNet50 model
- `trained_modelv16.h5` - VGG16 model
- `trained_modelv19.h5` - VGG19 model
- `trained_modeldense.h5` - DenseNet121 model

Plus cross-entropy variants with `_cross` suffix.

## Model Loading

To load a trained model for inference:
```python
import torch
model = torch.load('saved_models/trained_modelr18.h5')
model.eval()
```

## Pre-trained Models

For demonstration purposes, you may want to include some pre-trained checkpoints here, but they are not included in this repository due to size constraints.