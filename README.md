# Hand Gesture Recognition

This repository contains a Jupyter notebook and supporting code for a deep learning project that recognises hand gestures from images. The work demonstrates how to train, evaluate and infer with convolutional neural networks on the HaGRID dataset.

## 📁 Repository Structure

```
HandGestureRecognition/
├── HandGestureRecognition.ipynb   # Main analysis and training notebook
├── requirements.txt               # Python dependencies
├── dataset/                       # (empty) - download HaGRID dataset here
│   └── README.md                  # Dataset download instructions
├── saved_models/                  # (empty) - trained models saved here
│   └── README.md                  # Model information
└── utils/                         # Utility modules
    ├── models.py                  # Model definitions and training class
    ├── DataLoader.py              # Dataset loading and preprocessing
    ├── plotter.py                 # Visualization utilities
    ├── prediction_model.py        # Inference helper
    └── combine_json.py            # Annotation processing
```

## 📘 Dataset

The project uses the [HaGRID (Hand Gesture Recognition Image Dataset)](https://github.com/hukenovs/hagrid).

### Download Instructions

1. Visit [HaGRID GitHub](https://github.com/hukenovs/hagrid)
2. Download the dataset (~50GB)
3. Extract into the `dataset/` directory
4. If needed, run `python utils/combine_json.py` to combine annotations

See `dataset/README.md` for detailed instructions.

## 🛠 Setup & Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+ with CUDA (recommended for GPU training)

### Installation

1. **Clone/download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** (see Dataset section above)

4. **Run the notebook**:
   ```bash
   jupyter notebook HandGestureRecognition.ipynb
   ```

## 🚀 Usage

### Training Models

The notebook demonstrates training multiple architectures:
- ResNet18/50
- VGG16/19  
- DenseNet121

With different loss functions:
- Cross-Entropy Loss
- Margin Loss

### Key Results

| Model | Loss Function | Accuracy |
|-------|---------------|----------|
| VGG16 | Margin Loss   | ~95%     |
| VGG19 | Margin Loss   | ~93%     |
| ResNet18 | Margin Loss | ~88%     |

### Inference

After training, use saved models for prediction on new images.

## 📊 Model Details

- **Input**: 224×224 RGB images
- **Output**: 19-class classification (18 gestures + no_gesture)
- **Training**: 10 epochs, batch size 64, SGD optimizer
- **Evaluation**: Accuracy, F1-score, confusion matrices

## 🏃‍♂️ Quick Start

1. Download dataset → `dataset/`
2. Install requirements
3. Run notebook cells sequentially
4. Models save to `saved_models/`
5. Use trained models for inference

## 📄 License & Attribution

Dataset from HaGRID project. Code for educational purposes.

---

**Note**: Training requires significant GPU resources. The notebook includes GPU checks and CPU fallbacks.

## 🧠 Inference Example

After training, the final section of the notebook demonstrates how to load new images, preprocess them, and perform inference using a saved model (e.g., ResNet18). The code reads `*.jpg` files from a directory, resizes and normalises them, converts to tensors, then runs the model to produce class labels.

## 📌 License & Attribution

Dataset and models used are for educational purposes. Please refer to the original HaGRID repository for terms of use.

---

Feel free to extend the notebook with additional architectures, augmentations, or a real‑time webcam demo.