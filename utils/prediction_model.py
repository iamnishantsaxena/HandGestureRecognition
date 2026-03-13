import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from .models import ResNet18  # Import the ResNet18 model from models.py

class ImageClassifier:
    """
    A wrapper class for loading trained gesture recognition models and performing inference.
    Handles model loading, preprocessing, and prediction on single images.
    """
    def __init__(self, num_classes, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(num_classes, model_path)
        self.model.eval()

        # Define image preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self, num_classes, model_path):
        # Create an instance of the ResNet18 model without pretrained weights
        model = ResNet18(num_classes=num_classes, pretrained=False, freezed=False)

        # Load the model's state dictionary from the H5 file
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)

        return model

    def preprocess_image(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path)
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add a batch dimension
        return image.to(self.device)

    def predict(self, image_path):
        # Preprocess the image
        image = self.preprocess_image(image_path)

        with torch.no_grad():
            output = self.model(image)

        # Convert the model's output tensors to probabilities using softmax
        probabilities_gesture = torch.softmax(output['gesture'], dim=1)
        probabilities_leading_hand = torch.softmax(output['leading_hand'], dim=1)

        # Get the predicted classes by taking the argmax of the probabilities
        predicted_gesture = torch.argmax(probabilities_gesture, dim=1).item()
        predicted_leading_hand = torch.argmax(probabilities_leading_hand, dim=1).item()

        return {
            'predicted_gesture': predicted_gesture,
            'probabilities_gesture': probabilities_gesture.squeeze().cpu().numpy(),
            'predicted_leading_hand': predicted_leading_hand,
            'probabilities_leading_hand': probabilities_leading_hand.squeeze().cpu().numpy()
        }