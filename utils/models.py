import torch
from torch import nn, Tensor
from torchvision import models
from collections import defaultdict
from sklearn.metrics import f1_score
from torchmetrics.functional import f1_score, confusion_matrix
import pandas as pd
from torchvision import transforms
from PIL import Image
import cv2
import tensorflow as tf
import h5py

# Custom VGG19 model for gesture recognition
class VGG19(nn.Module):
    def __init__(self, num_classes, pretrained=False, freezed=False):
        super().__init__()
        torchvision_model = models.vgg19(pretrained=pretrained)

        if freezed:
            for param in torchvision_model.parameters():
                param.requires_grad = False

        self.features = torchvision_model.features

        num_features = 25088  # Number of output features for VGG19
        
        # Classifier for gesture prediction
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        # Classifier for leading hand prediction (left/right)
        self.leading_hand = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2)  # 2 classes: left or right
        )

    def forward(self, img):
        x = self.features(img)
        x = x.view(x.size(0), -1)

        gesture = self.classifier(x)
        leading_hand = self.leading_hand(x)

        return {'gesture': gesture, 'leading_hand': leading_hand}

class VGG16(nn.Module):
    def __init__(self, num_classes, pretrained=False, freezed=False):
        super().__init__()
        torchvision_model = models.vgg16(pretrained=pretrained)

        if freezed:
            for param in torchvision_model.parameters():
                param.requires_grad = False

        self.features = torchvision_model.features

        num_features = 25088  # Number of output features for VGG16
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        self.leading_hand = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2)
        )

    def forward(self, img):
        x = self.features(img)
        x = x.view(x.size(0), -1)

        gesture = self.classifier(x)
        leading_hand = self.leading_hand(x)

        return {'gesture': gesture, 'leading_hand': leading_hand}


class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=False, freezed=False):
        super().__init__()
        torchvision_model = models.resnet50(pretrained=pretrained)

        if freezed:
            for param in torchvision_model.parameters():
                param.requires_grad = False

        self.backbone = nn.Sequential(
            torchvision_model.conv1,
            torchvision_model.bn1,
            torchvision_model.relu,
            torchvision_model.maxpool,
            torchvision_model.layer1,
            torchvision_model.layer2,
            torchvision_model.layer3,
            torchvision_model.layer4,
            torchvision_model.avgpool
        )

        num_features = torchvision_model.fc.in_features

        self.classifier = nn.Sequential(nn.Linear(num_features, num_classes))
        self.leading_hand = nn.Sequential(nn.Linear(num_features, 2))

    def forward(self, img):
        x = self.backbone(img)
        x = torch.flatten(x, 1)

        gesture = self.classifier(x)
        leading_hand = self.leading_hand(x)

        return {'gesture': gesture, 'leading_hand': leading_hand}


class DenseNet(nn.Module):
    def __init__(self, num_classes, pretrained=False, freezed=False):
        super().__init__()
        torchvision_model = models.densenet121(pretrained=pretrained)

        if freezed:
            for param in torchvision_model.parameters():
                param.requires_grad = False

        self.backbone = nn.Sequential(
            torchvision_model.features,
            nn.AdaptiveAvgPool2d(1)
        )

        num_features = torchvision_model.classifier.in_features

        self.classifier = nn.Sequential(nn.Linear(num_features, num_classes))
        self.leading_hand = nn.Sequential(nn.Linear(num_features, 2))

    def forward(self, img):
        x = self.backbone(img)
        x = torch.flatten(x, 1)

        gesture = self.classifier(x)
        leading_hand = self.leading_hand(x)

        return {'gesture': gesture, 'leading_hand': leading_hand}

#####################################################################################################
#####################################################################################################
class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=False, freezed=False):
        super().__init__()
        torchvision_model = models.resnet18(pretrained=pretrained)

        if freezed:
            for param in torchvision_model.parameters():
                param.requires_grad = False

        self.backbone = nn.Sequential(
            torchvision_model.conv1,
            torchvision_model.bn1,
            torchvision_model.relu,
            torchvision_model.maxpool,
            torchvision_model.layer1,
            torchvision_model.layer2,
            torchvision_model.layer3,
            torchvision_model.layer4,
            torchvision_model.avgpool
        )

        num_features = torchvision_model.fc.in_features

        self.classifier = nn.Sequential(nn.Linear(num_features, num_classes))
        self.leading_hand = nn.Sequential(nn.Linear(num_features, 2))

    def forward(self, img):
        x = self.backbone(img)
        x = torch.flatten(x, 1)
        
        gesture = self.classifier(x)
        leading_hand = self.leading_hand(x)

        return {'gesture': gesture, 'leading_hand': leading_hand}

    
#####################################################################################################
class MarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MarginLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        batch_size = output.size(0)
        correct_score = output[range(batch_size), target].view(-1, 1)
        margin_loss = (output - correct_score + self.margin).clamp(min=0)
        margin_loss[range(batch_size), target] = 0
        margin_loss = margin_loss.sum(dim=1).mean()
        return margin_loss


class train_eval_model:
    def __init__(self, num_classes, model_name, Losstype):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_name == 'ResNet50':
            self.model = ResNet50(num_classes=num_classes, pretrained=True, freezed=True).to(self.device)
        elif model_name == 'ResNet18':
            self.model = ResNet18(num_classes=num_classes, pretrained=True, freezed=True).to(self.device)
        elif model_name == 'VGG16':
            self.model = VGG16(num_classes=num_classes, pretrained=True, freezed=True).to(self.device)
        elif model_name == 'VGG19':
            self.model = VGG19(num_classes=num_classes, pretrained=True, freezed=True).to(self.device)
        elif model_name == 'Densenet':
            self.model = DenseNet(num_classes=num_classes, pretrained=True, freezed=True).to(self.device)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(self.params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        if Losstype == "margin":
          self.criterion = MarginLoss()
        else:
           self.criterion = nn.CrossEntropyLoss() 
        self.train_loss = []
        self.train_accuracy = []
        self.number_classes = num_classes
        
        self.short_class_names = self.short_class_n()
        print("Loss type = ", self.criterion)
    def training(self, epochs, train_dataloader, print_epochs):
        """
        Train the model for a specified number of epochs.
        
        Args:
            epochs (int): Number of training epochs
            train_dataloader: PyTorch DataLoader for training data
            print_epochs (bool): Whether to print progress per epoch
            
        Returns:
            tuple: (train_accuracy_list, train_loss_list)
        """
        for epoch in range(epochs):
            self.model.train()
            
            f1_history = defaultdict(list)
            running_loss = 0.0
            loss_history = []
            correct_predictions = 0
            correct_predictions = torch.tensor(correct_predictions, device='cuda')
            total_samples = 0
            
            
            for i, (images, labels) in enumerate(train_dataloader):
                step = i + len(train_dataloader) * epoch

                images = torch.stack(list(image.to(self.device) for image in images))
                output = self.model(images)
                loss = []
                f1 = {}
                for target in list(labels)[0].keys():

                    target_labels = [label[target] for label in labels]
                    target_labels = torch.as_tensor(target_labels).to(self.device)
                    #loss.append(self.criterion(output[target], target_labels))
                    loss.append(self.criterion(output[target], target_labels))
                    f1[target] = float(f1_score(torch.argmax(output[target], dim=1), 
                                                target_labels, average='weighted', 
                                                num_classes=self.number_classes,
                                                task='multiclass'))
                    f1_history[target].append(f1[target])

                loss = sum(loss)
                loss_value = loss.item()
                
                loss_history.append(loss_value)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_samples += sum(entry['gesture'] for entry in labels)
                running_loss += loss.item()
                _, predicted = torch.max(output['gesture'], 1)
                predicted = predicted.to('cuda')
                labels_tensor = torch.tensor([label['gesture'] for label in labels], device='cuda')
                correct_predictions += (predicted == labels_tensor).sum().item()

            
            
            epoch_loss = round(sum(loss_history) / (i + 1), 3)
            epoch_f1_gesture = round(sum(f1_history['gesture']) / (i + 1), 3)
            epoch_f1_lh = round(sum(f1_history['leading_hand']) / (i + 1), 3)
            epoch_accuracy = correct_predictions / total_samples
            if print_epochs:
              print(f'epoch: {epoch}, loss: {epoch_loss}')
              print(f'f1 gestures: {epoch_f1_gesture}, f1 leading hand: {epoch_f1_lh}, epoch loss: {epoch_loss}, epoch accuracy: {epoch_accuracy}')
            self.train_loss.append(epoch_loss*0.1)
            self.train_accuracy.append(epoch_accuracy*10)
        return self.train_accuracy, self.train_loss
    
    def evaluate(self, test_dataloader):
        with torch.no_grad():
            self.model.eval()
            predicts, targets = defaultdict(list), defaultdict(list)
            total_loss = 0.0
            total_samples = 0
            correct_predictions = 0
            correct_predictions = torch.tensor(correct_predictions, device='cuda')

            print("=========start evaluate=========")
            for i, (images, labels) in enumerate(test_dataloader):
                
                images = torch.stack(list(image.to(self.device) for image in images))
                output = self.model(images)

                for target in list(labels)[0].keys():
                    target_labels = [label[target] for label in labels]
                    predicts[target] += list(output[target].detach().cpu().numpy())
                    targets[target] += target_labels
                # Compute loss
                labels_gesture = torch.tensor([label['gesture'] for label in labels], device=self.device)
                loss_gesture = self.criterion(output['gesture'], labels_gesture)
                total_loss += loss_gesture.item()
                # Compute accuracy
                _, predicted_gesture = torch.max(output['gesture'], 1)
                predicted_gesture = predicted_gesture.to('cuda')
                labels_tensor = torch.tensor([label['gesture'] for label in labels], device='cuda')
                correct_predictions += (predicted_gesture == labels_tensor).sum().item()
                total_samples += labels_gesture.size(0)
            
            f1 = {}
            for target in targets.keys():
                f1[target] = float(f1_score(torch.argmax(torch.tensor(predicts[target]), dim=1), 
                                            torch.tensor(targets[target]), average='weighted', 
                                            num_classes=self.number_classes,
                                            task='multiclass'))
                
            accuracy = correct_predictions / total_samples
            average_loss = total_loss / len(test_dataloader)

            print(f"f1 gestures: {round(f1['gesture'], 3)}")
            print(f"f1 leading hand: {round(f1['leading_hand'], 3)}")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Average Loss: {average_loss:.3f}")


        cm = confusion_matrix(torch.tensor(predicts['gesture']), 
                            torch.tensor(targets['gesture']), 
                            num_classes=self.number_classes,
                            task='multiclass')

        cm = cm.cpu()
        df_cm = pd.DataFrame(cm.numpy(), index=[i for i in self.short_class_names], 
                            columns=[i for i in self.short_class_names])
        return df_cm
    
    """def predict(self, data):
      images_return = []
      with torch.no_grad():
            self.model.eval()
            images = torch.stack(list(image.to(self.device) for image in data))
            output = self.model(images)
            for image in images:
              _, predicted_gesture = torch.max(output['gesture'], 1)
              images_return.append(image)
      return predicted_gesture, images_return"""



    
    def save_model(self, file_path):
        # Save the model weights to the file in PyTorch format
        torch.save(self.model,file_path)
        #torch.save(self.model.state_dict(), file_path)
        print("Model saved to file:", file_path)

    def load_model(self, file_path):
        # Load the model weights from the H5 file
        h5file = torch.load(file_path)
        with h5py.File(file_path, 'r') as h5_file:
            self.model.load_state_dict(torch.load(h5_file['model_weights']))
        print("Model loaded from H5 file:", file_path)

    def predict(self, data):
        images_return = []
        with torch.no_grad():
            self.model.eval()
            images = torch.stack(list(image.to(self.device) for image in data))
            output = self.model(images)
            for image in images:
                _, predicted_gesture = torch.max(output['gesture'], 1)
                images_return.append(image)
        return predicted_gesture, images_return

    def preprocess_image(self, image_path):
        # Define image preprocessing transformations
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load and preprocess the image
        image = Image.open(image_path)
        image = transform(image)
        image = image.unsqueeze(0)  # Add a batch dimension

        return image

            
    def short_class_n(self):
        class_names = ['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one',
               'palm', 'peace', 'rock', 'stop', 'stop_inverted', 'three', 'two_up', 
               'two_up_inverted', 'three2', 'peace_inverted', 'no_gesture']
        short_class_names = []
        for name in class_names:
            if name == 'stop_inverted':
                short_class_names.append('stop inv.')
            elif name == 'peace_inverted':
                short_class_names.append('peace inv.')
            elif name == 'two_up':
                short_class_names.append('two up')
            elif name == 'two_up_inverted':
                short_class_names.append('two up inv.')
            elif name == 'no_gesture':
                short_class_names.append('no gesture')
            else:
                short_class_names.append(name)
        return short_class_names

