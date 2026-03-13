import torch
import numpy as np
import random
import json
import pandas as pd
import os
from PIL import Image, ImageOps
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from torch import nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class GestureDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for loading and preprocessing hand gesture images from HaGRID dataset.
    Handles annotation parsing, train/test splitting by user, and image cropping/augmentation.
    """

    def __init__(self, path, class_names, is_train=True, transform=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.path = path
        self.is_train = is_train
        self.transform = transform
        self.class_names = class_names
        self.labels = {label: num for (label, num) in
                         zip(class_names, range(len(class_names)))}
        
        self.leading_hand = {'right': 0, 'left': 1}  
        self.annotations = self.__read_annotations(self.path)
        
        # Split users for train/test (80/20)
        users = self.annotations['user_id'].unique()
        users = sorted(users)
        random.Random(42).shuffle(users)
        
        train_users = users[:int(len(users) * 0.8)]
        test_users = users[int(len(users) * 0.8):]

        self.annotations = self.annotations.copy()
        if self.is_train:
            train_users_annotations = self.annotations['user_id'].isin(train_users) 
            self.annotations = self.annotations[train_users_annotations]
        else:
            test_users_annotations = self.annotations['user_id'].isin(test_users)
            self.annotations = self.annotations[test_users_annotations]

    def __read_annotations(self, path):
        # Load and process JSON annotations
        json_annotations = json.load(open(os.path.join(self.path, "ann_subsample.json")))
        json_annotations_with_names = []
        for name, ann in json_annotations.items():
            ann_with_name = ann.copy()
            ann_with_name['name'] = f'{name}.jpg'
            json_annotations_with_names.append(ann_with_name)
        annotations = pd.DataFrame(json_annotations_with_names)
        labels = list(annotations['labels'])
        targets = []
        for label in labels:
            targets.append([item for item in label if item != 'no_gesture'][0])

        annotations['target'] = targets
        return annotations

    def __prepare_image_target(self, target, name, bboxes, labels, leading_hand):
        """
        Prepare a single image sample by cropping to gesture bounding box and resizing.
        
        Args:
            target: Gesture class name
            name: Image filename
            bboxes: List of bounding boxes
            labels: List of labels corresponding to bboxes
            leading_hand: Which hand is leading ('left' or 'right')
            
        Returns:
            tuple: (processed_image, gesture_label, leading_hand_label)
        """
        image_pth = os.path.join(self.path, target, name)
        image = Image.open(image_pth).convert('RGB')
        width, height = image.size

        # Randomly choose between gesture and no_gesture if available
        choice = np.random.choice(['gesture', 'no_gesture'], p=[0.7, 0.3])
        bboxes_by_class = {}
        for i, bbox in enumerate(bboxes):
            x1, y1, w, h = bbox
            bbox_abs = [x1 * width, y1 * height, (x1 + w) * width, (y1 + h) * height]

            if labels[i] == 'no_gesture':
                bboxes_by_class['no_gesture'] = (bbox_abs, labels[i])
            else:
                bboxes_by_class['gesture'] = (bbox_abs, labels[i])

        if choice not in bboxes_by_class:
            choice = list(bboxes_by_class.keys())[0]

        box_scale = 1.0
        image_cropped, bbox_orig = self.get_crop_from_bbox(image, 
                                                           bboxes_by_class[choice][0], 
                                                           box_scale=box_scale)
        image_resized = ImageOps.pad(image_cropped, tuple([224, 224]), color=(0, 0, 0))

        gesture = bboxes_by_class[choice][1]
        leading_hand_class = leading_hand

        if gesture == 'no_gesture':
            leading_hand_class = 'right' if leading_hand == 'left' else 'left'

        return image_resized, gesture, leading_hand_class
      
    @staticmethod
    def get_crop_from_bbox(image, bbox, box_scale=1.):
        int_bbox = np.array(bbox).round().astype(np.int32)
        x1 = int_bbox[0]
        y1 = int_bbox[1]
        x2 = int_bbox[2]
        y2 = int_bbox[3] 
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        w = h = max(x2 - x1, y2 - y1)
        x1 = max(0, cx - box_scale * w // 2)
        y1 = max(0, cy - box_scale * h // 2)
        x2 = cx + box_scale * w // 2
        y2 = cy + box_scale * h // 2
        x1, y1, x2, y2 = list(map(int, (x1, y1, x2, y2)))

        crop_image = image.crop((x1, y1, x2, y2))
        bbox_orig = np.array([x1, y1, x2, y2]).reshape(2, 2)

        return crop_image, bbox_orig

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, index):
        row = self.annotations.iloc[[index]].to_dict('records')[0]
        image_resized, gesture, leading_hand = self.__prepare_image_target(
            row['target'],
            row['name'],
            row['bboxes'],
            row['labels'],
            row['leading_hand']
        )

        label = {'gesture': self.labels[gesture],
                 'leading_hand': self.leading_hand[leading_hand]}

        if self.transform is not None:
            image_resized = self.transform(image_resized)
        
          # Convert label dictionary to PyTorch tensors
        label = {key: torch.tensor(value, device=self.device) for key, value in label.items()}
        image_resized = image_resized.to(self.device)
        return image_resized, label
    
class ImageLoader:
  def __init__(self, folder_path, batch_size=32):
      # Define the transformations to apply to the images
      transform = transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])

      # Create an instance of the ImageFolder dataset
      self.dataset = ImageFolder(root=folder_path, transform=transform)

      # Create a DataLoader to efficiently load and batch the images
      self.batch_size = batch_size
      self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

  def get_dataloader(self):
      return self.dataloader

  def get_class_names(self):
      return self.dataset.classes

  def get_class_indices(self):
      return self.dataset.class_to_idx

    
class ToTensor(nn.Module):
    @staticmethod
    def forward(image):
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image

def get_transform():
    transforms = [ToTensor()]
    return Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))