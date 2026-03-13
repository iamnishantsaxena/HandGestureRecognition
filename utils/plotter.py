import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def plot_test_results(images, predicted_gesture, predicted_leading_hand, class_names, num_samples=9):
    """
    Visualize model predictions on test images.
    
    Args:
        images: Batch of test images (tensor)
        predicted_gesture: Predicted gesture class indices
        predicted_leading_hand: Predicted leading hand indices
        class_names: List of gesture class names
        num_samples: Number of samples to display (default: 9)
    """
    # Create a grid of images
    grid = make_grid(images[:num_samples], nrow=3, padding=10, pad_value=1)

    # Create a subplot for plotting
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle('Test Results', fontsize=16)

    for i in range(num_samples):
        # Unnormalize the image to display it correctly
        image = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        # Get the predicted class names
        predicted_gesture_name = class_names[predicted_gesture[i]]
        predicted_leading_hand_name = class_names[predicted_leading_hand[i]]

        # Plot the image and its predicted class
        ax = axs[i // 3, i % 3]
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f'Gesture: {predicted_gesture_name}\nLeading Hand: {predicted_leading_hand_name}', fontsize=10)

    plt.show()