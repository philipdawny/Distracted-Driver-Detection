import matplotlib.pyplot as plt
import numpy as np

def display_test_results(images, predictions, true_labels, class_labels):
    
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(20, 2))
    
    for idx in range(num_images):
        ax = axes[idx]
        img = images[idx].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        color = "green" if predictions[idx] == true_labels[idx] else "red"
        ax.set_title(f"Pred: {class_labels[predictions[idx]]}\nTrue: {class_labels[true_labels[idx]]}", 
                    color=color, fontsize=8)
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()
