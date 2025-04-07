from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import ViTImageProcessor
import torch

def get_vit_transforms(processor):
    
    size = processor.size["height"]
    
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=processor.image_mean, 
                std=processor.image_std
            )
        ]),
        "val": transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=processor.image_mean, 
                std=processor.image_std
            )
        ])
    }


# Custom collator for ViT model inputs
def collate_fn(batch):
    
    images, labels = zip(*batch)
    return {
        "pixel_values": torch.stack(images),
        "labels": torch.tensor(labels)
    }

def get_data_loaders(config):
    
    processor = ViTImageProcessor.from_pretrained(config["model_name"])
    transforms = get_vit_transforms(processor)
    
    train_dataset = ImageFolder(config["train_data_path"], transform=transforms["train"])
    val_dataset = ImageFolder(config["val_data_path"], transform=transforms["val"])
    
    return train_dataset, val_dataset, processor, train_dataset.class_to_idx
