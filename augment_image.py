import os
import torch
from torchvision import transforms
from PIL import Image
import random

# input and output directories
input_dir = "squirrel_images"
output_dir = "augmented_squirrel_images"
os.makedirs(output_dir, exist_ok=True)

# Define augmentation pipeline
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    # transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def save_image(tensor, path):
    tensor = denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    tensor = torch.clamp(tensor, 0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(path)

# Load and Augment
num_augmentations = 5
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpeg', '.jpg'))]

num_images = len(image_files) * 6
j = 0

for img in image_files:
    img_path = os.path.join(input_dir, img)
    image = Image.open(img_path).convert('RGB')

    # Save original image
    save_image(transforms.ToTensor()(image), os.path.join(output_dir, f"squirrel_{j}_originial.jpg"))
    print(f"Saving original image. {j}/{num_images}")
    j += 1

    # Apply Augmentations
    for i in range(num_augmentations):
        augmented_image = augmentations(image)
        file_name = os.path.join(output_dir, f"squirrel_{j}_aug_{i}.jpg")
        save_image(augmented_image, file_name)
        print(f"Saving augmented image: {file_name}. {j}/{num_images}")
        j += 1

