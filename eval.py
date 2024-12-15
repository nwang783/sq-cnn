import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import os

# Assuming your model is based on a pre-trained ResNet18 (modify this to your actual model)
model = models.resnet18(pretrained=False)  # Or use the model architecture you used
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming a binary classification (squirrel vs. no squirrel)

# Load the trained model weights (replace 'squirrel_classifier.pth' with your model file)
model.load_state_dict(torch.load('squirrel_classifier_v3.pth'))
model.eval()

# define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to match the input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize using the same stats as training
])

# load and preprocess the image
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)

    # Check if GPU is available and move the model and input tensor to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    model.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

def eval_prediction(images_dir):
    images = [img for img in os.listdir(images_dir) if img.endswith(('.jpg'))]
    total_correct = 0
    for img in images:
        perdiction = predict_image(os.path.join(images_dir, img))
        if perdiction == 1:
            print("CORRECT!")
            total_correct += 1
        else:
            print("INCORRECT XXX")

    return total_correct / len(images)

percent_correct = eval_prediction("data2/train/squirrel")
print(f"The percent correct for this directory was: {100 * percent_correct:.2f}%")
        