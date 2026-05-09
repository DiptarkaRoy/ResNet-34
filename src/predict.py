import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import resnet34  # Import your architecture

def predict(image_path, model_path):
    # 1. Set Device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    # 2. Define Classes (Must match CIFAR-10 order)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 3. Load the Model Architecture
    model = resnet34(num_classes=10)
    
    # 4. Load the Saved Weights
    # Using weights_only=True for security and speed
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() # CRITICAL: Sets Batchnorm to evaluation mode

    # 5. Prepare the Image
    # Must match the transformations used in training (Resize and Normalize)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device) # Add batch dimension [1, 3, 128, 128]

    # 6. Run Inference
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        conf, predicted = torch.max(probabilities, 1)

    # 7. Output Results
    class_name = classes[predicted.item()]
    confidence = conf.item() * 100
    
    print(f"Prediction: {class_name} ({confidence:.2f}%)")
    return class_name, confidence

if __name__ == "__main__":
    # Example usage
    # Ensure you have an image file to test with
    IMG_PATH = "test_image.jpg" 
    MODEL_PATH = "models/resnet34_cifar10_v1.pth"
    
    try:
        predict(IMG_PATH, MODEL_PATH)
    except FileNotFoundError:
        print(f"Please provide a valid image path at {IMG_PATH}")