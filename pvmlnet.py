import sys
import os
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Ensure the current directory is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import low-level feature extraction functions from image_features.py
    from image_features import color_histogram, edge_direction_histogram, cooccurrence_matrix, rgb_cooccurrence_matrix # type: ignore

    # Print statement to check if import is working
    print("Successfully imported image_features module!")

except ImportError as e:
    print(f"Failed to import a module: {e}")

# Define the PVMLNet model
class PVMLNet(nn.Module):
    def __init__(self):
        super(PVMLNet, self).__init__()
        self.model = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
        self.layer_outputs = []
        def hook(module, input, output):
            self.layer_outputs.append(output)
        # Register hooks to the desired layers
        self.model.features[4].register_forward_hook(hook)  # Example layer
        self.model.features[8].register_forward_hook(hook)  # Example layer
        self.features = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        self.layer_outputs = []
        _ = self.features(x)
        return self.layer_outputs

# Define the custom dataset
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = plt.imread(self.image_paths[idx])
        image = Image.fromarray((image * 255).astype(np.uint8))
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Load the dataset
def load_data(train_dir, test_dir, transform):
    image_extensions = ('.jpg', '.jpeg', '.png')
    train_image_paths = []
    train_labels = []
    test_image_paths = []
    test_labels = []

    for label, subdir in enumerate(os.listdir(train_dir)):
        subdir_path = os.path.join(train_dir, subdir)
        for file_name in os.listdir(subdir_path):
            if file_name.lower().endswith(image_extensions):
                train_image_paths.append(os.path.join(subdir_path, file_name))
                train_labels.append(label)

    for label, subdir in enumerate(os.listdir(test_dir)):
        subdir_path = os.path.join(test_dir, subdir)
        for file_name in os.listdir(subdir_path):
            if file_name.lower().endswith(image_extensions):
                test_image_paths.append(os.path.join(subdir_path, file_name))
                test_labels.append(label)

    train_dataset = CustomDataset(train_image_paths, train_labels, transform)
    test_dataset = CustomDataset(test_image_paths, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

# Extract low-level features
def extract_low_level_features(image_paths):
    features = []
    for img_path in image_paths:
        img = plt.imread(img_path)
        h = color_histogram(img)
        e = edge_direction_histogram(img)
        c = cooccurrence_matrix(img)
        r = rgb_cooccurrence_matrix(img)
        combined_features = np.concatenate((h.flatten(), e, c.flatten(), r.flatten()))
        features.append(combined_features)
    return np.array(features)

# Extract features using PVMLNet
def extract_neural_features(model, loader):
    features = []
    labels = []
    with torch.no_grad():
        for images, target in loader:
            outputs = model(images)
            layer4_output = outputs[0].view(images.size(0), -1)  # Example for layer 4
            layer8_output = outputs[1].view(images.size(0), -1)  # Example for layer 8
            combined_features = torch.cat((layer4_output, layer8_output), dim=1)
            features.append(combined_features)
            labels.append(target)
    return torch.cat(features), torch.cat(labels)

# Define a combined model for training
class CombinedModel(nn.Module):
    def __init__(self, neural_input_size, low_level_input_size, num_classes):
        super(CombinedModel, self).__init__()
        self.fc1 = nn.Linear(neural_input_size + low_level_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, neural_features, low_level_features):
        x = torch.cat((neural_features, low_level_features), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train the model
def train_combined_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for neural_features, low_level_features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(neural_features, low_level_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return train_losses

# Evaluate the model
def evaluate_combined_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for neural_features, low_level_features, labels in loader:
            outputs = model(neural_features, low_level_features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())
    accuracy = correct / total
    return accuracy, all_labels, all_predictions

# Main script
if __name__ == "__main__":
    train_dir = "D:\\UNIPV\\Year 1\\Semester 2\\Machine Learning\\cake\\cake-images\\train"
    test_dir = "D:\\UNIPV\\Year 1\\Semester 2\\Machine Learning\\cake\\cake-images\\test"

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load data
    train_loader, test_loader = load_data(train_dir, test_dir, transform)

    # Load PVMLNet model
    pvmlnet = PVMLNet()
    pvmlnet.eval()

    # Extract neural features
    train_neural_features, train_labels = extract_neural_features(pvmlnet, train_loader)
    test_neural_features, test_labels = extract_neural_features(pvmlnet, test_loader)

    # Extract low-level features
    train_image_paths = [train_loader.dataset.image_paths[i] for i in range(len(train_loader.dataset))]
    test_image_paths = [test_loader.dataset.image_paths[i] for i in range(len(test_loader.dataset))]

    train_low_level_features = extract_low_level_features(train_image_paths)
    test_low_level_features = extract_low_level_features(test_image_paths)

    # Normalize low-level features
    def normalize_features(features):
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        return (features - mean) / std

    train_low_level_features = normalize_features(train_low_level_features)
    test_low_level_features = normalize_features(test_low_level_features)

    # Convert features and labels to tensors
    train_neural_features = train_neural_features.numpy()
    test_neural_features = test_neural_features.numpy()
    train_low_level_features = torch.tensor(train_low_level_features, dtype=torch.float32)
    test_low_level_features = torch.tensor(test_low_level_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels.numpy(), dtype=torch.long)
    test_labels = torch.tensor(test_labels.numpy(), dtype=torch.long)

    # Create a custom dataset for the combined features
    class CombinedDataset(Dataset):
        def __init__(self, neural_features, low_level_features, labels):
            self.neural_features = neural_features
            self.low_level_features = low_level_features
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            neural_feature = self.neural_features[idx]
            low_level_feature = self.low_level_features[idx]
            label = self.labels[idx]
            return neural_feature, low_level_feature, label

    train_combined_dataset = CombinedDataset(train_neural_features, train_low_level_features, train_labels)
    test_combined_dataset = CombinedDataset(test_neural_features, test_low_level_features, test_labels)

    train_combined_loader = DataLoader(train_combined_dataset, batch_size=32, shuffle=True)
    test_combined_loader = DataLoader(test_combined_dataset, batch_size=32, shuffle=False)

    # Define the combined model
    neural_input_size = train_neural_features.shape[1]
    low_level_input_size = train_low_level_features.shape[1]
    num_classes = len(set(train_labels.numpy()))
    combined_model = CombinedModel(neural_input_size, low_level_input_size, num_classes)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)

    # Evaluate before fine-tuning
    initial_accuracy, _, _ = evaluate_combined_model(combined_model, test_combined_loader)
    print(f"Initial Test Accuracy: {initial_accuracy:.4f}")

    # Train the combined model
    train_losses = train_combined_model(combined_model, train_combined_loader, criterion, optimizer, num_epochs=10)

    # Evaluate the combined model
    test_accuracy, all_labels, all_predictions = evaluate_combined_model(combined_model, test_combined_loader)
    print(f"Test Accuracy after Fine-Tuning: {test_accuracy:.4f}")

    # Plot training loss
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

    # Confusion matrix and analysis
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Identify commonly confused classes
    confused_pairs = np.argwhere(cm > np.mean(cm))
    for i, j in confused_pairs:
        if i != j:
            print(f"Class {i} is often confused with Class {j}")

    # Misclassified samples analysis
    misclassified_indices = np.where(np.array(all_predictions) != np.array(all_labels))[0]
    print("Misclassified samples:")
    for idx in misclassified_indices:
        print(f"Image: {test_image_paths[idx]}, Predicted: {all_predictions[idx]}, Actual: {all_labels[idx]}")
