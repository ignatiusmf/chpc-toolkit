import torch
import matplotlib.pyplot as plt
import numpy as np
from sandbox.toolbox.data_loader import Cifar10, Cifar100
from sandbox.toolbox.models import ResNet112, ResNet56, ResNet20, ResNetBaby

# Assuming your ResNet code and DataHelper code are available

def visualize_feature_maps(model112, model56, model20, modelbaby, data_helper, num):
    # Set all models to evaluation mode and move to cuda
    models = [model112, model56, model20, modelbaby]
    model_names = ['ResNet112', 'ResNet56', 'ResNet20', 'ResNetBaby']
    for model in models:
        model.eval()
        model = model.to('cuda')
    
    # Get one batch from testloader
    iterdata = iter(data_helper.testloader)
    for i in range(num + 1):
        images, labels = next(iterdata)

    input_tensor = images[0].unsqueeze(0).to('cuda')  # Take first image, add batch dimension

    # Print correct class
    class_idx = labels[0].item()
    class_name = data_helper.testloader.dataset.classes[class_idx] if hasattr(data_helper.testloader.dataset, 'classes') else f"Class {class_idx}"
    print(f"\nExample {num} - Correct class: {class_name}")

    # Get feature maps for all models
    feature_maps_list = []
    with torch.no_grad():
        for model in models:
            feature_maps = model(input_tensor)
            feature_maps_list.append(feature_maps)

    # Print predictions for each model
    for model_name, feature_maps in zip(model_names, feature_maps_list):
        output = feature_maps[-1]
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_idx].item() * 100
        predicted_name = (data_helper.testloader.dataset.classes[predicted_idx] 
                         if hasattr(data_helper.testloader.dataset, 'classes') 
                         else f"Class {predicted_idx}")
        print(f"{model_name} predicted class: {predicted_name} (Confidence: {confidence:.2f}%)")

    # Function to process feature maps (average and softmax)
    def process_feature_map(feature_map):
        fmap = feature_map.cpu()  # Shape: [1, C, H, W]
        fmap_avg = torch.mean(fmap, dim=1, keepdim=True)  # Shape: [1, 1, H, W]
        batch_size, _, h, w = fmap_avg.shape
        fmap_flat = fmap_avg.view(batch_size, -1)  # Shape: [1, H*W]
        tau = 0.5
        fmap_scaled = fmap_flat / tau
        fmap_softmax = torch.softmax(fmap_scaled, dim=1)  # Shape: [1, H*W]
        fmap_softmax = fmap_softmax * (h * w)  # Scale by H*W
        fmap_processed = fmap_softmax.view(1, 1, h, w)  # Shape: [1, 1, H, W]
        return fmap_processed[0, 0].numpy()  # Shape: [H, W]

    # Prepare original image
    original_img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)  # Convert CHW to HWC
    original_img = (original_img * np.array([0.247, 0.243, 0.261]) +  # Denormalize
                    np.array([0.4914, 0.4822, 0.4465]))
    original_img = np.clip(original_img, 0, 1)

    # Create a single plot for all feature maps and the original image
    layer_names = ['Layer1', 'Layer2', 'Layer3']
    fig = plt.figure(figsize=(20, 20))
    plt.suptitle(f'Example {num} - Averaged Feature Maps with Softmax and Original Image')

    # Plot feature maps for each layer and model
    for layer_idx, layer_name in enumerate(layer_names):
        for model_idx, (feature_maps, model_name) in enumerate(zip(feature_maps_list, model_names)):
            # Plot feature map at grid position (layer_idx, model_idx)
            ax = fig.add_subplot(4, 4, (layer_idx * 4) + model_idx + 1)  # 4 rows, 4 cols
            fmap_processed = process_feature_map(feature_maps[layer_idx])
            ax.imshow(fmap_processed, cmap='viridis')
            ax.axis('off')
            # Add title: model name for first row, layer name for first column
            if layer_idx == 0:
                ax.set_title(model_name)
            if model_idx == 0:
                ax.text(-0.5, 0.5, layer_name, va='center', ha='right', fontsize=12, transform=ax.transAxes)

    # Plot original image in the fourth row, spanning all columns
    ax = fig.add_subplot(4, 4, (3 * 4) + 1)  # Start at position 13
    ax.imshow(original_img)
    ax.axis('off')
    ax.set_title('Original Image')
    # Remove the unused subplots in the fourth row
    for col in range(1, 4):
        fig.add_subplot(4, 4, (3 * 4) + col + 1).axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust for suptitle
    plt.show()

device = 'cuda'
import torch.nn.functional as F
def eval(model, data):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    for inputs, targets in data.testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs[3], targets)
        running_loss += loss.item()
        _, predicted = torch.max(outputs[3].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
    avg_loss = running_loss / len(data.testloader)
    accuracy = 100 * correct / total
    print(f'TEST | Loss: {avg_loss:.3f} | Acc: {accuracy:.2f} |')

def load_model(Model, Data):
    loaded_model = Model(Data.class_num).to(device)
    checkpoint = torch.load(f'models/{Data.name}_{loaded_model.model_type}.pth', weights_only=True)
    loaded_model.load_state_dict(checkpoint['weights']) 
    return loaded_model

def main():
    # Create model and data helper
    Data = Cifar100()  # Using your Cifar10 function
    resnet112 = load_model(ResNet112, Data)
    resnet56 = load_model(ResNet56, Data)
    resnet20 = load_model(ResNet20, Data)
    resnetbaby = load_model(ResNetBaby, Data)

    eval(resnet112, Data)
    # Visualize feature maps
    for i in range(10):  # For the first ten examples in the dataset
        visualize_feature_maps(resnet112, resnet56, resnet20, resnetbaby, Data, i)

if __name__ == "__main__":
    main()