import torch
import matplotlib.pyplot as plt
import numpy as np
from toolbox.data_loader import Cifar10, Cifar100
from toolbox.models import ResNet112, ResNet56, ResNet20, ResNetBaby

# Assuming your ResNet code and DataHelper code are available

def visualize_feature_maps(model, data_helper, num):
    # Set model to evaluation mode and move to cuda
    model.eval()
    model = model.to('cuda')
    
    # Get one batch from testloader
    iterdata = iter(data_helper.testloader)
    for i in range(num + 1):
        images, labels = next(iterdata)

    input_tensor = images[0].unsqueeze(0).to('cuda')  # Take first image, add batch dimension

    class_idx = labels[0].item()
    class_name = data_helper.testloader.dataset.classes[class_idx] if hasattr(data_helper.testloader.dataset, 'classes') else f"Class {class_idx}"
    print(f"Correct class for this image: {class_name}") 

    # Get feature maps
    with torch.no_grad():
        feature_maps = model(input_tensor)
    

    output = feature_maps[-1]  # Last element is the classification output
    probabilities = torch.softmax(output, dim=1)  # Convert logits to probabilities
    predicted_idx = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, predicted_idx].item() * 100  # Convert to percentage
    predicted_name = (data_helper.testloader.dataset.classes[predicted_idx] 
                     if hasattr(data_helper.testloader.dataset, 'classes') 
                     else f"Class {predicted_idx}")
    print(f"Model predicted class: {predicted_name} (Confidence: {confidence:.2f}%)")

    # Function to plot feature maps
    def plot_feature_maps(feature_map, layer_name, num_features=8):
        fmap = feature_map.cpu().numpy()[0]  # Remove batch dimension
        num_fmaps = min(fmap.shape[0], num_features)
        
        fig = plt.figure(figsize=(15, 5))
        plt.suptitle(f'Feature Maps - {layer_name}')
        
        for i in range(num_fmaps):
            ax = fig.add_subplot(1, num_fmaps, i+1)
            fmap_i = fmap[i]
            fmap_i = (fmap_i - fmap_i.min()) / (fmap_i.max() - fmap_i.min() + 1e-8)
            ax.imshow(fmap_i, cmap='viridis')
            ax.axis('off')
            ax.set_title(f'FM {i}')
        
        plt.tight_layout()
        return fig
    
    # Visualize feature maps from different layers
    layer_names = ['Layer1', 'Layer2', 'Layer3']
    figures = []
    
    for fmap, layer_name in zip(feature_maps[:-1], layer_names):  # Exclude output layer
        fig = plot_feature_maps(fmap, layer_name)
        figures.append(fig)
    
    # Show original image
    original_img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)  # Convert CHW to HWC
    original_img = (original_img * np.array([0.247, 0.243, 0.261]) +  # Denormalize
                    np.array([0.4914, 0.4822, 0.4465]))
    original_img = np.clip(original_img, 0, 1)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Show all plots
    plt.show()

# Example usage



device = 'cuda'
import torch.nn.functional as F
def eval(model, data):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    for inputs, targets in data.testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs= model(inputs)
        loss = F.cross_entropy(outputs[3], targets)
        running_loss += loss.item()
        _, predicted = torch.max(outputs[3].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
    avg_loss = running_loss / len(data.testloader)
    accuracy = 100 * correct / total
    print(f'TEST | Loss: {avg_loss:.3f} | Acc: {accuracy:.2f} |')




def main():
    # Create model and data helper
    Data = Cifar100()  # Using your Cifar10 function
    Model = ResNet112(Data.class_num).to(device)

    checkpoint = torch.load(f'models/{Data.name}_{Model.model_type}.pth', weights_only=True)
    Model.load_state_dict(checkpoint['weights']) 

    eval(Model, Data)
    # Visualize feature maps
    for i in range(10):
        visualize_feature_maps(Model, Data, i )

if __name__ == "__main__":
    main()