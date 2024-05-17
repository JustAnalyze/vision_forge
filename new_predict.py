from pathlib import Path
import torch
import torchvision
from PIL import Image
import json


def predict_with_model(image_path, model_path):
    # Dictionary of pretrained models
    pretrained_models: dict[str, dict] = {
        'mobilenet_v2': {
            'model': torchvision.models.mobilenet_v2,
            'weights': torchvision.models.MobileNet_V2_Weights.DEFAULT
        },
        'efficientnet_b0': {
            'model': torchvision.models.efficientnet_b0,
            'weights': torchvision.models.EfficientNet_B0_Weights.DEFAULT
        },
        'efficientnet_b1': {
            'model': torchvision.models.efficientnet_b1,
            'weights': torchvision.models.EfficientNet_B1_Weights.DEFAULT
        },
        'efficientnet_b2': {
            'model': torchvision.models.efficientnet_b2,
            'weights': torchvision.models.EfficientNet_B2_Weights.DEFAULT
        },
        'efficientnet_b3': {
            'model': torchvision.models.efficientnet_b3,
            'weights': torchvision.models.EfficientNet_B3_Weights.DEFAULT
        },
        'efficientnet_b4': {
            'model': torchvision.models.efficientnet_b4,
            'weights': torchvision.models.EfficientNet_B4_Weights.DEFAULT
        },
        'efficientnet_b5': {
            'model': torchvision.models.efficientnet_b5,
            'weights': torchvision.models.EfficientNet_B5_Weights.DEFAULT
        },
        'efficientnet_b6': {
            'model': torchvision.models.efficientnet_b6,
            'weights': torchvision.models.EfficientNet_B6_Weights.DEFAULT
        },
        'efficientnet_b7': {
            'model': torchvision.models.efficientnet_b7,
            'weights': torchvision.models.EfficientNet_B7_Weights.DEFAULT
        },
    }
    
    # convert model_path from str to Path
    model_path = Path(model_path)
    
    # Load the trained model
    model = torch.load(model_path)
    
    # Get settings path from model_path
    settings_path = model_path.parent / 'settings.json'
    
    # Load settings from the JSON file
    with open(settings_path, 'r') as f:
        settings = json.load(f)
    
    # Get the name of the pretrained model
    pretrained_model_name = settings['model_settings']['pretrained_model']
    
    # Get the weights and transformation function for the specified pretrained model
    weights = pretrained_models[pretrained_model_name]['weights']
    transformation_fn = weights.transforms()  # Extract transformation function
    
    # Apply transforms to the input image
    input_image = transformation_fn(Image.open(image_path))
    
    # Perform inference
    with torch.inference_mode():
        pred_logits = model(input_image.unsqueeze(0))

    # get the probabilities of the prediction.
    pred_probs = torch.softmax(pred_logits, dim=1)

    # get the predicted label
    pred_label = torch.argmax(pred_probs, dim=1)
    
    print(f'Predicted Label: {int(pred_label)} | Predicted Probability: {pred_probs.max() * 100}')
    
    return pred_label

# Sample use
image_path = r'dataset\pizza_steak_sushi\test\pizza\194643.jpg'
model_path = r'MobileNetV2_training_output_2024-05-17_17-33-45\model.pth'
output = predict_with_model(image_path, model_path)
print(output)