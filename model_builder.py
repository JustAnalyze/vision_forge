from typing import Callable, Tuple
import torchvision
import torch

def build_model(pretrained_model: str,
                num_hidden_units: int,
                output_shape: int,
                device: str) -> Tuple[Callable, torch.nn.Module]:
    """
    Build a neural network model using a pretrained model as a feature extractor.
    
    Args:
        pretrained_model (str): Name of the pretrained model to use.
        num_hidden_units (int): Number of hidden units in the new classifier layer.
        output_shape (int): Number of output units in the new classifier layer.
        device (str): Device where the model will be loaded ('cpu' or 'cuda').
        
    Returns:
        model: Pretrained neural network model.
        transforms: A callable transformation function that preprocesses input data.
    """
    
    # Dictionary mapping model names to their corresponding torchvision models and weights
    pretrained_models = {
        'EfficientNet': {
            'model': torchvision.models.efficientnet_b3,\
            'weights': torchvision.models.EfficientNet_B3_Weights.DEFAULT
        }
    }
    
    # Get the weights and transformation function for the specified pretrained model
    weights = pretrained_models[pretrained_model]['weights']
    transforms = weights.transforms()  # Extract transformation function

    
    # Setup the model with pretrained weights and send it to the target device
    model = pretrained_models[pretrained_model]['model'](weights=weights).to(device)

    # Uncomment the line below to output the model (it's very long)
    # print(model)
    
    # Freeze all base layers in the "features" section of the model (the feature extractor)
    # by setting requires_grad=False
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True,),
        torch.nn.Linear(in_features=num_hidden_units,
                        out_features=output_shape,  # use the length of class_names (one output unit for each class)
                        bias=True)).to(device)
    
    return  model, transforms

# Example usage:
model, transforms = build_model('EfficientNet', num_hidden_units=64, output_shape=3, device='cpu')
type(model)