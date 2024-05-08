import torch
import torchvision
import torchvision.models as models
from icecream import ic
import re


def get_convnet_input_output(model: torch.nn.Module,
                             weights: torchvision.models.Weights):
    """
    Retrieves the input shape and output feature shape of a convolutional neural network (CNN).

    Args:
    - model (torch.nn.Module): Pretrained CNN model instance.
    - weights (torchvision.models.Weights): Predefined weights for the model.

    Returns:
    - input_shape (tuple): The shape of the input tensor to the CNN.
    - output_feature_shape (int): The dimensionality of the output features of the CNN.
    """

    def hook(module, input, output):
        """
        Hook function to store the output shape of a specific layer in the CNN.
        """
        global feature_shape
        feature_shape = output.shape

    # get the transforms from the pretrained model
    transforms = weights.transforms

    # Extract the crop_size from transforms using regex
    string = str(transforms)
    pattern = r"crop_size=(\d+)"
    match = re.search(pattern, string)
    if match:
        crop_value = int(match.group(1))
    else:
        crop_value = None

    # Set the model to evaluation mode
    model.eval()

    # Define input shape based on crop_size
    input_shape = (1, 3, crop_value, crop_value) if crop_value is not None else None

    # Register hook to get output shape
    hook_handle = model.avgpool.register_forward_hook(hook)

    # Forward pass the input through the model
    with torch.no_grad():
        features = model(torch.randn(input_shape))

    # Remove the hook
    hook_handle.remove()

    # Return input shape and output feature shape
    return input_shape, feature_shape[1]

# Dictionary mapping model names to their corresponding torchvision models and weights
pretrained_models: dict[str, dict] = {
    'EfficientNet': {
        'model': torchvision.models.efficientnet_b3,
        'weights': torchvision.models.EfficientNet_B3_Weights.DEFAULT
    }
}

# Get the weights and transformation function for the specified pretrained model
weights = pretrained_models['EfficientNet']['weights']
transforms = weights.transforms()  # Extract transformation function


# Setup the model with pretrained weights and send it to the target device
model = pretrained_models['EfficientNet']['model'](weights=weights)


# sample use of the function
input_shape, feature_vectors = get_convnet_input_output(model,
                                                        weights)
ic(input_shape, feature_vectors)