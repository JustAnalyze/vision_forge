import torch
import torchvision
import torchvision.models as models
from icecream import ic


# Define a hook function to store the output shape
def hook(module, input, output):
    global feature_shape
    feature_shape = output.shape

# Load a pretrained ResNet model
model = models.efficientnet_b3(pretrained=True)
weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
transforms = weights.transforms

# FIXME: Get convnet input shape.
# input_size = model.input_size
# ic("Input size of the pre-trained ResNet model:", input_size)

# Set the model to evaluation mode
model.eval()

# Get the input shape of the model
input_shape = (1, 3, 300, 300)  # Assuming input image size is 224x224 and has 3 channels (RGB)

# Register the hook to the desired layer
hook_handle = model.avgpool.register_forward_hook(hook)

# Forward pass the input through the model
with torch.no_grad():
    features = model(torch.randn(input_shape))

# Remove the hook
hook_handle.remove()

# Print the shape of the features
ic("Shape of the Output features:", feature_shape[1])
