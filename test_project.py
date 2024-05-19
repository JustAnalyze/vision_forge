import pytest
import torch
import torchvision.transforms.v2 as T
from torchmetrics.classification import MulticlassAccuracy
import json
from pathlib import Path
from project import (plot_loss_curves,
                     predict_with_model,
                     data_setup, build_model,
                     pretrained_models,
                     train_step, validation_step)


def test_predict_with_model():
    # Define paths
    image_path = Path('unit_test_files/test_image.jpg')
    model_path = Path('unit_test_files/test_model.pth')
    settings_path = Path('unit_test_files/settings.json')
    
    # Load settings to get the classes and model name for assertions
    with open(settings_path, 'r') as f:
        settings = json.load(f)
    classes = settings['data_settings']['classes']
    
    # Run the function for CPU
    predicted_class, probability, inference_duration = predict_with_model(
        image_path, model_path, torch.device('cpu')
    )
    
    # Assertions for CPU
    assert predicted_class in classes
    assert isinstance(probability, float)
    assert 0 <= probability <= 100
    assert isinstance(inference_duration, float)
    assert inference_duration >= 0

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Run the function for GPU
        predicted_class, probability, inference_duration = predict_with_model(
            image_path, model_path, torch.device('cuda')
        )
        
        # Assertions for GPU
        assert predicted_class in classes
        assert isinstance(probability, float)
        assert 0 <= probability <= 100
        assert isinstance(inference_duration, float)
        assert inference_duration >= 0


def test_data_setup():
    # Define paths and parameters
    data_path = 'unit_test_files/test_data'
    batch_size = 2
    device = torch.device('cpu')
    transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])

    # Run the function
    train_dataloader, test_dataloader, classes = data_setup(data_path, batch_size, device, transform)

    # Assertions
    assert len(classes) == 3 
    assert len(train_dataloader.dataset) == 6 
    assert len(test_dataloader.dataset) == 6 
    assert classes == ['pizza', 'steak', 'sushi'] 

    for batch in train_dataloader:
        inputs, labels = batch
        assert inputs.shape[0] == batch_size
        assert inputs.shape[1:] == torch.Size([3, 128, 128])
        break  # We just need to test the first batch

    for batch in test_dataloader:
        inputs, labels = batch
        assert inputs.shape[0] == batch_size
        assert inputs.shape[1:] == torch.Size([3, 128, 128])
        break  # We just need to test the first batch


def test_build_model():
    # Parameters for the test
    pretrained_model = 'mobilenet_v3_small'
    num_hidden_units = 128
    output_shape = 16
    
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        print(f"Testing on {device}...")

        # Ensure the pretrained model is in the predefined dictionary
        assert pretrained_model in pretrained_models

        # Call the function
        model, transforms = build_model(pretrained_model, num_hidden_units, output_shape, device)

        # Assertions
        assert isinstance(model, torch.nn.Module)
        assert callable(transforms)  # Ensure transforms is callable

        # Check if the model has been moved to the correct device1
        for param in model.parameters():
            assert param.device.type == device

        # Check the classifier layers
        assert isinstance(model.classifier[0], torch.nn.Linear)
        assert model.classifier[0].out_features == num_hidden_units
        assert isinstance(model.classifier[1], torch.nn.Hardswish)
        assert isinstance(model.classifier[2], torch.nn.Dropout)
        assert isinstance(model.classifier[3], torch.nn.Linear)
        assert model.classifier[3].out_features == output_shape

        print(f"Model and transformations are correctly set up on {device}.")
    

def test_train_step():
    # Define parameters for the build_model and data_setup functions
    pretrained_model = 'mobilenet_v3_small'
    num_hidden_units = 8
    output_shape = 3
    batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Build the model using the build_model function
    model, transform = build_model(pretrained_model, num_hidden_units, output_shape, device)
    
    # Set up data_path and transformation
    data_path = 'unit_test_files/test_data'
    
    # Set up data loaders using the data_setup function
    train_dataloader, _, classes = data_setup(data_path, batch_size, 'cpu', transform)
    
    # Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Define accuracy function using MulticlassAccuracy
    num_classes = len(classes)
    accuracy_fn = MulticlassAccuracy(num_classes=num_classes).to(device)

    # Run the train_step function
    train_loss, train_acc = train_step(model, train_dataloader, loss_fn, accuracy_fn, optimizer, device)

    # Assertions
    assert isinstance(train_loss, torch.Tensor)
    assert isinstance(train_acc, torch.Tensor)
    assert train_loss >= 0
    assert 0 <= train_acc <= 1


def test_validation_step():
    # Define parameters for the build_model and data_setup functions
    pretrained_model = 'mobilenet_v3_small'
    num_hidden_units = 8
    output_shape = 3
    batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Build the model and get transformation using the build_model function
    model, transform = build_model(pretrained_model, num_hidden_units, output_shape, device)
    
    # Set up data_path
    data_path = 'unit_test_files/test_data'
    
    # Set up data loaders using the data_setup function
    _, test_dataloader, classes = data_setup(data_path, batch_size, 'cpu', transform)
    
    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Define accuracy function using MulticlassAccuracy
    num_classes = len(classes)
    accuracy_fn = MulticlassAccuracy(num_classes=num_classes).to(device)

    # Run the test_step function
    test_loss, test_acc = validation_step(model, test_dataloader, loss_fn, accuracy_fn, device)

    # Assertions
    assert isinstance(test_loss, torch.Tensor)
    assert isinstance(test_acc, torch.Tensor)
    assert test_loss >= 0
    assert 0 <= test_acc <= 1


def test_plot_loss_curves():
    # Create a sample results dictionary
    results = {
        "train_loss": [0.9, 0.8, 0.7],
        "train_acc": [0.6, 0.7, 0.8],
        "test_loss": [1.0, 0.9, 0.8],
        "test_acc": [0.5, 0.6, 0.7]
    }
    
    # Mock the device (CPU case)
    device = 'cpu'
    
    # Define the save path
    save_path = Path('unit_test_files/test_plot.jpg')
    
    # Ensure the directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Call the function with a CPU device
    plot_loss_curves(results, device, save_path)
    
    # Check if the plot was saved
    assert save_path.exists()
    
    # Cleanup: remove the generated plot
    save_path.unlink()
    
    
# Run the test
if __name__ == "__main__":
    pytest.main()
