import os
from tempfile import TemporaryDirectory
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
                     train_step, 
                     validation_step,
                     train,
                     save_outputs)

def setup(pretrained_model, 
          num_hidden_units,
          output_shape, 
          batch_size, 
          device):
    """
    Setup the model, data loaders, loss function, optimizer, and accuracy function for testing.
    """
    # Build the model
    model, transform = build_model(pretrained_model, num_hidden_units, output_shape, device)
    
    # Set up data_path
    data_path = 'unit_test_files/test_data'
    
    # Set up data loaders using the data_setup function
    train_dataloader, test_dataloader, classes = data_setup(data_path, batch_size, 'cpu', transform)
    
    # Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Define accuracy function using MulticlassAccuracy
    num_classes = len(classes)
    accuracy_fn = MulticlassAccuracy(num_classes=num_classes).to(device)
    
    return model, train_dataloader, test_dataloader, classes, loss_fn, optimizer, accuracy_fn

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
    
    # test conducting train_step in both cpu and cuda if cuda is available
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        # Setup the model and data loaders
        model, _, train_dataloader, _, loss_fn, optimizer, accuracy_fn = setup(pretrained_model,
                                                                                num_hidden_units,
                                                                                output_shape,
                                                                                batch_size, 
                                                                                device)   

        # Run the train_step function
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, accuracy_fn, optimizer, device)

        # Assertions
        assert isinstance(train_loss, torch.Tensor)
        assert isinstance(train_acc, torch.Tensor)
        assert train_loss >= 0
        assert 0 <= train_acc <= 1


def test_validation_step():
    """
    Test the validation_step function to ensure it returns the expected types and values.
    """
    # Define parameters for the build_model and data_setup functions
    pretrained_model = 'mobilenet_v3_small'
    num_hidden_units = 8
    output_shape = 3
    batch_size = 1
    
    # test conducting train_step in both cpu and cuda if cuda is available
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        # Setup the model and data loaders
        model, _, test_dataloader, _, loss_fn, _, accuracy_fn = setup(pretrained_model,
                                                                    num_hidden_units,
                                                                    output_shape,
                                                                    batch_size, 
                                                                    device)   
        # Run the test_step function
        test_loss, test_acc = validation_step(model, test_dataloader, loss_fn, accuracy_fn, device)

        # Assertions
        assert isinstance(test_loss, torch.Tensor)
        assert isinstance(test_acc, torch.Tensor)
        assert test_loss >= 0
        assert 0 <= test_acc <= 1


def test_train():
    """
    Test the train function to ensure it returns the expected results and types.
    """
    # Define parameters for the build_model and data_setup functions
    pretrained_model = 'mobilenet_v3_small'
    num_hidden_units = 8
    output_shape = 3
    batch_size = 1
    # test conducting train_step in both cpu and cuda if cuda is available
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        model, train_dataloader, test_dataloader, _, loss_fn, optimizer, accuracy_fn = setup(pretrained_model,
                                                                                            num_hidden_units,
                                                                                            output_shape,
                                                                                            batch_size, 
                                                                                            device)

        results, final_model, best_acc_model = train(model,
                                                    train_dataloader,
                                                    test_dataloader,
                                                    optimizer,
                                                    accuracy_fn,
                                                    device,
                                                    loss_fn,
                                                    epochs=2)

        # Assertions to check the keys in the results dictionary
        assert "train_loss" in results
        assert "train_acc" in results
        assert "test_loss" in results
        assert "test_acc" in results

        # Assertions to check the length of the lists in the results dictionary
        assert len(results["train_loss"]) == 2
        assert len(results["train_acc"]) == 2
        assert len(results["test_loss"]) == 2
        assert len(results["test_acc"]) == 2
        
        # Assertions to check the types of the returned the results dictionary values
        assert isinstance(results["train_loss"][0], torch.Tensor)
        assert isinstance(results["train_acc"][0], torch.Tensor)
        assert isinstance(results["test_loss"][0], torch.Tensor)
        assert isinstance(results["test_acc"][0], torch.Tensor)

        # Assertions to check the types of the returned models
        assert isinstance(final_model, torch.nn.Module)
        assert isinstance(best_acc_model, torch.nn.Module)

        # Assertions to check the values in the results dictionary
        assert results["train_loss"][0] >= 0
        assert 0 <= results["train_acc"][0] <= 1 
        assert results["test_loss"][0] >= 0
        assert 0 <= results["test_acc"][0] <= 1 


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
    

# Pytest test case
def test_save_outputs():
    # Set up a temporary directory to avoid cluttering the filesystem
    with TemporaryDirectory(dir = r'D:\CS50P\vision_forge\unit_test_files') as temp_dir:
        # Change to the temporary directory
        os.chdir(temp_dir)
        
        # Define dummy models
        model_1 = torch.nn.Linear(10, 2)
        model_2 = torch.nn.Linear(10, 2)
        models = [model_1, model_2]

        # Define dummy train results
        train_results = {
            "train_loss": [0.1, 0.05],
            "train_acc": [0.9, 0.95],
            "test_loss": [0.15, 0.1],
            "test_acc": [0.85, 0.9]
        }

        # Define dummy settings
        settings_dict = {"model_settings": {
                            "task_type": "Multiclass Classification",
                            "pretrained_model": "mobilenet_v3_small",
                            "optimizer": "Adam",
                            "epochs": 60,
                            "num_hidden_units": 128,
                            "learning_rate": 0.0009
                        },
                        "data_settings": {
                            "data_path": "D:/CS50P/vision_forge/data/pizza_steak_sushi_20_percent",
                            "batch_size": 4,
                            "num_classes": 3,
                            "data_split": "75.0/25.0",
                            "classes": ["pizza", "steak" ,"sushi"]
                        }}
        # Define device
        device = 'cpu'

        # Call the save_outputs function
        save_outputs(models, train_results, settings_dict, device)

        # Check if the 'runs' directory exists
        runs_folder_dir = Path('runs')
        assert runs_folder_dir.is_dir()

        # Get the output directory name
        output_dir = next(runs_folder_dir.iterdir())
        assert output_dir.is_dir()

        # Check if the expected files are created
        assert (output_dir / "final_model.pth").is_file()
        assert (output_dir / "best_model_acc.pth").is_file()
        assert (output_dir / "loss_accuracy_plot.jpg").is_file()
        assert (output_dir / "settings.json").is_file()

        # Validate the contents of the settings.json file
        with open(output_dir / "settings.json", 'r') as f:
            saved_settings = json.load(f)
        assert saved_settings == settings_dict

        # Cleanup: Change back to the original directory
        os.chdir("..")

    
# Run the test
if __name__ == "__main__":
    pytest.main()
