
import pytest
import torch
from torchvision.transforms import v2 as T
import json
from pathlib import Path
from project import plot_loss_curves, predict_with_model, data_setup


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

def test_data_setup():
    # Define paths and parameters
    data_path = Path('unit_test_files/test_data')
    batch_size = 2
    device = torch.device('cpu')
    transform = T.Compose([torch.T.Resize((128, 128)), T.ToTensor()])

    # Run the function
    train_dataloader, test_dataloader, classes = data_setup(data_path, batch_size, device, transform)

    # Assertions
    assert len(classes) > 3
    assert len(train_dataloader) > 0
    assert len(test_dataloader) > 0
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


# Run the test
if __name__ == "__main__":
    pytest.main()
