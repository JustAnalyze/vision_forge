
import pytest
import torch
import json
from pathlib import Path
from project import plot_loss_curves, predict_with_model


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

# Run the test
if __name__ == "__main__":
    pytest.main()
