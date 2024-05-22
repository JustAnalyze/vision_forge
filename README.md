# Vision Forge

#### Video Demo:  

#### Description:
Vision Forge is an intuitive application designed for creating, training, and utilizing image classification models through a user-friendly graphical user interface (GUI). This app eliminates the need for any coding, making it accessible to users of all skill levels.

### Features:
- **Custom Model Creation:** Customize your image classification models with options to select task type, pretrained models, optimizers, epochs, hidden units/neurons, and learning rate.
- **Data Loading:** Easily load your dataset, and the app will automatically infer the number of classes and the train/test split based on the selected data folder. You can manually adjust the batch size if needed. The app manages data loading in minibatches for efficient training.
- **Model Training:** Train your customized model with the click of a button.
- **Prediction:** Use the trained model to make predictions on new images, all through the simple GUI.

### GUI Overview:
Here is a preview of the GUI for your reference:

![Vision Forge GUI](file_for_readme\train_new_model_tab.png)

- **Model Tab:** Configure your model by selecting the task, pretrained model, optimizer, and other hyperparameters.
- **Data Tab:** Load your data, and the app will auto-populate the number of classes and train/test split based on the data. You can manually adjust the batch size if needed.
- **Prediction Tab:** Use a trained model to predict classes for new images.

### Getting Started:
1. **Clone the repository:**
    ```bash
    git clone <https://github.com/JustAnalyze/vision_forge>
    cd vision-forge
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application:**
    ```bash
    python project.py
    ```

### Requirements:
- Python 3.9+
- Required Python packages (listed in `requirements.txt`)

### Acknowledgments:
- [Pytorch](https://pytorch.org/)
- [tkinter](https://docs.python.org/3/library/tkinter.html)
- [customtkinter](https://github.com/TomSchimansky/CustomTkinter)
- [CTkMessagebox](https://github.com/Akascape/CTkMessagebox)

For any questions or support, please open an issue in the repository.