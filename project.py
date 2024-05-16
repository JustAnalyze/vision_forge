from PIL import Image
from datetime import datetime
import json
import sys
from threading import Thread
from matplotlib import pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm.auto import tqdm
from typing import Tuple, List
from pathlib import Path
from tkinter import filedialog
from CTkMessagebox import CTkMessagebox
from typing import Union
import customtkinter
from icecream import ic
import re


def main():
    """
    Main function to instantiate and run the GUI.
    """
    gui = ModelBuilderGUI()
    gui.run()


def data_setup(data_path: str,
               batch_size: int,
               device: torch.device,
               transform: T.Compose = None) -> Tuple[DataLoader, DataLoader, List[str]]:
    
    '''
    Set up the data using torchvision.transforms.Compose, torch.utils.data.DataLoader,
    and torchvision.datasets.ImageFolder.

    Args:
        data_path: Union[str, PosixPath], Path to the data directory with train and test folders.
        transform: Compose, A composition of transformations to apply to the data.
        batch_size: int, Batch size for the data loaders.
        device: torch.device, Device to load the data onto.

    Returns:
        train_dataloader: DataLoader, Data loader for the training dataset.
        test_dataloader: DataLoader, Data loader for the testing dataset.
        classes: List[str], List of class labels.
    '''
        
    # set train data
    train_data = ImageFolder(root=data_path + '/train',
                             transform=transform)

    # set test data
    test_data = ImageFolder(root=data_path + '/test',
                            transform=transform)

    # set train data loader
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  generator=torch.Generator(device))

    # set train data loader
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size)

    return train_dataloader, test_dataloader, train_data.classes


def build_model(pretrained_model: str,
                num_hidden_units: int,
                output_shape: int,
                device: str) -> Tuple[torch.nn.Module, torchvision.transforms.Compose]:
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
        input_shape: Shape of the input in the convolution neural network.
    """
    
    # Dictionary mapping model names to their corresponding torchvision models and weights
    # Add different sizes for the pretrained models for example: efficientnet_b0, efficientnet_b1 etc.
    pretrained_models: dict[str, dict] = {
        'EfficientNet': {
            'model': torchvision.models.efficientnet_b3,
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
    model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=True),
                                           torch.nn.Linear(in_features=model.classifier[1].in_features, 
                                                           out_features=num_hidden_units,  # use the length of class_names (one output unit for each class)
                                                           bias=True),
                                           torch.nn.Linear(in_features=num_hidden_units,
                                                           out_features=output_shape,  
                                                           bias=True)).to(device)
    
    return  model, transforms


# train step function
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               optimizer: torch.optim.Optimizer,
               device):

  # Put the model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0.0, 0.0

  # Loop through data loader and data batches
  for batch, (X, y) in enumerate(dataloader):
    # Send data to target device
    X, y = X.to(device), y.to(device)
    
    # 1. Forward pass
    preds_logits = model(X)

    # 2. Calculate and accumulate loss
    loss = loss_fn(preds_logits, y)
    train_loss += loss

    # 3. label predictions
    preds_labels = torch.argmax(preds_logits, dim=1)

    # 3.1 Calculate and accumualte accuracy metric across all batches
    acc = accuracy_fn(preds_labels, y)
    train_acc += acc

    # 4. Optimizer zero grad
    optimizer.zero_grad()

    # 5. Loss backward
    loss.backward()

    # 6. Optimizer step
    optimizer.step()

  # Adjust metrics to get average loss and average accuracy per batch
  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  return train_loss, train_acc


# test step function
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device):

  # Put model in eval mode
  model.eval()

  # Setup the test loss and test accuracy values
  test_loss, test_acc = 0.0, 0.0

  # Loop through DataLoader batches
  for batch, (X, y) in enumerate(dataloader):
    # Send data to target device
    X, y = X.to(device), y.to(device)

    # 1. Forward pass
    # Turn on inference context manager
    with torch.inference_mode():
      preds_logits = model(X)

    # 2. Calculuate and accumulate loss
    loss = loss_fn(preds_logits, y)
    test_loss += loss

    # 3. label predictions
    preds_labels = torch.argmax(preds_logits, dim=1)

    # Calculate and accumulate accuracy
    acc = accuracy_fn(preds_labels, y)
    test_acc += acc

  # Adjust metrics to get average loss and accuracy per batch
  test_loss /= len(dataloader)
  test_acc /= len(dataloader)

  return test_loss, test_acc


# Train model for specified epoch using train step and evaluate using test step
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          accuracy_fn,
          device,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),  # default loss function for multiclass classification
          epochs: int = 5,
          progress_bar_widget: customtkinter.CTkProgressBar = None):

    """
    Trains the given model using the provided data loaders.

    Args:
        model (torch.nn.Module): The model to train.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for testing/validation data.
        optimizer (torch.optim.Optimizer): Optimizer to use during training.
        accuracy_fn (callable): Function to compute accuracy.
        device: Device to use for training (e.g., 'cuda' for GPU, 'cpu' for CPU).
        loss_fn (torch.nn.Module, optional): Loss function. Default is nn.CrossEntropyLoss().
        epochs (int, optional): Number of epochs for training. Default is 5.

    Returns:
        dict: A dictionary containing training and testing metrics.
            Keys:
                - "train_loss": List of training losses for each epoch.
                - "train_acc": List of training accuracies for each epoch.
                - "test_loss": List of testing losses for each epoch.
                - "test_acc": List of testing accuracies for each epoch.
    """

    # Inform the user that the Training is starting
    print(f"Starting training for {epochs} epochs...")
    
    # Create results dictionary
    results = {"train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []}

    # Loop through the training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        # Train step
        train_loss, train_acc = train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        accuracy_fn=accuracy_fn,
                                        optimizer=optimizer,
                                        device=device)
        # Test step
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        accuracy_fn=accuracy_fn,
                                        loss_fn=loss_fn,
                                        device=device)

        # Print out training results
        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

        # TODO: Add a label in the progress bar representing the percentage (50% or 5/10)
        if progress_bar_widget:
            progress_bar_percentage: float = (epoch + 1) / epochs
            progress_bar_widget.set(progress_bar_percentage)

        # Update the results dictionary
        results["train_loss"].append(train_loss.detach() if device == 'cpu' else torch.Tensor.cpu(train_loss.detach()))
        results["train_acc"].append(train_acc if device == 'cpu' else torch.Tensor.cpu(train_acc))
        results["test_loss"].append(test_loss if device == 'cpu' else torch.Tensor.cpu(test_loss))
        results["test_acc"].append(test_acc if device == 'cpu' else torch.Tensor.cpu(test_acc))

    # Inform user that the training is done.
    print("Training is done.")
    
    # Return the results dictionary
    return results


def plot_loss_curves(results: dict[str, list[float]],
                     device,
                     save_path: str = None):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    # if Tensors are in cuda transfer them to cpu
    if device == 'cuda':
      def to_cpu(x):
        return torch.Tensor.cpu(x)

      # Get the loss values of the results dictionary (training and test)
      loss = list(map(to_cpu, results['train_loss']))
      test_loss = list(map(to_cpu, results['test_loss']))

      # Get the accuracy values of the results dictionary (training and test)
      accuracy = list(map(to_cpu, results['train_acc']))
      test_accuracy = list(map(to_cpu, results['test_acc']))

    else:
      loss = results['train_loss']
      test_loss = results['test_loss']
      accuracy = results['train_acc']
      test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    # set save path
    if save_path is not None:
        model_save_path = (save_path / 'loss_accuracy_plot.jpg')
    else:
        model_save_path = 'loss_accuracy_plot.jpg'
        
    # Save the plot as a JPG file
    plt.savefig(model_save_path, format='jpg')
    plt.close()


def save_outputs(model, train_results, settings_dict, device):
    """
    Save trained model, visualizations, and settings.
    """
    # Create a directory with current timestamp to store outputs
    output_dir = Path(f"training_output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trained model weights
    model_path = output_dir / "model.pth"
    torch.save(obj=model, f=model_path)
    
    # Save loss curves plot
    plot_loss_curves(train_results, device=device, save_path=output_dir)
    
    # Save model and data settings as JSON file
    settings_path = output_dir / "settings.json"
    with open(settings_path, 'w') as f:
        json.dump(settings_dict, f, indent=4)

    print(f"\nAll outputs saved in: {output_dir}")


class ModelBuilderGUI:
    """
    A class representing the GUI for training and predicting using an image classifier model.
    """
    def __init__(self):
        """
        Initialize the GUI and the Dictionary of settings.
        """
        # Set up the main window and the theme
        customtkinter.set_default_color_theme('dark-blue')
        customtkinter.set_appearance_mode("dark")
        
        self.root = customtkinter.CTk()
        self.root.title("Vision Forge")
        self.root.geometry("700x370")
        
        # Set up the dictionary of model and data settings
        self._settings_dict: dict[str, dict[str, Union[str, int, float]]] = {'model_settings': {},
                                                                             'data_settings': {}}
        
        # Create sidebar with options for training and predicting
        self._create_sidebar()
        
        # Create predict frame
        self._create_predict_frame()
        
        # Create tabs widgets for customizing the model and the data
        self._create_train_frame_tabs()
        
        # Create model tab widgets
        self._create_model_tab_widgets()
        
        # Create data tab widgets
        self._create_data_tab_widgets()
        
        # Create Train Button for starting training
        self._create_predict_button()
        
        # Create Train Button for starting training
        self._create_train_button()

    def _create_sidebar(self):
        """
        Create sidebar with options for training and predicting.
        """
        self.sidebar_frame = customtkinter.CTkFrame(self.root, width=100)
        self.sidebar_frame.pack(side='left', fill='y')

        # Define functions to handle frame switching
        def switch_to_train_frame():
            self.predict_button.pack_forget()
            self.predict_frame.pack_forget()
            self.tabview.pack(fill=customtkinter.BOTH, expand=True)
            self.train_button.pack(pady=10)

        def switch_to_predict_frame():
            self.train_button.pack_forget()
            self.tabview.pack_forget()
            self.predict_frame.pack(fill='both', expand=True)
            self.predict_button.pack(pady=10)

        predict_button = customtkinter.CTkButton(self.sidebar_frame, text="Predict Using Model", command=switch_to_predict_frame)
        predict_button.pack(padx=5, pady=10)

        train_button = customtkinter.CTkButton(self.sidebar_frame, text="Train Model", command=switch_to_train_frame)
        train_button.pack(padx=5,pady=5)

    def _create_predict_frame(self):
        """
        Create predict frame with entry boxes for model path and input data path.
        """
        self._predict_inputs: dict[str, Union[str, list]] = {'model_path': None,
                                                             'input_data_path': None}
        
        self.predict_frame = customtkinter.CTkFrame(self.root)
        
        predict_label = customtkinter.CTkLabel(self.predict_frame,
                                               text="Predict Using Trained Model")

        # Entry box and browse button for trained model path
        model_path_var = customtkinter.StringVar()
        model_path_entry = customtkinter.CTkEntry(self.predict_frame,
                                                  width=383, 
                                                  justify='center',
                                                  textvariable=model_path_var)

        def browse_model_path():
            model_path = filedialog.askopenfilename()
            
            if model_path:
                # Check if the selected file is a valid model
                if model_path.endswith(".pth") or model_path.endswith(".pt"):
                    model_path_var.set(model_path)
                    self._predict_inputs['model_path'] = model_path_var.get()
                    ic(self._predict_inputs)
                else:
                    # inform the user that the file is not a valid model
                    CTkMessagebox(title="Error", message="Selected file is not a valid Model", icon="cancel")

        browse_model_button = customtkinter.CTkButton(self.predict_frame,
                                                      width=105,
                                                      height=28,
                                                      text="Browse Model",
                                                      command=browse_model_path)

        # Entry box and browse button for input data path
        input_data_path_var = customtkinter.StringVar()
        input_data_path_entry = customtkinter.CTkEntry(self.predict_frame,
                                                       width=383, 
                                                       justify='center',
                                                       textvariable=input_data_path_var)

        def input_data_path():
            input_data_path = filedialog.askopenfilename()
            if input_data_path:
                # Check if the selected file is a valid image
                try:
                    with Image.open(input_data_path) as img:
                        # If opening the image succeeds, set the input_data_path_var
                        input_data_path_var.set(input_data_path)
                        self._predict_inputs['input_data_path'] = input_data_path_var.get()
                        ic(self._predict_inputs)
                except:
                    # If opening the image fails, inform the user that the file is not a valid image
                    CTkMessagebox(title="Error", message="Selected file is not a valid image.", icon="cancel")

        browse_input_data_button = customtkinter.CTkButton(self.predict_frame,
                                                           width=105,
                                                           height=28,
                                                           text="Browse Image",
                                                           command=input_data_path)

        # Widget Grid management
        predict_label.grid(row=0, column=0, columnspan=3)
        model_path_entry.grid(row=1, column=0, columnspan=2, padx=15, pady=20)
        browse_model_button.grid(row=1, column=2, padx=5, pady=20)
        input_data_path_entry.grid(row=2, column=0, columnspan=2, padx=15, pady=20)
        browse_input_data_button.grid(row=2, column=2, padx=5, pady=20)

    def _create_train_frame_tabs(self) -> None:
        """
        Create tabs for customizing the model and the data.
        """
        self.tabview = customtkinter.CTkTabview(self.root)
        self.tabview.pack(fill=customtkinter.BOTH, expand=True)

        self.tabview.add('Model')
        self.tabview.add('Data')
        self.tabview.set('Model') 

    def _create_model_tab_widgets(self) -> None:
        """
        Create widgets for the Model tab.
        """

        model_tab = self.tabview.tab('Model')
        
        # Add a ComboBox for choosing the type of task
        task_type_label = customtkinter.CTkLabel(master=model_tab, text="Task")
        type_list = ['Multiclass Classification'] # add multiclass classification task
        task_type_var = customtkinter.StringVar(value='Multiclass Classification')
        task_type = customtkinter.CTkComboBox(master=model_tab,
                                              values=type_list, width=200,
                                              justify='center',
                                              variable=task_type_var)
        
        # Add ComboBox for choosing the optimizer
        optimizer_label = customtkinter.CTkLabel(master=model_tab, text="Optimizer")
        optimizer_list =['SGD', 'Adam', 'AdamW', 'RMSProp']
        optimizer_var = customtkinter.StringVar(value='Adam')
        optimizer = customtkinter.CTkComboBox(master=model_tab,
                                              values=optimizer_list,
                                              width=200,
                                              justify='center',
                                              variable=optimizer_var)
        
        # Add entry box for setting the number of units/neurons in the hidden layer
        num_hidden_units_label = customtkinter.CTkLabel(master=model_tab, text="Hidden Units/Neurons")
        num_hidden_units_var = customtkinter.IntVar(value=64)
        num_hidden_units = customtkinter.CTkEntry(model_tab, 
                                                  placeholder_text='64',
                                                  width=200, 
                                                  justify='center',
                                                  textvariable=num_hidden_units_var)
        
        # Add Combobox for choosing a pretrained model for transfer learning
        pretrained_model_label = customtkinter.CTkLabel(master=model_tab, text="Pretrained Model")
        pretrained_model_list = ["EfficientNet"] # TODO: add more pre-trained model choices "EfficientNetV2",,"MobileNet V2","MobileNet V3","ResNet","Inception V3"
        pretrained_model_var = customtkinter.StringVar()
        pretrained_model = customtkinter.CTkComboBox(master=model_tab, 
                                                     values=pretrained_model_list, 
                                                     width=200,
                                                     justify='center',
                                                     variable=pretrained_model_var)
        
        # Add entry box for setting the value of learning rate
        learning_rate_label = customtkinter.CTkLabel(master=model_tab, text='Learning Rate')
        learning_rate_var = customtkinter.Variable(value=0.001)
        learning_rate = customtkinter.CTkEntry(master=model_tab, 
                                               placeholder_text='0.001',
                                               width=200, 
                                               justify='center',
                                               textvariable=learning_rate_var)
        
        # Add entry box for setting the number of epochs
        epochs_label = customtkinter.CTkLabel(master=model_tab, text='Epochs')
        epochs_var = customtkinter.IntVar(value=16)
        epochs = customtkinter.CTkEntry(master=model_tab, 
                                          placeholder_text='16',
                                          width=200, 
                                          justify='center',
                                          textvariable=epochs_var)
        
        # Add save button for saving the inputted values and handle invalid inputs
        def save_button_event() -> None:
            # list of user inputs
            user_inputs = {'task_type': task_type_var.get(),
                           'pretrained_model': pretrained_model_var.get(),
                           'optimizer': optimizer_var.get(),
                           'epochs': epochs_var.get(),
                           'num_hidden_units': num_hidden_units_var.get(),
                           'learning_rate': learning_rate_var.get()}
            
            if '' not in user_inputs.values():
                self._settings_dict['model_settings'] = {'task_type': task_type_var.get(),
                                                         'pretrained_model': pretrained_model_var.get(),
                                                         'optimizer': optimizer_var.get(),
                                                         'epochs': epochs_var.get(),
                                                         'num_hidden_units': num_hidden_units_var.get(),
                                                         'learning_rate': float(learning_rate_var.get())}
                
                # show a label when the inputs are valid
                show_info(message='Model settings successfully saved',
                          text_color='green')
                
                ic(self._settings_dict)
                
            else:
                # show a label about the blank input error
                show_info(message='Please Fill all Fields.',
                          text_color='red')
                
                print(f"Please Fill all Fields.")

        def show_info(message: str, text_color: str) -> None:
            save_info_label.configure(text=message,
                                      text_color=text_color,
                                      justify='center')
                
            save_info_label.place(x=0, y=200)
                
        save_info_label = customtkinter.CTkLabel(master=model_tab, 
                                                 justify='center',
                                                 width=550)
        save_button = customtkinter.CTkButton(model_tab,
                                              text='Save',
                                              width=90,
                                              height=28,
                                              command=save_button_event)
        
        
        # Grid layout management
        task_type_label.grid(row=0, column=0, padx=30)
        task_type.grid(row=1, column=0, pady=5, padx=30)
        
        pretrained_model_label.grid(row=2, column=0, padx=30)
        pretrained_model.grid(row=3, column=0, pady=5, padx=30)
        
        optimizer_label.grid(row=4, column=0, padx=30)
        optimizer.grid(row=5, column=0, padx=30)
        
        epochs_label.grid(row=0, column=1, padx=30)
        epochs.grid(row=1, column=1, padx=30)
        
        num_hidden_units_label.grid(row=2, column=1, padx=30)
        num_hidden_units.grid(row=3, column=1, padx=30)
        
        learning_rate_label.grid(row=4, column=1, padx=30)
        learning_rate.grid(row=5, column=1, padx=30)

        save_button.place(x=225, y=230)
    
    def _create_data_tab_widgets(self) -> None:
        """
        Create widgets for the Data tab.
        """
        # set data tab
        data_tab = self.tabview.tab('Data')
        
        # Function for browsing data directory
        def browse_data() -> None:
            '''
            This function opens a file dialog and gets the selected directory.
            If a directory is selected, sets the value of the 'data_path_var' variable to the selected directory.
            '''
            dir = filedialog.askdirectory() # Open file dialog and get selected directory
            if dir:
                print(f'Data Directory: {dir}')
                
                # get the file and folder names from the selected directory
                files_and_folders = [path.name for path in Path(dir).glob('*')]
                train_folder_path = [path for path in Path(dir).glob('*') if path.name == 'train']
                test_folder_path = [path for path in Path(dir).glob('*') if path.name == 'test']
                
                # check if the directory has a test and train folder inside it
                if 'train' in files_and_folders and 'test' in files_and_folders:
                    print('\n[INFO] train and test folders found!')
                    print(f'Files and Folders: {files_and_folders}')
                    
                    # Inform the user that the data is found
                    show_info(message='Data Found!', text_color='green')

                    # get total number of training data.
                    classes = [path.name for path in train_folder_path[0].glob('*')]
                    print(f'Classes: {classes}\nNumber of Classes: {len(classes)}')
                    
                    # get total number of training data.
                    num_of_train_data = len([path.name for path in train_folder_path[0].glob('**/*')]) - len(classes)
                    print(f'\nTotal Train Data: {num_of_train_data}')

                    # get total number of test data.
                    num_of_test_data = len([path.name for path in test_folder_path[0].glob('**/*')]) - len(classes)
                    print(f'Total Test Data: {num_of_test_data}\n')

                    # Get percentage of training and testing data
                    percent_train_data = (num_of_train_data / (num_of_test_data + num_of_train_data)) * 100
                    percent_test_data = 100 - percent_train_data
                    print(f'Percentage of Train Data: {percent_train_data}%')
                    print(f'Percentage of Test Data: {percent_test_data}%')
                    
                    # set the variables to the values we got from the Dataset
                    data_path_var.set(dir)
                    num_classes_var.set(len(classes))
                    data_split_var.set(f'{round(percent_train_data, 2)}/{round(percent_test_data, 2)}')
                    
                # else if there is train and no test folder inform user about the missing folder
                elif 'train' in files_and_folders and 'test' not in files_and_folders:
                    
                    print('[INFO] test folder not found!')
                    # Inform the user that the test folder is not found
                    show_info(message='Test folder not found!', text_color='red')
                    
                # else if there is no train and there is test folder inform user about the missing folder
                elif 'train' not in files_and_folders and 'test' in files_and_folders:
                    
                    print('[INFO] train folder not found!')
                    # Inform the user that the train folder is not found
                    show_info(message='Train folder not found!', text_color='red')
                    
                # else both folder missing
                else:
                    
                    print('[INFO] train and test folder not found!')
                    
                    # Inform the user that the train and test folders are not found
                    show_info(message='Train and test folder not found!', text_color='red')
                    
                    
        # Entry box for setting the direvtory of the data
        data_path_var = customtkinter.StringVar(value='C:/path/to/data/directory')
        data_path = customtkinter.CTkEntry(master=data_tab, 
                                          width=383, 
                                          justify='center',
                                          textvariable=data_path_var)
        
        # Button to browse directory
        browse_data_button = customtkinter.CTkButton(data_tab,
                                                     width=105,
                                                     height=28,
                                                     text="Browse",
                                                     command=browse_data)
        
        # Entry box for setting value of Batch Size
        batch_size_label = customtkinter.CTkLabel(data_tab, text='Batch Size')
        batch_size_var = customtkinter.IntVar(value=32)
        batch_size = customtkinter.CTkEntry(data_tab,
                                            justify='center',
                                            textvariable=batch_size_var)
        
        # Entry box for number of classes widget
        num_classes_label = customtkinter.CTkLabel(data_tab, text='Number of Classes')
        num_classes_var = customtkinter.Variable()
        num_classes = customtkinter.CTkEntry(data_tab,
                                             textvariable=num_classes_var,
                                             justify='center',
                                             state='disabled')
        
        # Entry box for the value of Train data percentage widget 
        data_split_label = customtkinter.CTkLabel(data_tab, text='Train/Test Split')
        data_split_var = customtkinter.Variable()
        data_split = customtkinter.CTkEntry(data_tab,
                                            textvariable=data_split_var,
                                            justify='center',
                                            state='disabled')
        
        # Add save button for saving the inputted values and handle invalid inputs
        def save_button_event() -> None:
            # list of user inputs
            user_inputs = {'data_path': data_path_var.get(),
                           'batch_size': batch_size_var.get(),
                           'num_classes': num_classes_var.get(),
                           'data_split': data_split_var.get(),}

            if 'C:/path/to/data/directory' == data_path_var.get():
                # show a label when the inputs are valid
                show_info('Invalid Data Directory', text_color='red')
            
            elif '' not in user_inputs.values():
                self._settings_dict['data_settings'] = {'data_path': data_path_var.get(),
                                                        'batch_size': batch_size_var.get(),
                                                        'num_classes': num_classes_var.get(),
                                                        'data_split': data_split_var.get(),}
                # show a label when the inputs are valid
                show_info('Data settings successfully saved', text_color='green')
                ic(self._settings_dict)
                
            else:
                # show a label about the blank input error
                show_info("Please Fill all Fields.", text_color='red')
                print(f"Please Fill all Fields.")

        def show_info(message: str, text_color: str) -> None:
            """
            This function configures the 'save_info_label' with the provided message, text color, and justification,
            then places it at the specified coordinates.
            """
            info_label.configure(text=message,
                                      text_color=text_color,
                                      justify='center')
                
            info_label.place(x=0, y=200)
                
        info_label = customtkinter.CTkLabel(master=data_tab, text='', justify='center', width=550)
        save_button = customtkinter.CTkButton(data_tab,
                                              text='Save',
                                              width=90,
                                              height=28,
                                              command=save_button_event)
        
        # Layout Management
        data_path.grid(row=0, column=0, columnspan=2, padx=15, pady=20)
        browse_data_button.grid(row=0, column=2, padx=5, pady=20)
        
        num_classes_label.place(x=50, y=60)
        num_classes.place(x=32, y=90)

        data_split_label.place(x=224, y=60)
        data_split.place(x=202, y=90)
        
        batch_size_label.place(x=410, y=60)
        batch_size.place(x=372, y=90)
        
        save_button.place(x=225, y=230)
    
    def _validate_settings_dict(self) -> bool:
        """
        Validate the settings dictionary. makes sure the user inputs does not cause errors in the training process.
        """
        # These are the valid values in the settings dictionary for the mean time.
        valid_values_dict = {'task_type':['Multiclass Classification'],
                             'pretrained_model':['EfficientNet'],
                             'optimizer':['SGD', 'Adam', 'AdamW', 'RMSProp'],}
        
        model_setttings = self._settings_dict['model_settings']
        data_settings = self._settings_dict['data_settings']
        
        # If the user did not saved the model settings
        if not model_setttings:
            CTkMessagebox(title="Error", message="Please Save the model settings.", icon="cancel")
            return False
        
        # If the user did not saved the Data settings
        if not data_settings:
            CTkMessagebox(title="Error", message="Please Save the Data settings.", icon="cancel")
            return False
        
        # Task type is not valid
        if model_setttings['task_type'] not in valid_values_dict['task_type']:
            CTkMessagebox(title="Error", message="Please select a Valid task type", icon="cancel")
            return False # return false prevent errors in the training process
            
        # pretrained model is not valid
        if model_setttings['pretrained_model'] not in valid_values_dict['pretrained_model']:
            CTkMessagebox(title="Error", message="Please select a Valid Pre-trained Model", icon="cancel")
            return False
        
        # Optimizer is not valid
        if model_setttings['optimizer'] not in valid_values_dict['optimizer']:
            CTkMessagebox(title="Error", message="Please select a Valid Optimizer", icon="cancel")
            return False
        
        # If number of classes and train test split percentage are not detected in the data directory
        if not data_settings['num_classes'] and not data_settings['data_split']:
            CTkMessagebox(title="Error", message="Please select a Valid Data Directory", icon="cancel")
            return False
        
        # Check if the learning rate is within a reasonable range
        if model_setttings['learning_rate'] >= 1 or model_setttings['learning_rate'] < 0.000001:
            CTkMessagebox(title="Error", message="Make sure that the learning rate is between 1 and 0.000001", icon="cancel")
            return False
        
        # Check if the number of hidden units is valid
        if model_setttings['num_hidden_units'] < 1 and isinstance(model_setttings['num_hidden_units'], int):
            CTkMessagebox(title="Error", message="Make sure that the number of hidden units is a positive whole number", icon="cancel")
            return False
        
        # If all settings are valid return True
        return True
    
 
    def _create_predict_button(self):
        """
        Create a button for starting the training.
        """
        # TODO: Create function for using a existing model for Prediction.
        def predict_button_event():
            print("Predict button Pressed")
        
        self.predict_button = customtkinter.CTkButton(master=self.root, text="Predict", command=predict_button_event)
        
    
    # Create method for creating Train button widget.
    def _create_train_button(self):
        """
        Create a button for starting the training.
        """
        
        def train_button_event():
            # if settings are valid continue to training
            if self._validate_settings_dict():
                # Start the training process
                self._train_and_save_model() 
        
        self.train_button = customtkinter.CTkButton(master=self.root, text="Train", command=train_button_event)
        self.train_button.pack(pady=10)
    
    def _train_and_save_model(self):
        """
        Load the data and start the training process.
        """
        
        # check if a CUDA-capable GPU is available and sets the default device to 'cuda' if it is. If not, set the default device to 'cpu'. 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_default_device(device)
        
        # setup model_settings dict and data_settings dict
        model_settings = self._settings_dict['model_settings']
        data_settings = self._settings_dict['data_settings']
        
        # Create a dictionary of the available optimizers
        optimizers: dict[str, torch.optim.Optimizer] = {'SGD': torch.optim.SGD,
                                                        'Adam': torch.optim.Adam,
                                                        'AdamW': torch.optim.AdamW,
                                                        'RMSProp': torch.optim.RMSprop}
        
        # Output training performance metrics in a pop up window    
        # Create a pop up window that can take up the text and has a progress bar
        def show_training_progress():
            """Display a popup window showing training progress."""
            
            popup_window = customtkinter.CTkToplevel(self.root)
            popup_window.title("Training Performance")

            # widget for storing the performance of the training
            output_text = customtkinter.CTkTextbox(popup_window,
                                                    wrap='word',
                                                    height=370, width=550)
            
            output_text.pack(expand=True, fill='both')
            
            # Create a progress bar to visualize the progress of training3
            training_progress_bar = customtkinter.CTkProgressBar(popup_window,
                                                                    width=520, 
                                                                    height=20, 
                                                                    determinate_speed=model_settings['epochs'])
            training_progress_bar.set(0)
            training_progress_bar.pack(side="bottom", anchor="s", padx=20, pady=10)
            
            # Create a class for redirecting the output to the Text widget.
            class StdoutRedirector(object):
                def __init__(self, text_widget):
                    self.text_space = text_widget

                def write(self, message):
                    self.text_space.insert(customtkinter.END, message)
                    
                def flush(self):
                    pass
            
            # Redirect stdout to the Text widget
            sys.stdout = StdoutRedirector(output_text)

            # return the trainig_progress_bar to be configured inside the train function
            return training_progress_bar
        
        # Show training progress
        training_progress_bar = show_training_progress()
        
        # Function to perform model building, load data, and train model (SEPARATE THREAD) 
        def train_save_model():
            # Build model
            model, transforms = build_model(pretrained_model=model_settings['pretrained_model'],
                                                        num_hidden_units=model_settings['num_hidden_units'],
                                                        output_shape=data_settings['num_classes'],
                                                        device=device)
            
            # Load and preprocess the training and testing datasets.
            train_dataloader, test_dataloader, classes = data_setup(data_path=data_settings['data_path'],
                                                                    batch_size=data_settings['batch_size'],
                                                                    device='cpu',
                                                                    transform=transforms)  # use transforms used in training the pretrained model
            
            # Setup a variable for the selected optimizer
            optimizer: torch.optim.Optimizer = optimizers[model_settings['optimizer']](params=model.parameters(),
                                                                                       lr=model_settings['learning_rate'])
            
            # Setup a variable for the accuracy function
            accuracy_fn = MulticlassAccuracy(num_classes=data_settings['num_classes']).to(device)
            
            # Debugging
            ic(classes)
            ic(transforms)
            
            print(f'Device: {device}')
            
            # Your existing training code here
            train_results = train(model=model,
                                  train_dataloader=train_dataloader,
                                  test_dataloader=test_dataloader,
                                  optimizer=optimizer,
                                  accuracy_fn=accuracy_fn,
                                  device=device,
                                  epochs=model_settings['epochs'],
                                  progress_bar_widget=training_progress_bar)
            
            # save the trained model, visualizations, and the model and data settings as a json file.
            save_outputs(model=model,
                         train_results=train_results,
                         settings_dict=self._settings_dict,
                         device=device)
            
        # Thread for training
        train_thread = Thread(target=train_save_model)
        train_thread.start()
        
        
    def run(self) -> None:
        """
        Run the GUI.
        """
        self.root.mainloop()


if __name__ == "__main__":
    main()