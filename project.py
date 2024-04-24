from pathlib import Path
from tkinter import filedialog
from turtle import width
from typing import Union
import customtkinter

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
        self.root.geometry("550x370")
        
        # Set up the dictionary of model and data settings
        self._settings_dict: dict[str, dict[str, Union[str, int, float]]] = {'model_settings': {},
                                                                             'data_settings': {}}
        # TODO: Add side bar where train and predict button is located so users can choose to train a new model or use a existing model to do some inferences/predictions
        # Create tabs widgets for customizing the model and the data
        self._create_tabs()
        
        # Create model tab widgets
        self._create_model_tab_widgets()
        
        # Create data tab widgets
        self._create_data_tab_widgets()
        
        # Create Train Button for starting training
        self._create_train_button()

    def _create_tabs(self) -> None:
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
        type_list = ['Binary Classification', 'Multiclass Classification']
        task_type_var = customtkinter.StringVar()
        task_type = customtkinter.CTkComboBox(master=model_tab,
                                              values=type_list, width=200,
                                              justify='center',
                                              variable=task_type_var)
        
        # Add ComboBox for choosing the optimizer
        optimizer_label = customtkinter.CTkLabel(master=model_tab, text="Optimizer")
        optimizer_list =['SGD', 'Adam', 'AdamW', 'RMSProp', 'AdaGrad', 'Adadelta', 'Adamax', 'Nadam']
        optimizer_var = customtkinter.StringVar()
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
        pretrained_model_list = ["EfficientNetV2","EfficientNet","MobileNet V2","MobileNet V3","ResNet","Inception V3"]
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
        epochs_var = customtkinter.Variable(value=16)
        epochs = customtkinter.CTkEntry(master=model_tab, 
                                          placeholder_text='16',
                                          width=200, 
                                          justify='center',
                                          textvariable=epochs_var)
        
        # Add save button for saving the inputted values and handle invalid inputs
        # FIXME: the save button event should also handle invalid inputs (very high lr, negative hidden units etc)
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
                                                         'learning_rate': learning_rate_var.get()}
                
                # show a label when the inputs are valid
                show_info(message='Model settings successfully saved',
                          text_color='green')
                
                print(self._settings_dict)
                
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
        
        
        # Grid layout manager
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
        batch_size_var = customtkinter.Variable(value=32)
        batch_size = customtkinter.CTkEntry(data_tab,
                                            justify='center',
                                            textvariable=batch_size_var)
        
        # Entry box for number of classes widget
        num_classes_label = customtkinter.CTkLabel(data_tab, text='Number of Classes')
        num_classes_var = customtkinter.Variable()
        num_classes = customtkinter.CTkEntry(data_tab,
                                             textvariable=num_classes_var,
                                             justify='center')
        
        # Entry box for the value of Train data percentage widget 
        data_split_label = customtkinter.CTkLabel(data_tab, text='Train/Test Split')
        data_split_var = customtkinter.Variable()
        data_split = customtkinter.CTkEntry(data_tab,
                                            textvariable=data_split_var,
                                            justify='center')
        
        # Add save button for saving the inputted values and handle invalid inputs
        # FIXME: the save button event should also handle invalid inputs
        
        def save_button_event() -> None:
            # list of user inputs
            user_inputs = {'data_path': data_path_var.get(),
                           'batch_size': batch_size_var.get()}
            
            if 'C:/path/to/data/directory' == data_path_var.get():
                # show a label when the inputs are valid
                show_info('Invalid Data Directory', text_color='red')
            
            elif '' not in user_inputs.values():
                self._settings_dict['data_settings'] = {'data_path': data_path_var.get(),
                                                         'batch_size': batch_size_var.get()}
                
                # show a label when the inputs are valid
                show_info('Data settings successfully saved', text_color='green')
                print(self._settings_dict)
                
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
        
        data_path.grid(row=0, column=0, columnspan=2, padx=15, pady=20)
        browse_data_button.grid(row=0, column=2, padx=5, pady=20)
        
        num_classes_label.place(x=50, y=60)
        num_classes.place(x=32, y=90)

        data_split_label.place(x=224, y=60)
        data_split.place(x=202, y=90)
        
        batch_size_label.place(x=410, y=60)
        batch_size.place(x=372, y=90)
        
        save_button.place(x=225, y=230)
    
    # TODO: Create a validate _settings_dict method
    def _validate_settings_dict():
        """
        Validate the settings dictionary.
        """
        # this will also be used inside the train_buttn_event
        # to make sure the user inputs does not cause errors in the training process
        pass
    def _create_train_button(self):
        """
        Create a button for starting the training.
        """
        self.train_button = customtkinter.CTkButton(master=self.root, text="Train", command=None)
        self.train_button.pack(pady=10)
        
    def run(self) -> None:
        """
        Run the GUI.
        """
        self.root.mainloop()
        
def main():
    """
    Main function to instantiate and run the GUI.
    """
    gui = ModelBuilderGUI()
    gui.run()

if __name__ == "__main__":
    main()
