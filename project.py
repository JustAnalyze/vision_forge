from pathlib import Path
from tkinter import filedialog
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
                save_info_label.configure(text='Model settings successfully saved',
                                          text_color='green',
                                          justify='center')
                
                save_info_label.place(x=170, y=200)
                
                print(self._settings_dict)
                
            else:
                # show a label about the blank input error
                save_info_label.configure(text='Please Fill all Fields.',
                                          text_color='red',
                                          justify='center')
                
                save_info_label.place(x=200, y=200)
                
                print(f"Please Fill all Fields.")
                
        save_info_label = customtkinter.CTkLabel(master=model_tab, text='', justify='center')
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

        save_button.place(x=215, y=230)
    
    def _create_data_tab_widgets(self) -> None:
        """
        Create widgets for the Data tab.
        """
        # set data tab
        data_tab = self.tabview.tab('Data')
        
        # Function to handle button click event
        # TODO: When the data directory has been set and has valid folder structure (with test and train folder) the number of classes and amount of data should be detected and the entry box for the classes should be occupied and disabled so its not allowed to be editted
        def browse_data() -> None:
            '''
            This function opens a file dialog and gets the selected directory.
            If a directory is selected, sets the value of the 'data_path_var' variable to the selected directory.
            '''
            dir = filedialog.askdirectory() # Open file dialog and get selected directory
            if dir:
                # TODO: this function should also show if the selected directory has the valid folder structure (train and test inside)
                print(f'Data Directory: {dir}')
                
                # get the file and folder names from the selected directory
                files_and_folders = [path.name for path in Path(dir).glob('*')]
                train_folder_path = [path for path in Path(dir).glob('*') if path.name == 'train']
                test_folder_path = [path for path in Path(dir).glob('*') if path.name == 'test']
                # check if the directory has a test and train folder inside it
                if 'train' in files_and_folders and 'test' in files_and_folders:
                    #TODO: set the number of classes variable to the number of classes found inside the train folder and show it on the widget
                    #TODO: Do the same for the train data percentage. set it to 100% if there is no test data/ no test folder and inform the user that test data is necessary
                    print('\n[INFO] train and test folders found!')
                    print(f'Files and Folders: {files_and_folders}')
                    
                    # get total number of training data.
                    classes = [path.name for path in train_folder_path[0].glob('*')]
                    print(f'Classes: {classes}\nNumber of Classes: {len(classes)}')
                    
                    # get total number of training data.
                    num_of_train_data = len([path.name for path in train_folder_path[0].glob('**/*')]) - len(classes)
                    print(f'\nTotal Train Data: {num_of_train_data}')

                    # get total number of test data.
                    num_of_test_data = len([path.name for path in test_folder_path[0].glob('**/*')]) - len(classes)
                    print(f'Total Test Data: {num_of_test_data}\n')

                    # Get percentage of training data
                    percent_train_data = (num_of_train_data / (num_of_test_data + num_of_train_data)) * 100
                    print(f'Percentage of Train Data: {percent_train_data}%')
                    
                    data_path_var.set(dir)
                    
                # else if there is train and no test folder inform user about the missing folder
                elif 'train' in files_and_folders and 'test' not in files_and_folders:
                    
                    print(f'Files and Folders: {files_and_folders}')
                    
                    print('[INFO] test folder not found!')
                
                # else if there is no train and there is test folder inform user about the missing folder
                elif 'train' not in files_and_folders and 'test' in files_and_folders:
                    
                    print(f'Files and Folders: {files_and_folders}')
                    
                    print('[INFO] train folder not found!')
                
                # else both folder missing
                else:
                    
                    print(f'Files and Folders: {files_and_folders}')
                    
                    print('[INFO] train and test folder not found!')
        
        # Entry box for setting the direvtory of the data
        data_path_var = customtkinter.StringVar(value='C:/path/to/data/directory')
        data_path = customtkinter.CTkEntry(master=data_tab, 
                                          width=383, 
                                          justify='center',
                                          textvariable=data_path_var)
        
        # Button to browse directory
        browse_data_button = customtkinter.CTkButton(data_tab,
                                                     width=90,
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
                                             textvariable=num_classes_var)
        
        # Entry box for the value of Train data percentage widget 
        data_split_label = customtkinter.CTkLabel(data_tab, text='Train/Test Split')
        data_split_var = customtkinter.Variable()
        data_split = customtkinter.CTkEntry(data_tab,
                                            textvariable=data_split_var)
        
        # Add save button for saving the inputted values and handle invalid inputs
        # FIXME: the save button event should also handle invalid inputs
        
        def save_button_event() -> None:
            # list of user inputs
            user_inputs = {'data_path': data_path_var.get(),
                           'batch_size': batch_size_var.get()}
            
            if 'C:/path/to/data/directory' == data_path_var.get():
                # show a label when the inputs are valid
                save_info_label.configure(text='Invalid Data Directory',
                                          text_color='red',
                                          justify='center')
                
                save_info_label.place(x=205, y=200)
            
            elif '' not in user_inputs.values():
                self._settings_dict['data_settings'] = {'data_path': data_path_var.get(),
                                                         'batch_size': batch_size_var.get()}
                
                # show a label when the inputs are valid
                save_info_label.configure(text='Data settings successfully saved',
                                          text_color='green',
                                          justify='center')
                
                save_info_label.place(x=170, y=200)
                
                print(self._settings_dict)
                
            else:
                # show a label about the blank input error
                save_info_label.configure(text='Please Fill all Fields.',
                                          text_color='red',
                                          justify='center')
                
                save_info_label.place(x=200, y=200)
                
                print(f"Please Fill all Fields.")
                
        save_info_label = customtkinter.CTkLabel(master=data_tab, text='', justify='center')
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
        
        save_button.place(x=215, y=230)
    
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
