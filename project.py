from doctest import master
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
        self.root.geometry("535x370")
        
        # Set up the dictionary of settings
        self._model_settings_dict = {}
        
        # Create tabs widgets for customizing the model and the data
        self._create_tabs()
        
        # Create model tab widgets
        self._create_model_tab_widgets()
        
        # Create Train Button for starting training
        self._create_train_button()

    # create a get_settings for getting the model settings.
    @property
    def model_settings_dict(self):
        return self._model_settings_dict

    @model_settings_dict.setter
    def model_settings_dict(self, new_settings):
        if isinstance(new_settings, dict):
            if '' in new_settings.values():
                raise ValueError("Settings must not contain empty strings.")
            self._model_settings_dict = new_settings
        else:
            raise ValueError("Settings must be a dictionary.")
    
    def _create_tabs(self):
        """
        Create tabs for customizing the model and the data.
        """
        self.tabview = customtkinter.CTkTabview(self.root)
        self.tabview.pack(fill=customtkinter.BOTH, expand=True)

        self.tabview.add('Model')
        self.tabview.add('Data')
        self.tabview.set('Model') 

    def _create_model_tab_widgets(self):
        """
        Create widgets for the Model tab.
        """
        # TODO: Add variable for each input to store model configurations data
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
        
        # Add save button for saving the inputted values and handle invalid inputs
        def save_button_event():
            try:
                self.model_settings_dict = {'task_type': task_type_var.get(),
                                        'optimizer': optimizer_var.get(),
                                        'num_hidden_units': num_hidden_units_var.get(),
                                        'pretrained_model': pretrained_model_var.get(),
                                        'learning_rate': learning_rate_var.get()}
                print(self.model_settings_dict)
                
            except ValueError:
                save_info_label.configure(text='Please Fill all Settings.',
                                          text_color='red',
                                          justify='center')
                print(f"Please Fill all settings.")
                
        save_info_label = customtkinter.CTkLabel(master=model_tab, text='', justify='center')
        save_button = customtkinter.CTkButton(model_tab,
                                              text='Save',
                                              width=90,
                                              height=28,
                                              command=save_button_event)
        
        # Grid layout manager
        task_type_label.grid(row=0, column=0, padx=30)
        task_type.grid(row=1, column=0, pady=5, padx=30)
        optimizer_label.grid(row=2, column=0, padx=30)
        optimizer.grid(row=3, column=0, pady=5, padx=30)
        num_hidden_units_label.grid(row=0, column=1, padx=30)
        num_hidden_units.grid(row=1, column=1, padx=30)
        pretrained_model_label.grid(row=2, column=1)
        pretrained_model.grid(row=3, column=1)
        learning_rate_label.grid(row=4, column=0, padx=30)
        learning_rate.grid(row=5, column=0, padx=30)
        save_info_label.place(x=200, y=200)
        save_button.place(x=215, y=230)
    
    def _create_train_button(self):
        """
        Create a button for starting the training.
        """
        self.train_button = customtkinter.CTkButton(master=self.root, text="Train", command=None)
        self.train_button.pack(pady=10)
        
    def run(self):
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
    gui.model_settings_dict
if __name__ == "__main__":
    main()
