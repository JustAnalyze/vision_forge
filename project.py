import customtkinter

# create a class that has all the necessary GUI for training and predicting using a Image classifier model

class ModelBuilder:
    def __init__(self):
        customtkinter.set_default_color_theme('dark-blue')
        customtkinter.set_appearance_mode("dark")
        self.root = customtkinter.CTk()
        self.root.title("Vision Forge")
        self.root.geometry("600x300")
        
        # Create tabs widgets for customizing the model and the data
        self.CreateTabs()

        # Add Train Button for starting training
        # TODO: Add button command for training
        self.button = customtkinter.CTkButton(master=self.root, text="Train", command=None)
        self.button.pack(pady=10)
    # TODO: Create a CreateTabs Method for creating the widgets for customizing the model and the data
    def CreateTabs(self):
        # add tabs for Data loading, and Model Customization
        self.tabview = customtkinter.CTkTabview(self.root)
        self.tabview.pack(fill=customtkinter.BOTH, expand=True)

        self.tabview.add('Model')
        self.tabview.add('Data')
        self.tabview.set('Model') 

        # add widgets in the Model tab for customizing model architecture.
        model_tab = self.tabview.tab('Model')
        label = customtkinter.CTkLabel(master=model_tab, text="Task")
        label.pack(pady=5)
        # Add a ComboBox for choosing the loss function
        classification_type_list = ['Binary Classification', 'Multiclass Classification']
        self.classification_type = customtkinter.CTkComboBox(master=model_tab,
                                                             values=classification_type_list)
        self.classification_type.pack()
        
    def Run(self):
        self.root.mainloop()
    
def main():
    gui = ModelBuilder()
    gui.Run()

if __name__ == "__main__":
    main()