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

        # Create model tab widgets
        self.CreateModelTabWidgets()
        
        # Add Train Button for starting training
        # TODO: Add button command for training
        self.train_button = customtkinter.CTkButton(master=self.root, text="Train", command=None)
        self.train_button.pack(pady=10)
        
    # TODO: Create a CreateTabs Method for creating the widgets for customizing the model and the data
    def CreateTabs(self):
        # add tabs for Data loading, and Model Customization
        self.tabview = customtkinter.CTkTabview(self.root)
        self.tabview.pack(fill=customtkinter.BOTH, expand=True)

        self.tabview.add('Model')
        self.tabview.add('Data')
        self.tabview.set('Model') 

        
        
    #========================= MODEL TAB WIDGETS =========================#
    def CreateModelTabWidgets(self):
        model_tab = self.tabview.tab('Model')
        
        # Add a ComboBox for choosing the type of task
        task_type_label = customtkinter.CTkLabel(master=model_tab, text="Task")
        
        type_list = ['Binary Classification', 'Multiclass Classification']
        task_type = customtkinter.CTkComboBox(master=model_tab,
                                                   values=type_list,
                                                   width=200,
                                                   justify='center')
        
        # Add ComboBox for choosing the optimizer
        optimizer_label = customtkinter.CTkLabel(master=model_tab, text="Optimizer")
        
        # list of available optimizers in pytorch
        optimizer_list =['SGD', 'Adam', 'AdamW', 'RMSProp', 'AdaGrad', 'Adadelta', 'Adamax', 'Nadam']

        optimizer = customtkinter.CTkComboBox(master=model_tab,
                                                   values=optimizer_list,
                                                   width=200,
                                                   justify='center')
        
        # pack widgets
        task_type_label.pack(pady=5)
        task_type.pack()
        
        optimizer_label.pack(pady=5)
        optimizer.pack()
        
    def run(self):
        self.root.mainloop()
    
def main():
    gui = ModelBuilder()
    gui.run()

if __name__ == "__main__":
    main()