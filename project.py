import customtkinter



# create a class that has all the necessary GUI for training and predicting using a Image classifier model

class GUI:
    def __init__(self):
        customtkinter.set_default_color_theme('dark-blue')
        self.root = customtkinter.CTk()
        self.root.title("Image Classifier")
        self.root.geometry("500x500")
        
        self.CreateTabs()

    def CreateTabs(self):
        # add tabs for Data loading, and Model Customization
        self.tabview = customtkinter.CTkTabview(self.root)
        self.tabview.pack(padx=20, pady=20)
        
        self.tabview.add('Model')
        self.tabview.add('Data')  
        self.tabview.set('Model')  

        
def main():
    gui = GUI()
    gui.root.mainloop()

if __name__ == "__main__":
    main()