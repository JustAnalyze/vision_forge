from tkinter import *
import customtkinter
from PIL import Image

customtkinter.set_appearance_mode("light")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green

# Create the main window using customtkinter
root = customtkinter.CTk()

root.title('Tkinter.com - CustomTkinter Images')
root.geometry('400x550')

# Use raw string (r) for the image path or double backslashes
image_path = r'data\pizza_steak_sushi\test\pizza\194643.jpg'
try:
    my_image = customtkinter.CTkImage(
        light_image=Image.open(image_path),
        dark_image=Image.open(image_path),
        size=(180, 250)  # Width x Height
    )

except Exception as e:
    print(f"Error loading image: {e}")

root.mainloop()
