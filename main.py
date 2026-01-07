'''
Product Classification & Defect Identification (Anomaly Detection)
    ..
Dataset Used: MVTEC-AD (CC BY-NC-SA 4.0) [Non-commercial use only]
    ..
Author: SWAPPY404 <challaniswapnil98@gmail.com>
Date: 30-SEP-2024

BSD 3-Clause License

Copyright (c) 2024, SWAPPY404

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''


# Libraries Required
import os
import cv2
import time
import numpy as np
import random
from tkinter import Tk, Label, Button, font, filedialog
from PIL import Image, ImageTk
import product_defect_classifier

# Application Constants
DISPLAY_IMAGE_INTERVAL = 2.0
GUI_WIDTH = 960
GUI_HEIGHT = 480
IMAGE_DISPLAY_WIDTH = 640
IMAGE_DISPLAY_HEIGHT = 480
TARGET_IMAGE_SIZE = 100
GUI_BG_COLOR = "#000000"

# Color Constants
COLOR_BLACK = "#000000"
COLOR_WHITE = "#FFFFFF"
COLOR_CYAN = "#00FFFF"
COLOR_GREEN = "#00FF00"
COLOR_RED = "#FF0000"
COLOR_DARK_CYAN = "#002F2F"

# GUI state variables
class GUIState:
    """Encapsulate GUI state to avoid global variables."""
    def __init__(self):
        self.product_pred_label = "None"
        self.product_pred_prob = None
        self.defect_pred_label = "None"
        self.defect_pred_prob = None
        self.img_filenames = []
        self.img_indx = 0
        self.dir_flag = False

gui_state = GUIState()

# Initializing GUI
screen = Tk()
screen.geometry(f"{GUI_WIDTH}x{GUI_HEIGHT}")
screen.title("Product Defect Classifier")
screen.configure(bg=GUI_BG_COLOR)

# TK callback function to update content over GUI
def tk_show_update_screen():
    """Update GUI with predictions and display current image."""
    # In case if user browse image
    if not gui_state.dir_flag:
        gui_state.img_indx = 0
    
    # In case if user browse directory
    elif gui_state.dir_flag:
        try:
            # Loading input image
            input_img = cv2.imread(gui_state.img_filenames[gui_state.img_indx])
            if input_img is None:
                raise IOError(f"Failed to load image: {gui_state.img_filenames[gui_state.img_indx]}")
            
            # Classifying product class
            gui_state.product_pred_prob, _, gui_state.product_pred_label = \
                product_defect_classifier.product_class_predict(input_img)
            
            # Resizing image for GUI
            input_img = cv2.resize(input_img, (IMAGE_DISPLAY_WIDTH, IMAGE_DISPLAY_HEIGHT))
            img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            tk_img_frame.imgtk = imgtk
            tk_img_frame.configure(image=imgtk)

            # Update product classification confidence scores
            for i, label in enumerate(product_defect_classifier.products_labels):
                fg = COLOR_RED if label == gui_state.product_pred_label else COLOR_BLACK
                confidence_text = f"{label.upper()} (Conf.Score) -> {gui_state.product_pred_prob[i]*100.0:.3f}%"
                tk_pred_classes[i].configure(text=confidence_text, fg=fg)

            # Identify product condition using appropriate model
            defect_predictor_map = {
                'capsule': product_defect_classifier.capsule_defect_predict,
                'leather': product_defect_classifier.leather_defect_predict,
                'screw': product_defect_classifier.screw_defect_predict
            }
            
            if gui_state.product_pred_label in defect_predictor_map:
                predict_func = defect_predictor_map[gui_state.product_pred_label]
                gui_state.defect_pred_prob, indx, gui_state.defect_pred_label = predict_func(input_img)
                
                # Update product's condition confidence scores
                fg = COLOR_GREEN if gui_state.defect_pred_label == "good" else COLOR_RED
                condition_text = f"Condition -> {gui_state.defect_pred_label.upper()} ({gui_state.defect_pred_prob[indx]*100.0:.3f}%)"
                tk_condition_label.configure(text=condition_text, fg=fg, bg=COLOR_WHITE)

            # Image array index increment (in case of folder)
            gui_state.img_indx = (gui_state.img_indx + 1) % len(gui_state.img_filenames)
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            tk_condition_label.configure(text=f"Error: {str(e)}", fg=COLOR_RED, bg=COLOR_WHITE)

    # Update predicted product label
    tk_prediction_label.configure(text=f"Product : {gui_state.product_pred_label.upper()}")

    # Re-trigger update after interval
    update_interval = 50 if len(gui_state.img_filenames) <= 1 else int(DISPLAY_IMAGE_INTERVAL * 1000)
    tk_img_frame.after(update_interval, tk_show_update_screen)


# Callback function for IMAGE-INPUT button
def browse_img_button_callback():
    """Handle single image file selection."""
    filename = filedialog.askopenfilename(
        initialdir="./",
        title="Select an image file",
        filetypes=(("Image files", "*.jpg *.jpeg *.png *.JPG *.PNG"),
                   ("All files", "*.*"))
    )
    if os.path.exists(filename):
        gui_state.dir_flag = False
        gui_state.img_indx = 0
        gui_state.img_filenames = [filename]
        gui_state.dir_flag = True


# Callback function for FOLDER-INPUT button
def browse_dir_button_callback():
    """Handle folder selection with image file filtering."""
    dirname = filedialog.askdirectory()
    if os.path.exists(dirname):
        gui_state.dir_flag = False
        gui_state.img_indx = 0
        
        # Collect image files with efficient extension checking
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG'}
        gui_state.img_filenames = [
            os.path.join(dp, fn)
            for dp, dn, filenames in os.walk(dirname)
            for fn in filenames
            if os.path.splitext(fn)[1] in valid_extensions
        ]
        
        if len(gui_state.img_filenames) > 0:
            random.shuffle(gui_state.img_filenames)
            gui_state.dir_flag = True

# Create & initialize TK GUI components
tk_img_frame = Label(screen)
tk_img_frame.grid(row=0, column=0)

# Product prediction label
tk_prediction_label = Label(
    screen,
    text=f"Product: {gui_state.product_pred_label}",
    bg=COLOR_DARK_CYAN,
    fg=COLOR_GREEN,
    font=('Segoe UI Semibold', 16)
)
tk_prediction_label.place(x=710, y=10)

# Button to browse image
browse_img_button_obj = Button(
    screen,
    text="IMAGE\nINPUT",
    bg=COLOR_CYAN,
    fg=COLOR_BLACK,
    font=('Segoe UI Semibold', 14),
    command=browse_img_button_callback
)
browse_img_button_obj.place(x=710, y=120)

# Button to browse folder/directory
browse_dir_button_obj = Button(
    screen,
    text="FOLDER\nINPUT",
    bg=COLOR_CYAN,
    fg=COLOR_BLACK,
    font=('Segoe UI Semibold', 14),
    command=browse_dir_button_callback
)
browse_dir_button_obj.place(x=810, y=120)

# Products classification confidence scores
tk_pred_classes = []
for i, product_label in enumerate(product_defect_classifier.products_labels):
    tk_pred_class = Label(
        screen,
        text=f"{product_label.upper()} (Conf.Score) -> NONE",
        bg=COLOR_WHITE,
        fg=COLOR_BLACK,
        font=('Segoe UI Semibold', 14)
    )
    tk_pred_class.place(x=650, y=(26*i + 270))
    tk_pred_classes.append(tk_pred_class)

# Product condition label & its confidence score
tk_condition_label = Label(
    screen,
    text="Condition -> NONE",
    bg=COLOR_WHITE,
    fg=COLOR_BLACK,
    font=('Segoe UI Semibold', 14)
)
tk_condition_label.place(x=650, y=420)

# Entry Point
if(__name__ == "__main__"):
    # Trigger update GUI callback
    tk_show_update_screen()
    # TK mainloop()
    screen.mainloop()
