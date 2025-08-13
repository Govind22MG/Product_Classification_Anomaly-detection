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
from tkinter import *
from tkinter import font
from tkinter import filedialog
from PIL import Image, ImageTk
import product_defect_classifier

# Interval between images (in case of multiple images)
DISPLAY_IMAGE_INTERVAL = 2.0

# Globals
product_pred_label, product_pred_prob = "None", "None"
defect_pred_label, defect_pred_prob = "None", "None"
img_filenames = []
img_indx = 0
dir_flag = False

# Initializing GUI
screen = Tk()
screen.geometry("960x480")
screen.title("Product Defect Classifier")
screen.configure(bg="#000000")

# TK callback function to update content over GUI
def tk_show_update_screen():
    global product_pred_label, product_pred_prob, defect_pred_label, defect_pred_prob
    global dir_flag, img_filenames, img_indx, DISPLAY_IMAGE_INTERVAL

    # In case if user browse image
    if(dir_flag == False):
        img_indx = 0
    # In case if user browse directory
    elif(dir_flag == True):
        # Loading input image
        input_img = cv2.imread(img_filenames[img_indx])
        # Classifying product class
        product_pred_prob,_,product_pred_label = product_defect_classifier.product_class_predict(input_img)
        # Resizing image for GUI
        input_img = cv2.resize(input_img, (640, 480))
        # Rearranging color channels
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        # PIL image format
        img = Image.fromarray(img)
        # Setting PIL format image into TK GUI
        imgtk = ImageTk.PhotoImage(image = img)
        tk_img_frame.imgtk = imgtk
        tk_img_frame.configure(image=imgtk)

        # Update product classification confidence scores
        for i in range(len(product_defect_classifier.products_labels)):
            fg="#000000"
            # Highlight predicted class
            if(product_defect_classifier.products_labels[i] == product_pred_label):
                fg="#FF0000"
            tk_pred_classes[i].configure(text="{} (Conf.Score) -> {:.3f}%"\
                                         .format(product_defect_classifier.products_labels[i].upper(),\
                                                 product_pred_prob[i]*100.0),\
                                         fg=fg)

        # Identifying product condition
        if(product_pred_label == 'capsule'):
            defect_pred_prob,indx,defect_pred_label = product_defect_classifier.capsule_defect_predict(input_img)
        elif(product_pred_label == 'leather'):
            defect_pred_prob,indx,defect_pred_label = product_defect_classifier.leather_defect_predict(input_img)
        elif(product_pred_label == 'screw'):
            defect_pred_prob,indx,defect_pred_label = product_defect_classifier.screw_defect_predict(input_img)

        # Update product's condition confidence scores
        fg="#ff0000"
        bg="#ffffff"
        # Highlight 'GOOD' condition as green
        if(defect_pred_label == "good"):
            fg="#00ff00"
        tk_condition_label.configure(text="Condition -> {} ({:.3f}%)"\
                                     .format(defect_pred_label.upper(),\
                                             defect_pred_prob[indx]*100.0),\
                                     fg=fg,
                                     bg=bg)

        # Image array index increment (in case of folder)
        img_indx = img_indx + 1
        if(img_indx == len(img_filenames)):
            img_indx = 0

    # Update predicted product label
    tk_prediction_label.configure(text="Product : " + product_pred_label.upper())

    # Re-trigger 'tk_show_update_screen' after sometime
    tk_img_frame.after((50 if(len(img_filenames) <= 1) else int(DISPLAY_IMAGE_INTERVAL*1000)),\
                       tk_show_update_screen)


# Callback function for IMAGE-INPUT button
def browse_img_button_callback():
    global product_pred_label, product_pred_prob, defect_pred_label, defect_pred_prob
    global dir_flag, img_filenames, img_indx
    filename = filedialog.askopenfilename(initialdir = "./",
                                          title = "Select an 80x80 image file",
                                          filetypes = (("all files", "*.*"),
                                                       ("Text files", "*.txt*")))
    if(os.path.exists(filename)):
        dir_flag = False
        img_indx = 0
        img_filenames = [filename]
        dir_flag = True

# Callback function for FOLDER-INPUT button
def browse_dir_button_callback():
    global img_filenames, dir_flag
    dirname = filedialog.askdirectory()
    if(os.path.exists(dirname)):
        dir_flag = False
        img_indx = 0
        img_filenames = [os.path.join(dp, fn)\
                         for dp, dn, filenames in os.walk(dirname)\
                         for fn in filenames\
                         if (os.path.splitext(fn)[1] == '.JPG' or\
                         os.path.splitext(fn)[1] == '.jpg' or\
                         os.path.splitext(fn)[1] == '.PNG' or\
                         os.path.splitext(fn)[1] == '.png')]
        if(len(img_filenames) > 0):
            random.shuffle(img_filenames)
            dir_flag = True

# Create & initialize TK GUI components
tk_img_frame = Label(screen)
tk_img_frame.grid(row=0, column=0)

# Product prediction label
tk_prediction_label = Label(screen,
                            text="Product: " + product_pred_label,
                            bg="#002f2f",
                            fg="#00ff00",
                            font=('Segoe UI Semibold', 16))
tk_prediction_label.place(x=710, y=10)

# Button to browse image
browse_img_button_obj = Button(screen,
                               text ="IMAGE\nINPUT",
                               bg="#00ffff",
                               fg="#000000",
                               font=('Segoe UI Semibold', 14),
                               command = browse_img_button_callback)
browse_img_button_obj.place(x=710, y=120)

# Button to browse folder/directory
browse_dir_button_obj = Button(screen,
                               text ="FOLDER\nINPUT",
                               bg="#00ffff",
                               fg="#000000",
                               font=('Segoe UI Semibold', 14),
                               command = browse_dir_button_callback)
browse_dir_button_obj.place(x=810, y=120)

# Products classification confidence scores
tk_pred_classes = []
for i in range(len(product_defect_classifier.products_labels)):
    tk_pred_class = Label(screen,
                          text="{} (Conf.Score) -> NONE"\
                          .format(product_defect_classifier.products_labels[i].upper()),
                          bg="#ffffff",
                          fg="#000000",
                          font=('Segoe UI Semibold', 14))
    tk_pred_class.place(x=650, y=((26*i)+270))
    tk_pred_classes.append(tk_pred_class)

# Product condition label & its confidence score
tk_condition_label = Label(screen,
                           text="Condition -> NONE",
                           bg="#ffffff",
                           fg="#000000",
                           font=('Segoe UI Semibold', 14))
tk_condition_label.place(x=650, y=420)

# Entry Point
if(__name__ == "__main__"):
    # Trigger update GUI callback
    tk_show_update_screen()
    # TK mainloop()
    screen.mainloop()
