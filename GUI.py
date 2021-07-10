import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile, askopenfilename
from PIL import ImageTk, Image
import numpy
from keras.models import load_model, Model
import cv2
import keras
import time
from tensorflow.keras import datasets
import numpy as np
import os
import pickle

# untuk memaksimalkan penggunaan GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

classes = {
    0 : 'Aeroplane',
    1 : 'Automobile',
    2 : 'Bird',
    3 : 'Cat',
    4 : 'Deer',
    5 : 'Dog',
    6 : 'Frog',
    7 : 'Horse',
    8 : 'Ship',
    9 : 'Truck'
}

def open_image():

    dialog_var.set("Waiting image to be loaded.......")
    file_path = askopenfilename(initialdir='/', filetypes=[('jpeg file','*.jpg'), ('png file','*.png')])
    image = Image.open(file_path)
    image = image.resize((320,180))
    photo = ImageTk.PhotoImage(image)

    img_lbl = Label(mid_frame, image=photo)
    img_lbl.image = photo 
    img_lbl.place(x=30, y=80)
    global imgs
    imgs = Image.open(file_path)
    imgs = image.resize((32,32))
    imgs = numpy.expand_dims(imgs, axis=0)
    imgs = numpy.array(imgs)
    print(imgs.shape)
    imgs = imgs / 255
    dialog_var.set("Image loaded!")

def weight_model():
    file_data_name = "test_data.pkl"
    file_lable_name = "test_lable.pkl"

    open_file_data = open(file_data_name, "rb")
    test_images = pickle.load(open_file_data)
    open_file_data.close()

    open_file_label = open(file_lable_name, "rb")
    test_labels = pickle.load(open_file_label)
    open_file_label.close()

    num_classes = 10
    test_labels = keras.utils.to_categorical(test_labels, num_classes)
    test_images = test_images / 255.
    dialog_var.set("Waiting model to be loaded.......")
    weight_path = askopenfilename(initialdir='\\', filetypes=[('Model file','*.h5')])
    filename = os.path.basename(weight_path)
    model_name.set(filename)
    global model
    model = load_model(weight_path)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
    loss, accuracy = model.evaluate(test_images, test_labels)
    test_accuracy.set("%.1f%%" % (accuracy * 100))
    test_loss.set("%.3f" % (loss))
    correct = 0

    pred = model.predict(test_images)
    y_pred = [np.argmax(element) for element in pred]
    y_test = [np.argmax(element) for element in test_labels]

    for i in range (test_labels.shape[0]):
        if y_test[i] == y_pred[i]:
            correct += 1

    pred_result.set("%d of %d test images" % (correct, test_labels.shape[0]))        
    model.summary()

    dialog_var.set("Model loaded!")

def test_predict():
    # train
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
    # predict
    pred = model.predict_classes([imgs])[0]
    print(pred)
    sign = classes[pred]
    result_text = "Prediction Result : "+str(sign)
    test_result_var.set(result_text)
    dialog_var.set("Prediction Complete !")
    


root = tk.Tk()
root.geometry('1920x1080')
root.title("CIFAR-10 Image Classification")

top_frame = Frame(root,width=100,highlightbackground='red', highlightthicknes=3)
top_frame.grid(row=0, column= 0, padx=200, pady=20, ipadx= 500, ipady=50)

mid_frame = Frame(root,width=100,highlightbackground='red', highlightthicknes=3)
mid_frame.grid(row=1, column= 0, padx=200, pady=20, ipadx= 500, ipady=200)

btm_frame = Frame(root,width=100)
btm_frame.grid(row=2, column=0,padx=50, pady=20, ipadx= 200, ipady=50)

#Top Frame
Label_Title = Label(top_frame, text="CNN Model Tester on CIFAR-10 Dataset",font=(None,25))
Label_Title.place(x= 550, y= 45, anchor="center")

#Mid Frame
Label_Image = Label(mid_frame, text="Image Shows Here :",font=(None,15))
Label_Image.place(x=30, y=50)

Label_Model = Label(mid_frame, text="Model info:",font=(None,15))
Label_Model.place(x=800, y=50)

Label_Model = Label(mid_frame, text="Model Name: ",font=(None,10))
Label_Model.place(x=800, y=80)

Label_Model = Label(mid_frame, text="Validation Accuracy: ",font=(None,10))
Label_Model.place(x=800, y=120)

Label_Model = Label(mid_frame, text="Validation Loss: ",font=(None,10))
Label_Model.place(x=800, y=160)

Label_Model = Label(mid_frame, text="Model True Prediction: ",font=(None,10))
Label_Model.place(x=800, y=200)

btn_upload = Button(mid_frame, text='Browse Image',command=open_image)
btn_upload.place(x=30, y=350)

btn_model = Button(mid_frame, text='Browse Model',command=weight_model)
btn_model.place(x=980, y=350)

btn_test = Button(mid_frame,text='Test Prediction', command=test_predict)
btn_test.place(x=475, y=350)

test_result_var = StringVar()
test_result_var.set("Prediction result shown here")
test_result_label = Label(mid_frame,font=("Courier", 15), height=2, textvariable=test_result_var, bg="white", fg="black").place(x=360, y=290)

model_name = StringVar()
model_name.set("None")
model_name_label = Label(mid_frame,font=(None, 10), textvariable=model_name).place(x=880, y=80)

test_accuracy = StringVar()
test_accuracy.set("None")
test_accuracy_label = Label(mid_frame,font=(None, 10), textvariable=test_accuracy).place(x=925, y=120)

test_loss = StringVar()
test_loss.set("None")
test_loss_label = Label(mid_frame,font=(None, 10), textvariable=test_loss).place(x=900, y=160)

pred_result = StringVar()
pred_result.set("None")
pred_result_label = Label(mid_frame,font=(None, 10), textvariable=pred_result).place(x=935, y=200)


#Bottom Frame
dialog_var = StringVar()
dialog_var.set("Welcome to CIFAR-10 Model Tester !")

labelframe1 = LabelFrame(btm_frame, text="Notification Box", bg="yellow")
labelframe1.place(x=250,y=35, anchor="center")

toplabel = Label(labelframe1,font=("Courier", 15), height=2, textvariable=dialog_var, fg="red", bg="lightcyan")
toplabel.grid(row=0, column=0)




root.mainloop()
