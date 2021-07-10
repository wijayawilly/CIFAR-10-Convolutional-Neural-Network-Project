import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model, Model
import numpy as np
import keras as k

# untuk memaksimalkan penggunaan GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#Load data dan split data
(train_images, train_labels),(test_images, test_labels) = datasets.cifar10.load_data()

#Normalize Data
train_images = train_images / 255.0
test_images = test_images / 255.0

#Convert menjadi one-hot-encode
num_classes = 10
train_labels = k.utils.to_categorical(train_labels, num_classes)
test_labels = k.utils.to_categorical(test_labels, num_classes)

# Data Augmentation
datagen = ImageDataGenerator(
width_shift_range=0.1,
height_shift_range=0.1,
horizontal_flip=True,)

datagen.fit(train_images)

num_filters=32
ac='relu'
adm=Adam(lr=0.001,decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
drop_dense=0.5
drop_conv=0.2

model = models.Sequential()

model.add(layers.Conv2D(num_filters, (3, 3), activation=ac,input_shape=(32, 32, 3),padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(num_filters, (3, 3), activation=ac,padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))   
model.add(layers.Dropout(drop_conv))

model.add(layers.Conv2D(2*num_filters, (3, 3), activation=ac,padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(2*num_filters, (3, 3), activation=ac,padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))   
model.add(layers.Dropout(drop_conv))

model.add(layers.Conv2D(4*num_filters, (3, 3), activation=ac,padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(4*num_filters, (3, 3), activation=ac,padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(2 * drop_conv))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation=ac))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(drop_dense))
model.add(layers.Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=adm)

model.summary()


history=model.fit_generator(datagen.flow(train_images, train_labels, batch_size=256),
                    steps_per_epoch = len(train_images) / 256, epochs=200, 
                            validation_data=(test_images, test_labels))

loss, accuracy = model.evaluate(test_images, test_labels)
print("Accuracy is : ", accuracy * 100)
print("Loss is : ", loss)


N = 200
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.show()

model.save("cnnmodel.h5")

