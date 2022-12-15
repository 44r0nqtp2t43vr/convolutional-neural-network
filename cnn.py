import numpy as np
# import tkinter as tk 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import keras.utils as image
# from tkinter import filedialog 

cnn = Sequential()
cnn.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Convolution2D(32, 3, 3, activation='relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Flatten())
cnn.add(Dense(128, activation = 'relu'))
cnn.add(Dense(1, activation = 'sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64,64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64,64), batch_size=32, class_mode='binary')
cnn.fit(x = training_set, validation_data = test_set, epochs = 10)

# root = tk.Tk() 
# root.withdraw() 
# file_path = filedialog.askopenfilename() 
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)