import random
random.seed(0)

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.random.set_seed(0)

import os
import json
from zipfile import ZipFile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Define paths
base_dir = r'C:\Users\bksin\PycharmProjects\plant\.venv\dataset\color' 
img_size = 150  # Image size, adjust based on your needs
batch_size = 32  # Batch size, adjust based on your needs

# Image data generators
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Using 20% of the data for validation
)

#Train generator
train_generator = data_gen.flow_from_directory(
	base_dir,
	target_size = (img_size, img_size),
batch_size=batch_size,
subset='training',
class_mode='categorical'
)

#validation generator
validation_generator = data_gen.flow_from_directory(
base_dir,
target_size=(img_size, img_size),
batch_size=batch_size,
subset='training',
class_mode='categorical'
)




#modeldefinition
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(img_size,img_size, 3)))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(train_generator.num_classes,activation='softmax'))

#modelsummary
model.summary()

#compilethe model
model.compile(optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy'])





#training the model
history = model.fit(
	train_generator,
	steps_per_epoch=train_generator.samples // batch_size,
	epochs=5,
	validation_data=validation_generator,
	validation_steps=validation_generator.samples //batch_size
)


#modelvalidation
print("Evaluating model...")
val_loss, val_accuracy= model.evaluation(validation_generator,steps=validation_generator.samples // batch_size)
print(f"validation accuracy: {val_accuracy*100:.2f}%")


#plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabl('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


#function to load and preprocess the image using pillow
def load_and_preprocess_image(image_path, target_size=(224,224)):
#load_image
img=Image.open(image_path)
#Resize
img=img.resize(target_size)
#convert tonumpy array
img_array=np.array(img)
#add batch idimension
img_array = np.expand>dims(img_array,axis=0)
#scale the image
img_array=img_array.astype('float32')/255.
return img_array

#function to predicttheclass of an image
def predict_image_class(model, imae_path, class indices):
preprocessed_img = load_and_preprocess_image(image_path)
predictions=modelpredict(preprocessed_img)
predicted_class_index = no.argmax(predictions, axis=1)[0]
predicted_class_name = class_indices[predicted_class_index]
return predicted_class_name

#create mapping from class indicesto class names
class_indices = {v: k for k, v in train_generator.class_indices.items()}

#saving class names as json file
json.dump(class_indices, open('class_indices.json', 'w'))

model.save('trained_models/plant_dis_pred_model.h5)