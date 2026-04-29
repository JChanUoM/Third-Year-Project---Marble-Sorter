import os
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from time import time

# get the list of categories:
categories = os.listdir( "dataset/dataset_for_model/train")
categories.sort()
print(categories)

# load the saved model
modelSavedPath = "dataset/dataset_for_model/MarbleV3.keras"
model = tf.keras.models.load_model(modelSavedPath)

#load image function
def load_image(imageFile):
     img = Image.open(imageFile)
     img.load()
     img = img.convert("RGB") #previously RGBA
     img = img.resize((320, 320), Image.LANCZOS) #previously.ANTIALIAS, LANCZOS is the modern one for pillow
     return img

# predict the image

def classify_image(imageFile):
     x = []

     img = load_image(imageFile)
     
     x = image.img_to_array(img)
     x = np.expand_dims(x,axis=0)

     print(x.shape)
     pred = model.predict(x)
     print(pred)

     # get the highest prediction value
     categoryValue = np.argmax(pred, axis=1)
     categoryValue = categoryValue[0]

     print(categoryValue)

     result = categories[categoryValue]
     return result

img_path = "dataset/test_images/green transparent.png"
time_before = time()
resultText = classify_image(img_path)
time_after = time()
keras_model_time = time_after - time_before
print("Total prediction time for keras is: ", keras_model_time)


print(resultText)

img = cv2.imread(img_path)
img = cv2.putText(img, resultText, (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255), 2)
cv2.imshow('img', img)


# testing TFlite
tfliteModelPath = "dataset/dataset_for_model/MarbleV3.tflite"

#load tflite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=tfliteModelPath)

# force input shape to 320x320
input_details = interpreter.get_input_details()
interpreter.resize_tensor_input(input_details[0]['index'], [1,320,320,3])

interpreter.allocate_tensors()

# get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on input data
input_shape = input_details[0]['shape']

#load image
img = Image.open(img_path)
img = img.convert("RGB")
img = img.resize((320, 320), Image.LANCZOS)

input_data = np.array(img, dtype=np.float32) # float32 [0.0, 1.0]
input_data = np.expand_dims(input_data, axis=0)        # shape (1, 320, 320, 3)


interpreter.set_tensor(input_details[0]['index'], input_data)

time_before = time()
interpreter.invoke()
time_after = time()
total_tflite_time = time_after - time_before
print("Total prediction time for tflite without opt model is: ", total_tflite_time)

output_data_tflite =  interpreter.get_tensor(output_details[0]['index'])
print("The tflite without opt prediction for this image is: ", output_data_tflite)

# get the highest prediction value
categoryValue = np.argmax(output_data_tflite, axis=1)
categoryValue = categoryValue[0]

print(categories[categoryValue])

cv2.waitKey(0)
cv2.destroyAllWindows()