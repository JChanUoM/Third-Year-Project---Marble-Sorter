# transfer learning using MobileNet-V3

from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

trainPath = "dataset/dataset_for_model/train"
validatePath = "dataset/dataset_for_model/validate"

# MobileNetV3 requires square image of multiples of 32 (320x320 is used for object detection for better accuracy)
trainGenerator = ImageDataGenerator(
    rotation_range=15, width_shift_range=0.1,
    height_shift_range=0.1, brightness_range=(0.6,1.4)).flow_from_directory(trainPath,target_size=(320,320), batch_size = 32)

ValidGenerator = ImageDataGenerator(
    rotation_range=15, width_shift_range=0.1,
    height_shift_range=0.1, brightness_range=(0.6,1.4)).flow_from_directory(validatePath,target_size=(320,320), batch_size = 32)

# Build the model
baseModel = MobileNetV3Large(weights="imagenet", include_top=False)

x = baseModel.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)


# 5 for none, white, blue, green and transparent
predictionLayer = Dense(5, activation='softmax')(x)
model = Model(inputs=baseModel.input, outputs=predictionLayer)
print(model.summary())

# freeze
for layer in model.layers[:-5]:
    layer.trainable = False

# compile

optimizer = Adam(learning_rate = 0.0001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

# train 
model.fit(trainGenerator, validation_data=ValidGenerator, epochs=5)

modelSavedPath = "dataset/dataset_for_model/MarbleV3.keras"
model.save(modelSavedPath)