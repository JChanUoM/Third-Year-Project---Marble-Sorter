# Python 3.13.2
# tensor flow version 2.20.0

import os
import random
import shutil # built in python module interface for file and directory operations

# 85-15 split
splitsize = .85
categories = []

source_folder = "dataset/marble_dataset"
folders = os.listdir(source_folder)
print(folders)

for subfolder in folders:
    if os.path.isdir(source_folder + "/" + subfolder):
        categories.append(subfolder)

categories.sort()
print(categories)

# create a target folder
target_folder = "dataset/dataset_for_model"
existDataSetPath = os.path.exists(target_folder)
if existDataSetPath == False:
    os.mkdir(target_folder)

# create function to split training and validation data
def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    files=[]

    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        print(file)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + "is 0 length, ignore it...")
    print(len(files))
    
    trainingLength = int(len(files) * SPLIT_SIZE)
    shuffleSet = random.sample(files, len(files))
    trainingSet = shuffleSet[0:trainingLength]
    validSet = shuffleSet[trainingLength:]

    # copy the train images
    for filename in trainingSet:
        thisFile = SOURCE + filename
        destination = TRAINING + filename
        shutil.copyfile(thisFile, destination)
    
    # copy the validation images
    for filename in validSet:
        thisFIle = SOURCE + filename
        destination = VALIDATION + filename
        shutil.copyfile(thisFile, destination)

trainPath = target_folder + "/train"
validatePath = target_folder + "/validate"

# create the target folders:
existDataSetPath = os.path.exists(trainPath)
if existDataSetPath == False:
    os.mkdir(trainPath)

existDataSetPath = os.path.exists(validatePath)
if existDataSetPath == False:
    os.mkdir(validatePath)

# run the function for each folder
for category in categories:
    trainDestPath = trainPath + "/" + category
    validateDestPath = validatePath + "/" + category
    
    if os.path.exists(trainDestPath) == False:
        os.mkdir(trainDestPath)
    if os.path.exists(validateDestPath) == False:
        os.mkdir(validateDestPath)

    sourcePath = source_folder + "/" + category + "/"
    trainDestPath = trainDestPath + "/"
    validateDestPath= validateDestPath + "/"

    print("Copy from: "+ sourcePath + " to: " + trainDestPath + " and " + validateDestPath)
    split_data(sourcePath, trainDestPath, validateDestPath, splitsize)

