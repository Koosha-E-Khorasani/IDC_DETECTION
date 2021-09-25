# Here I am going to organize the raw dataset and break it down to 3
# categories which are training,validation and test.
# Before running the network code this file should be run separately
import config
from imutils import paths
import shutil
import random
import os

imagePaths = list(paths.list_images(config.ORIGIN_DATASET_PATH))
random.shuffle(imagePaths)

training_separator_index = int(len(imagePaths) * config.TRAIN_PERCENTAGE)
train_paths = imagePaths[:training_separator_index]
test_paths = imagePaths[training_separator_index:]

validation_separator_index = int(len(train_paths) * config.VALIDATION_PERCENTAGE)
validation_paths = train_paths[:validation_separator_index]
train_paths = train_paths[validation_separator_index:]

datasets = [
    ("training", train_paths, config.TRAINING_PATH),
    ("validation", validation_paths, config.VALIDATION_PATH),
    ("testing", test_paths, config.TEST_PATH)
]
print("[INFO] 'total number of images  is {}".format(len(imagePaths)))
for (data_type, input_images, input_path) in datasets:
    print("*" * 8)
    print("[INFO] 'start the configuration of {}".format(data_type))

    if not os.path.exists(input_path):
        print("[INFO] 'creating {}' directory".format(input_path))
        os.makedirs(input_path)

    counter = 1
    l = len(input_images)
    print("[INFO] 'total number of images for {} is {}".format(data_type,l))
    for inputPath in input_images:
        print("[INFO] 'progress :{:.3f} %".format((counter/l)*100), end='\r')
        filename = inputPath.split(os.path.sep)[-1]
        label = filename[-5:-4]
        labelPath = os.path.sep.join([input_path, label])
        if not os.path.exists(labelPath):
            print("[INFO] 'creating {}' directory".format(labelPath))
            os.makedirs(labelPath)
        p = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, p)
        counter += 1
