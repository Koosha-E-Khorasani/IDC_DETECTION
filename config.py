# Here is the configuration of the project.
# It contains address in which we place out training, validation and test set
# also the percentages of pictures that are used as training set and validation set.
# Further more there paths to different partitions of dataset.

import os
# For Colab
ORIGIN_DATASET_PATH = "/content/dataset/breast_images"
BASE_PATH = "/content/dataset_refined_images"
# ORIGIN_DATASET_PATH = "/dataset/breast_images"
# BASE_PATH = "/dataset_refined_images"
TRAINING_PATH = os.path.sep.join([BASE_PATH, "training"])
VALIDATION_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "test"])

TRAIN_PERCENTAGE = 0.8
VALIDATION_PERCENTAGE = 0.15

NUM_EPOCHS = 40
INIT_LR = 1e-2
BS = 32