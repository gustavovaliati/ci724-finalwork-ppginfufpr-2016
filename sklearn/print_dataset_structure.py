#!/usr/bin/python
import matplotlib.pyplot as plt
import glob, sys, os, json, datetime, argparse, random
import numpy as np


now = datetime.datetime.now()

train_dir = "../tf_img_classifier_scratch/data/original_jpg_structured_train-validation-only"
test_dir = "../tf_img_classifier_scratch/data/original_jpg_structured_test-only"

ap = argparse.ArgumentParser()

ap.add_argument("-dtr", "--dataset-train",
                required = False,
                help = "The dataset for training.",
                default= train_dir,
                dest='train_dir')
ap.add_argument("-dte", "--dataset-test",
                required = False,
                help = "The dataset for testing.",
                default= test_dir,
                dest='test_dir')
ap.add_argument("-l", "--load",
                required = False,
                default = False,
                help = "Load a saved classifier for testing.",
                dest='load_file')
ap.add_argument("-te", "--test",
                action='store_true',
                help = "The test dataset will be tested.",
                dest="need_test")
ap.add_argument("-tr", "--train",
                action='store_true',
                help = "The training will happen using the training dataset",
                dest='need_train')

args = vars(ap.parse_args())

"""
dataset_test = 84
dataset_training = 138
dataset_validation = 58
"""

def load_dataset(dataset_dir):
    print("Loading dataset:",dataset_dir)
    dataset = {"file_name" : [] , "labels": [], "class_name": [], "img_path": []}
    class_map = {}
    counter = 0
    images = glob.glob(dataset_dir + "/**/*.jpg")

    # random.shuffle(images)

    for img_path in images:

        file_name = os.path.basename(img_path)
        class_name = file_name.split("_")[0]
        label = False
        if class_name in class_map:
            label = class_map[class_name]
        else:
            class_map[class_name] = label = counter
            counter += 1

        dataset["img_path"].append(img_path)
        dataset["file_name"].append(file_name)
        dataset["labels"].append(label)
        dataset["class_name"].append(class_name)

    dataset["labels"] = np.array(dataset["labels"])
    dataset["class_map"] = class_map
    print(class_map)

    to_print = np.empty(0)

    for i in range(len(images)):
        to_print = np.append(to_print, {'class': dataset['class_name'][i], 'label': dataset['labels'][i], 'file_name': dataset['file_name'][i], 'img_path': dataset['img_path'][i]})

    return to_print

train_dataset = load_dataset(args['train_dir'])
np.savetxt('dataset_print_train'+now.strftime("%Y-%m-%d_%H%M%S")+'.txt', train_dataset, fmt="%s")

test_dataset = load_dataset(args['test_dir'])
np.savetxt('dataset_print_test'+now.strftime("%Y-%m-%d_%H%M%S")+'.txt', test_dataset, fmt="%s")
