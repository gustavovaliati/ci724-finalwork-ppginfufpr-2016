#!/usr/bin/python
import matplotlib.pyplot as plt
import glob, sys, os, json, datetime, argparse, random
import numpy as np
from sklearn import datasets, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from skimage import feature


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
    dataset = {"images" : [] , "labels": [], "class_name": []}
    class_map = {}
    counter = 0
    images = glob.glob(dataset_dir + "/**/*.jpg")

    random.shuffle(images)

    for img_path in images:

        file_name = os.path.basename(img_path)
        class_name = file_name.split("_")[0]
        label = False
        if class_name in class_map:
            label = class_map[class_name]
        else:
            class_map[class_name] = label = counter
            counter += 1

        # dataset["images"].append(plt.imread(img_path))
        im = plt.imread(img_path)
        h_feat = feature.hog(im)
        dataset["images"].append(h_feat)
        dataset["labels"].append(label)
        dataset["class_name"].append(class_name)

    dataset["images"] = np.array(dataset["images"])
    dataset["labels"] = np.array(dataset["labels"])
    dataset["class_map"] = class_map
    print(class_map)

    return dataset

if(args['load_file']):
    print("Loading classifier.")
    classifier = joblib.load(args['load_file'])
elif(args['need_train']):
    train_dataset = load_dataset(args['train_dir'])
    n_samples = len(train_dataset["images"])
    print("Loaded:", n_samples, train_dataset["images"].shape)
    with open('class_map_train_'+now.strftime("%Y-%m-%d_%H%M%S")+'.json', 'w') as f:
        json.dump(train_dataset['class_map'], f)
else:
    print("Error: No classifier or training dataset specified");
    sys.exit()

if(args['need_test']):
    test_dataset = load_dataset(args['test_dir'])
    n_samples = len(test_dataset["images"])
    print("Loaded:", n_samples, test_dataset['images'].shape)
    with open('class_map_test_'+now.strftime("%Y-%m-%d_%H%M%S")+'.json', 'w') as f:
        json.dump(test_dataset['class_map'], f)

if args['need_train'] and 'classifier' not in locals():
    print("Training...")
    print(train_dataset["labels"][0], train_dataset["class_name"][0])
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(train_dataset["images"], train_dataset["labels"])

    joblib.dump(classifier, 'classifier_'+now.strftime("%Y-%m-%d_%H%M%S")+'.pkl')

if args['need_test']:
    print("Predicting...")
    predicted = classifier.predict(test_dataset["images"])
    print(predicted)
    print("Classification %s:\n%s\n"
    % (classifier, metrics.classification_report(test_dataset["labels"], predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_dataset["labels"], predicted))
