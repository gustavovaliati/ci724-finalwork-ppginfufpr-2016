#!/usr/bin/python

import glob,sys,os,shutil

"""
#REMEMBER TO CONVERT TIF TO JPG WITH:

for f in *.tif; do  echo "Converting $f"; convert "$f"  "$(basename "$f" .tif).jpg"; done

"""

dataset_dir = "/home/gustavo/workspace/coal/dataset/original_jpg"
destination_dir_train = "./data/original_jpg_structured_train-validation-only"
destination_dir_test = "./data/original_jpg_structured_test-only"

if os.path.exists(destination_dir_train) or os.path.exists(destination_dir_test):
    print("ERROR: The destination dataset directories already exist.")
    sys.exit()

images = glob.glob(dataset_dir + "/*.jpg")

"""
dataset_test = 84
dataset_training = 138
dataset_validation = 58
"""
image_number_by_class = 280
class_number = 10
class_sequences = 7
testing_percentage = 0.3
training_percentage = 0.7
validation_percentage = 0.3

num_testing_by_class =  int(image_number_by_class * testing_percentage)
num_testing_by_sequence = int(num_testing_by_class / class_sequences)
print("Testing by class: ", num_testing_by_class, " | Training by class: ", image_number_by_class - num_testing_by_class)

# Build folder structure for each class
counter = {}
for img_path in images:

    file_name = os.path.basename(img_path)
    class_name = file_name.split("_")[0]
    image_sequence = file_name.split("_")[1]

    if class_name+image_sequence in counter:
        counter[class_name+image_sequence] = counter[class_name+image_sequence] + 1
    else:
        counter[class_name+image_sequence] = 1


    class_dir_test = destination_dir_test + "/" + class_name
    class_dir_train = destination_dir_train + "/" + class_name

    if not os.path.exists(class_dir_test):
        os.makedirs(class_dir_test)

    if not os.path.exists(class_dir_train):
        os.makedirs(class_dir_train)

    if counter[class_name+image_sequence] <= num_testing_by_sequence:
        shutil.copyfile(img_path, class_dir_test + "/" + file_name)
        print(img_path, counter[class_name+image_sequence], 'moved to TESTING')
    else:
        shutil.copyfile(img_path, class_dir_train + "/" + file_name)
        # print(img_path, 'moved to TRAINING')
print(counter)
print(len(counter))
