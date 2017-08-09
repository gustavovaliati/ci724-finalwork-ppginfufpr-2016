#!/usr/bin/python
import datetime, argparse, sys, os
import numpy as np
from multiprocessing import Pool

ap = argparse.ArgumentParser()
ap.add_argument("-tr", "--train", required = True, help = "Is the training dataset path.")
ap.add_argument("-te", "--test", required = True, help = "Is the testing dataset path.")
ap.add_argument("-k", required = True, help = "Is K for the KNN algorithm.")
ap.add_argument("-lte", "--limit-test", required = False, help = "Sets a limit for how many testing sets must be used instead of the whole file.")
ap.add_argument("-ltr", "--limit-train", required = False, help = "Sets a limit for how many training sets must be used instead of the whole file.")
ap.add_argument("-p", "--print-all", nargs='?', const=True, required = False, help = "Prints summary on every test. Can slow down the execution on small training sets.")

args = vars(ap.parse_args())

train_file_path = args["train"]
test_file_path = args["test"]
k_number = int(args["k"])

test_calculation_limit = False
test_calculation_limit_arg = args["limit_test"]
if (test_calculation_limit_arg):
    test_calculation_limit = int(test_calculation_limit_arg)
else:
    print "Be aware you didn't set a limit for the testing set. We are going to test all."

train_calculation_limit = False
train_calculation_limit_arg = args["limit_train"]
if (train_calculation_limit_arg):
    train_calculation_limit = int(train_calculation_limit_arg)
else:
    print "Be aware you didn't set a limit for the training set. We are going to use it all."

print_all = False
print_all_arg = args["print_all"]
if (print_all_arg):
    print_all = print_all_arg

############
# STATIC PARAMETERS
############

classes = 10 #todo remove harded coded.
confusion_matrix = np.zeros((classes,classes), dtype=np.int)
result_error = 0
result_rejection = 0
total_testing = 0
total_training = 0
process_number = 4


############
#LOAD TRAINING FILE
############

train_file = open(train_file_path, "r")

print "Reading file: ", train_file_path

header = train_file.readline().split(" ")
train_number_lines = int(header[0])
number_features = int(header[1])

print "Lines {} | Features {}".format(train_number_lines, number_features)
if train_calculation_limit:
    print "We are limiting to {} training sets.".format(train_calculation_limit)
    if train_number_lines > train_calculation_limit:
        total_training = train_calculation_limit
    else:
        print "\nERROR: the training limit is bigger than the actual number of testing sets."
        sys.exit()
else:
    total_training = train_number_lines

train_features = []
train_real_class = []
train_guessed_class = []

for train_index, features in enumerate(train_file):
    if train_calculation_limit and train_index >= train_calculation_limit:
        break

    features = features.split(" ")
    features_class = features.pop(number_features)
    features = np.array(map(float, features))
    features_class = int(features_class.replace("\n",""))

    train_features.append(features)
    train_real_class.append(features_class)


############
#LOAD TEST FILE
############

test_file = open(test_file_path, "r")

print "Reading file: ", test_file_path

header = test_file.readline().split(" ")
test_number_lines = int(header[0])
number_features = int(header[1])
print "Lines {} | Features {}".format(test_number_lines, number_features)
if test_calculation_limit:
    print "We are limiting to {} testing sets.".format(test_calculation_limit)
    if test_number_lines > test_calculation_limit:
        total_testing = test_calculation_limit
    else:
        print "\nERROR: the testing limit is bigger than the actual number of testing sets."
        sys.exit()
else:
    total_testing = test_number_lines

test_features = []
test_real_class = []
test_guessed_class = []

test_processed_lines = 0
for test_index, features in enumerate(test_file):
    if test_calculation_limit and test_index >= test_calculation_limit:
        break

    features = features.split(" ")
    features_class = features.pop(number_features)
    features = np.array(map(float, features))
    features_class = int(features_class.replace("\n",""))

    test_features.append(features)
    test_real_class.append(features_class)


############
# CALCULATION
############

def print_summary(tested):
    valid_total = tested - result_rejection
    time_end = datetime.datetime.now()
    print "Calculation time: {}".format(time_end - time_start)

    if valid_total > 0:
        correct = (valid_total - result_error) * 100.0 / valid_total
    else:
        correct = 0.0

    print "Tested {} | Error: {} | Rejection {} | Correct {} %".format(tested, result_error, result_rejection, correct)
    print confusion_matrix

def calc_distance(test_feat_index, train_feat_index):
    return np.sum((test_features[test_feat_index] - train_features[train_feat_index])**2)

def calc_train(start, end, test_feat_index):
    # print "pid", os.getpid(), start, end
    # current_ranking = np.zeros(0)
    current_ranking = np.zeros(0, dtype=np.float16)

    dictionary = {}
    for index in range(start, end):
        distance = calc_distance(test_feat_index, index)

        if current_ranking.size >= k_number:
            # dictionary.pop(current_ranking[k_number-1])
            current_ranking = np.delete(current_ranking, k_number-1, 0)

        current_ranking = np.append(current_ranking, distance)
        # print distance
        dictionary[distance] = train_real_class[index]
        current_ranking = np.sort(current_ranking, kind="mergesort")


    # print current_ranking, dictionary
    new_dic = {}
    for r in current_ranking:
        new_dic[r] = dictionary[r]

    return new_dic, current_ranking

time_start = datetime.datetime.now()
offset = int(total_training / process_number)
pool = Pool(processes=process_number)


for test_index, test_feat in enumerate(test_features):
    start = 0
    workers = []
    for i in range(process_number):
        end = start+offset
        worker = pool.apply_async(calc_train, (start, end, test_index))
        workers.append(worker)
        start = end
    # print "workers",workers

    k_ranking_dict = {}
    ranking = []
    for worker in workers:
        d, r = worker.get()
        ranking = np.concatenate((ranking, r))
        k_ranking_dict.update(d)

    ranking = np.sort(ranking, kind="mergesort")
    ranking = ranking[0:k_number]
    # print "here",ranking, k_ranking_dict

    to_count_array = []
    for r in ranking:
        # print "k_ranking_dict[key]",k_ranking_dict[key]
        to_count_array.append(k_ranking_dict[r])
    counting = np.bincount(to_count_array)
    guessed_class = np.argmax(counting)
    guessed_counter = counting[guessed_class]
    counting = np.delete(counting, guessed_class)
    if guessed_counter in counting:
        result_rejection = result_rejection + 1
        continue
    real_class = test_real_class[test_index]

    confusion_matrix[real_class,guessed_class] = confusion_matrix[real_class,guessed_class] + 1
    # print real_class, guessed_class

    if real_class != guessed_class:
        result_error = result_error + 1

    if print_all:
        print_summary(test_index+1)


############
# END - PRESENT RESULTS
############

print "\n FINISHED. Final summary: \n"
tested = len(test_features)
print_summary(tested)
