import os, sys, glob

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
# image_path = sys.argv[1]

testing_images = glob.glob("/home/gustavo/tf-training/carvao_50/dataset_test/**/*.jpg")
mistakes = 0

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
in tf.gfile.GFile("retrained_labels.txt")]
# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

for image_path in testing_images:

    print image_path
    file_name = os.path.basename(image_path)
    class_name = file_name.split("_")[0].lower()

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        counter = 1
        for node_id in top_k:
            if counter > 3:
                break

            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))

            if counter == 1 and class_name != human_string:
                #The guess is wrong.
                mistakes = mistakes + 1
                print "Mistake"


            counter = counter + 1

testing_images_count = len(testing_images)
print "Total of testing images: ", testing_images_count
print "Total of wrong guesses: ", mistakes
print "Success rate: ", ((testing_images_count - mistakes) * 100.0) / testing_images_count
