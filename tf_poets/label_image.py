import os, sys, glob, argparse
from sklearn import metrics
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ap = argparse.ArgumentParser()

ap.add_argument("-dte", "--dataset-test",
                required = False,
                help = "The dataset for testing.",
                dest='test_dir')
args = vars(ap.parse_args())
testing_images = glob.glob(args['test_dir'] + "**/*.jpg")
mistakes = 0
labels = []
predicted = []
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
in tf.gfile.GFile("retrained_labels.txt")]
# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        for image_path in testing_images:

            print(image_path)
            file_name = os.path.basename(image_path)
            class_name = file_name.split("_")[0].lower()

            # Read in the image_data
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            predictions = sess.run(softmax_tensor, \
                     {'DecodeJpeg/contents:0': image_data})

            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            node_id = top_k[0]
            pred = label_lines[node_id]
            predicted.append(pred)
            labels.append(class_name)
            if class_name != pred:
                mistakes = mistakes + 1
                print("Mistake")

            for i, node_id in enumerate(top_k):
                if i > 3:
                    break
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))


testing_images_count = len(testing_images)
print("Classification:\n %s\n" % (metrics.classification_report(labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels, predicted))

print("Total of testing images: ", testing_images_count)
print("Total of wrong guesses: ", mistakes)
print("Success rate: ", ((testing_images_count - mistakes) * 100.0) / testing_images_count)
