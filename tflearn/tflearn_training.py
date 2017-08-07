from pathlib import Path
import h5py
import tflearn

# Load path/class_id image file:
dataset_origin_train = './data/original_jpg_structured_train-validation-only/'
dataset_origin_test = './data/original_jpg_structured_test-only/'

dataset_train_hdf5 = 'coal_train_dataset.h5'
dataset_test_hdf5 = 'coal_test_dataset.h5'

# Build a HDF5 dataset (only required once)
from tflearn.data_utils import build_hdf5_image_dataset

dataset_hdf5_file = Path(dataset_train_hdf5)
if not dataset_hdf5_file.exists():
    print("Creating",dataset_train_hdf5)
    build_hdf5_image_dataset(dataset_origin_train, image_shape=(128, 128), mode='folder', output_path=dataset_train_hdf5, categorical_labels=True, normalize=True)

dataset_hdf5_file = Path(dataset_test_hdf5)
if not dataset_hdf5_file.exists():
    print("Creating",dataset_test_hdf5)
    build_hdf5_image_dataset(dataset_origin_test, image_shape=(128, 128), mode='folder', output_path=dataset_test_hdf5, categorical_labels=True, normalize=True)


# Load HDF5 dataset
h5f = h5py.File(dataset_train_hdf5, 'r')
X = h5f['X']
Y = h5f['Y']

h5f = h5py.File(dataset_test_hdf5, 'r')
X_test = h5f['X']
Y_test = h5f['Y']

# Building deep neural network
input_layer = tflearn.input_data(shape=[128, 128, 3])
dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.8)
dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)
softmax = tflearn.fully_connected(dropout2, 10, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=20, validation_set=(X_test, Y_test),
          show_metric=True, run_id="dense_model")
