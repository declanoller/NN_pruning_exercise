import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import os
import time

'''
For creating a smaller network, using the results of pruning a larger,
trained network.

Here, I use the 80% prune level as a benchmark, since it decreases the size significantly, while still maintaining ~85% accuracy.
'''


################## Load dataset

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)

# Normalize pixel vals
train_data = train_data/255.0
test_data = test_data/255.0

################## Load/train/save/etc

#If trained model file does not exist already, train and save it.

prune_model_fname = 'prune_model_trained.h5'

if not os.path.exists(prune_model_fname):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28), name='flatten'),
        keras.layers.Dense(1000, activation=tf.nn.relu, use_bias=False, name='prune_layer_1'),
        keras.layers.Dense(1000, activation=tf.nn.relu, use_bias=False, name='prune_layer_2'),
        keras.layers.Dense(500, activation=tf.nn.relu, use_bias=False, name='prune_layer_3'),
        keras.layers.Dense(200, activation=tf.nn.relu, use_bias=False, name='prune_layer_4'),
        keras.layers.Dense(10, activation=tf.nn.softmax, use_bias=False, name='output')
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

    model.fit(train_data, train_labels, epochs=5)

    model.save(prune_model_fname)
    model.summary()
    del model


################## Weight pruning functions


def get_unit_pruned_NN(full_model, prune_percent):
    print('\nPruning NN units to {} percentile'.format(prune_percent))

    # Create the smaller model
    small_model = keras.Sequential()
    small_model.add(keras.layers.Flatten(input_shape=(28, 28),name='flatten'))

    layer_weights_dict = {}
    keep_columns = None
    for l in full_model.layers:
        if 'prune_layer' in l.name:
            print('\n\nPruning units of layer {}...'.format(l.name))

            # get_weights()[0] are the layer weights
            w = l.get_weights()[0]

            # Copy because we'll need to use w after changing w_small
            w_small = w

            # Find the percentile of the weight matrix *columns*
            # ranked by their L2 norm values
            prune_cutoff = np.percentile(np.linalg.norm(w, axis=0), prune_percent)
            print('\tpruning all with magnitude below {:.5f}'.format(prune_cutoff))

            # If it's not the first layer matrix, then only keep the rows that
            # correspond to the columns kept in the previous layer iteration
            if keep_columns is not None:
                w_small = w_small[keep_columns, :]

            # Select all columns to keep (above prune_cutoff)
            keep_columns = np.where(np.linalg.norm(w, axis=0) >= prune_cutoff)[0]

            # Only keep these columns
            w_small = w_small[:,keep_columns]

            small_model.add(keras.layers.Dense(w_small.shape[1], activation=tf.nn.relu, use_bias=False, name=l.name))

            # Save the weights for this layer to layer_weights_dict
            layer_weights_dict[l.name] = w_small

        if 'output' in l.name:
            # This is for the final layer. We don't prune it, but we do want
            # to copy the weight matrix.
            w = l.get_weights()[0]
            w_small = w
            w_small = w_small[keep_columns, :]
            layer_weights_dict[l.name] = w_small

    # Add the final output layer
    small_model.add(keras.layers.Dense(10, activation=tf.nn.softmax, use_bias=False, name='output'))

    # Compile, no need to train
    small_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

    # Set the weights from layer_weights_dict
    for l in small_model.layers:
        if l.name in layer_weights_dict.keys():
            l.set_weights([layer_weights_dict[l.name]])

    return(small_model)


# Load model fresh
full_model = keras.models.load_model(prune_model_fname)
full_model.summary()
full_model.save('full_model.h5')
# Get smaller, unit pruned model
small_model = get_unit_pruned_NN(full_model, 80.0)
small_model.summary()
small_model.save('small_model.h5')
start = time.time()
test_loss, test_acc = full_model.evaluate(test_data, test_labels)
print('took {} seconds to evaluate test data for full_model'.format(time.time() - start))
print('test acc for full_model:', test_acc)

start = time.time()
test_loss, test_acc = small_model.evaluate(test_data, test_labels)
print('took {} seconds to evaluate test data for small_model'.format(time.time() - start))
print('test acc for small_model:', test_acc)






#
