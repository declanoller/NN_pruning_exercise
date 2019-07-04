import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import os

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

def weight_prune(model, prune_percent):
    print('\nPruning NN weights to {} percentile'.format(prune_percent))

    for l in model.layers:
        if 'prune_layer' in l.name:
            print('Pruning weights of layer {}...'.format(l.name))

            # get_weights()[0] are the layer weights
            w = l.get_weights()[0]

            # find the value of the prune_percent percentile of the
            # individual weight magnitudes
            prune_cutoff = np.percentile(np.abs(w), prune_percent)
            print('\tpruning all with magnitude below {:.5f}'.format(prune_cutoff))

            # Select and set all the weights with a magnitude below that value to 0
            w[np.abs(w) < prune_cutoff] = 0

            # Set the model weights to these new pruned values
            l.set_weights([w])

    return(model)


def unit_prune(model, prune_percent):
    print('\nPruning NN units to {} percentile'.format(prune_percent))

    for l in model.layers:
        if 'prune_layer' in l.name:
            print('Pruning units of layer {}...'.format(l.name))

            # get_weights()[0] are the layer weights
            w = l.get_weights()[0]

            # Find the percentile of the weight matrix *columns*
            # ranked by their L2 norm values
            prune_cutoff = np.percentile(np.linalg.norm(w, axis=0), prune_percent)
            print('\tpruning all with magnitude below {:.5f}'.format(prune_cutoff))

            # Select and set all the columns with an L2 norm below that value to 0
            w[:, np.linalg.norm(w, axis=0) < prune_cutoff] = 0

            # Set the model weights to these new pruned values
            l.set_weights([w])

    return(model)


prune_percent_list = np.array([0.0, 25, 50, 60, 70, 80, 90, 95, 97, 99], dtype=float)

weight_prune_stats = []
for prune_p in prune_percent_list:

    # Load model fresh
    p_m = keras.models.load_model(prune_model_fname)

    # Weight prune model
    p_m = weight_prune(p_m, prune_p)

    # Evaluate on test set
    test_loss, test_acc = p_m.evaluate(test_data, test_labels)

    # Record accuracy
    weight_prune_stats.append([prune_p, test_acc])


unit_prune_stats = []
for prune_p in prune_percent_list:

    # Load model fresh
    p_m = keras.models.load_model(prune_model_fname)

    # Unit prune model
    p_m = unit_prune(p_m, prune_p)

    # Evaluate on test set
    test_loss, test_acc = p_m.evaluate(test_data, test_labels)

    # Record accuracy
    unit_prune_stats.append([prune_p, test_acc])



weight_prune_stats = np.array(weight_prune_stats)
unit_prune_stats = np.array(unit_prune_stats)

plt.plot(*weight_prune_stats.transpose(), 'o-', color='dodgerblue', label='Weight pruning')
plt.plot(*unit_prune_stats.transpose(), 'o-', color='tomato', label='Unit pruning')
plt.xlabel('Percent of NN pruned')
plt.ylabel('Test accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('prune.png')
plt.show()






#
