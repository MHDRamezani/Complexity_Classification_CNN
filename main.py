"""
Author: Mohammad Ramezani
Created: July 31, 2022
"""

import glob
import winsound
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

SENTIMENT_LABELS = ['0', '1', '2', '3']  # , '4', '5', '6', '7', '8', '9', '10', '11']
COMPLEXITY_LEVEL_NUMBER = len(SENTIMENT_LABELS)
DIRECTORY = 'D:/Business/Idea_Music/Data/Original_Data/ISMLP_PngPhotos'
EPOCH = 20
BATCH = 30
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1
SCALING_PERCENTAGE = 50

class_num = []
track_num = []
sample_num = []
greyscale_pianoroll = []


def loading_data(directory):
    directory_png = directory + '/*'
    for filename in glob.glob(directory_png):
        class_tmp = int(filename[73:75])
        track_tmp = int(filename[81:84])
        sample_tmp = int(filename[91:94])
        # print(filename, class_tmp, track_tmp, sample_tmp)

        image = cv2.imread(filename)  # IMAGE
        # print('Original Image Dimensions : ', image.shape)
        image = image[:, :, 1]        # IMAGE: Greyscale
        # print('Original Greyscale Image Dimensions : ', image.shape)
        # window_name = 'image'
        # cv2.imshow(window_name, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # save_filename = filename[:-4] + '_Greyscale.png'
        # cv2.imwrite(save_filename, image)

        # resize image to special percentage of original size
        width = int(image.shape[1] * SCALING_PERCENTAGE / 100)
        height = int(image.shape[0] * SCALING_PERCENTAGE / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        # print('Resized Image Dimensions : ', image.shape)
        # window_name = 'image'
        # cv2.imshow(window_name, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        save_filename = filename[:-4] + '_Resized.png'
        cv2.imwrite(save_filename, image)

        image = image[..., np.newaxis]

        class_num.append(class_tmp)
        track_num.append(track_tmp)
        sample_num.append(sample_tmp)
        greyscale_pianoroll.append(image)

    return class_num, track_num, sample_num, greyscale_pianoroll


def data_distribution_plot(greyscale_pianoroll, class_num):
    # Visualizing the data distribution
    data_dim = [greyscale_pianoroll[0].shape[0],
                greyscale_pianoroll[0].shape[1],
                greyscale_pianoroll[0].shape[2]]

    print('\ndata dim is:', data_dim)

    plt.close('all')
    plt.xlim([min(class_num) - 2, max(class_num) + 2])
    counts, edges, bars = plt.hist(class_num, bins=COMPLEXITY_LEVEL_NUMBER)
    plt.bar_label(bars)
    plt.title('Class Distribution')
    plt.xlabel('Complexity Level')
    plt.ylabel('Sample Count')
    plt.show()


def data_splitting(greyscale_pianoroll, class_num, test_size=TEST_SIZE, validation_size=VALIDATION_SIZE):
    X_train, X_test, Y_train, Y_test = train_test_split(greyscale_pianoroll,
                                                        class_num,
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        random_state=8)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train,
                                                      Y_train,
                                                      test_size=validation_size,
                                                      shuffle=True,
                                                      random_state=8)

    return X_train, X_test, Y_train, Y_test, X_val, Y_val


def get_CNN_model(input_shape):

    return models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(SENTIMENT_LABELS), activation='softmax')
        ]
    )


def plot_cnn_accuracy(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCH)

    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def confusion_matrix(model, X_val, Y_val, X_val_image):

    Y_pred = model.predict(X_val)
    Y_pred = tf.argmax(Y_pred, axis=-1)
    Y_pred = np.array(Y_pred)

    validation_save_path = 'D:/Business/Idea_Music/Data/Original_Data/Validation/'
    print('Number of Evaluation samples:', len(X_val_image))
    for i in range(len(X_val_image)):
        tmp = X_val_image[i]
        # window_name = 'Predicted_image'
        # cv2.imshow(window_name, tmp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        save_path = validation_save_path + \
                    '{:04d}_True_Label_{:02d}_Predicted_Label_{:02d}.png'.format(i,
                                                                                 Y_val[i],
                                                                                 Y_pred[i])
        cv2.imwrite(save_path, tmp)

    confusion_mtx = tf.math.confusion_matrix(Y_val,
                                             Y_pred,
                                             dtype=tf.dtypes.int32)

    confusion_mtx = confusion_mtx / confusion_mtx.numpy().sum(axis=1)[:, tf.newaxis]
    ax = plt.subplot()
    sns.heatmap(confusion_mtx,
                annot=True,
                ax=ax,
                xticklabels=SENTIMENT_LABELS,
                yticklabels=SENTIMENT_LABELS,
                annot_kws={"size": 15},
                fmt='.2%')

    ax.set_title('Confusion Matrix')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


if __name__ == "__main__":
    class_num, track_num, sample_num, greyscale_pianoroll = loading_data(DIRECTORY)
    print('\nNumber of total samples:', len(greyscale_pianoroll))

    data_distribution_plot(greyscale_pianoroll, class_num)

    test_size = 0.2
    validation_size = 0.25
    X_train, X_test, Y_train, Y_test, X_val, Y_val = data_splitting(greyscale_pianoroll,
                                                                    class_num,
                                                                    test_size,
                                                                    validation_size)
    print('Training:', len(X_train))
    print('Testing:', len(X_test))
    print('Validation:', len(X_val))

    X_val_image = X_val

    # CNN model
    input_shape = (X_train[0].shape[0], X_train[0].shape[1], X_train[0].shape[2])
    model = get_CNN_model(input_shape)
    print(model.summary())

    # Compiling the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    X_train = np.array(X_train).astype('float32') / 255.0
    X_test = np.array(X_test).astype('float32') / 255.0
    X_val = np.array(X_val).astype('float32') / 255.0

    Y_train = np.asarray(Y_train).astype('int')
    Y_test = np.asarray(Y_test).astype('int')
    Y_val = np.asarray(Y_val).astype('int')

    history = model.fit(X_train,
                        Y_train,
                        epochs=EPOCH,
                        batch_size=BATCH,
                        validation_data=(X_test, Y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
    print("\n%s: %.2f%%" % (model.metrics_names[1], test_acc * 100))

    plot_cnn_accuracy(history)

    # Serialize model to JSON and weights to HDF5
    model_json = model.to_json()
    with open("model-epoch-%d-batch-%d.json" % (EPOCH, BATCH), "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model-epoch-%d-batch-%d.h5" % (EPOCH, BATCH))
    print(f"\nSaved model to disk")

    # Load json and weights and create model
    json_file = open("model-epoch-%d-batch-%d.json" % (EPOCH, BATCH), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model-epoch-%d-batch-%d.h5" % (EPOCH, BATCH))
    print("Loaded model from disk")

    loaded_model.compile(optimizer='adam',
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])

    # Confusion matrix, CM (using test samples)
    confusion_matrix(loaded_model, X_val, Y_val, X_val_image)

    duration = 2000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("\nCurrent Time =", current_time)
