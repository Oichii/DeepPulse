from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, Dropout, \
    AveragePooling2D, Input, Layer
from tensorflow.keras import models, optimizers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
import os
import cv2
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import resample

np.set_printoptions(threshold=np.inf)


class NormalisedDifferenceLayer(Layer):
    def __init__(self, name="", trainable=True, dtype=tf.float64):
        super(NormalisedDifferenceLayer, self).__init__()

    def call(self, inputs):
        tf.print(inputs[0].dtype)
        normalized_difference = tf.math.divide_no_nan((inputs[1]-inputs[0]), (inputs[1]+inputs[0]))
        # normalized_difference = tf.clip_by_norm(normalized_difference, tf.keras.backend.std(normalized_difference))
        # tf.print(tf.keras.backend.min(inputs[1]+inputs[0]))

        mean = tf.keras.backend.mean(normalized_difference)
        std = tf.keras.backend.std(normalized_difference)
        normalized_difference = tf.clip_by_value(normalized_difference, mean-std, mean+std)
        # normalized_difference = tf.clip_by_value(normalized_difference, -1, 1)
        tf.print(tf.keras.backend.max(normalized_difference), tf.keras.backend.min(normalized_difference), mean)

        return normalized_difference


def frames_data_generator(sequence_list, img_height=32, img_width=32, batch_size=1):
    while True:
        np.random.shuffle(sequence_list)
        for i in sequence_list:
            file_path = 'PURE'
            print(i, 'mmmmmmmmmmmmmmmmm')
            sequence_dir = os.path.join(file_path, i)  # folder zawierający klatki sekwencji

            frames = glob.glob(sequence_dir + '/cropped/*.png')  # lista klatek sekwencji
            frames.sort()
            reference = json.load(open(sequence_dir + '.JSON', 'r'))

            ref = []
            seq = []
            seq2 = []
            for sample in reference['/FullPackage']:
                ref.append(sample['Value']['waveform'])
            ref = np.array(ref)
            ref_resample = []
            clean_ref_resample = resample(ref, len(frames))
            print(len(clean_ref_resample))
            for s in range(len(clean_ref_resample) - 1):
                ref_resample.append(clean_ref_resample[s + 1] - clean_ref_resample[s])

            for j in range(len(frames)-1):
                # print(j)
                ct = np.array(cv2.imread(frames[j]), dtype='float64') / 255
                ct = cv2.resize(ct, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

                # cv2.imwrite('sequence/ct00_{}.png'.format(j), ct)

                ct1 = np.array(cv2.imread(frames[j + 1]), dtype='float64') / 255
                ct1 = cv2.resize(ct1, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

                seq.append(ct)
                seq2.append(ct1)

            seq = np.array(seq)
            mean = seq.mean()
            std = seq.std()
            seq2 = np.array(seq2)
            mean2 = seq2.mean()
            std2 = seq2.std()
            norm_ref = np.array(ref_resample)

            norm_ref = (norm_ref - norm_ref.mean()) / norm_ref.std()
            # norm_ref = 2*((norm_ref - np.amin(norm_ref)) / (np.amax(norm_ref) - np.amin(norm_ref)))-1
            print(norm_ref.shape)

            for b in range(0, len(seq), batch_size):
                seq_out_batch = (np.array(seq[b:b + batch_size]) - mean) / std
                seq_out_batch2 = (np.array(seq2[b:b + batch_size]) - mean2) / std2
                seq_out = [seq_out_batch, seq_out_batch2]
                ref_out_batch = np.array(norm_ref[b:b + batch_size])
                print(np.amax(seq_out_batch), np.amin(seq_out_batch2))
                print(ref_out_batch)
                yield seq_out, ref_out_batch


def rotate_img(img, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(img, rot_matrix, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def augment_frames(frame1, frame2=None):
    # add noise
    if np.random.randint(0, 1):
        noise1 = np.random.normal(0, np.random.randint(0, 1), frame1.shape)
        frame1 = frame1 + noise1
        if frame2 is not None:
            noise2 = np.random.normal(0, np.random.randint(0, 1), frame2.shape)
            frame2 = frame2 + noise2
    # flip
    if np.random.randint(0, 1):
        frame1 = cv2.flip(frame1, 1)
        if frame2 is not None:
            frame2 = cv2.flip(frame2, 1)
    # rotate
    if np.random.randint(0, 1):
        frame1 = rotate_img(frame1, np.random.randint(0, 45))
        if frame2 is not None:
            frame2 = rotate_img(frame2, np.random.randint(0, 45))
    # blur
    if np.random.randint(0, 1):
        frame1 = cv2.blur(frame1, 3)
        if frame2 is not None:
            frame2 = cv2.blur(frame2, 3)
    if frame2 is not None:
        return frame1, frame2
    else:
        return frame1


def rescale_values(frame1, frame2):
    frame1 = frame1 * 3
    frame2 = frame2 * 3
    return frame1, frame2


def motion_diff_model(img_width, img_height):
    tf.keras.backend.set_floatx('float64')
    ct = Input(shape=(img_width, img_height, 3), dtype='float64')
    ct1 = Input(shape=(img_width, img_height, 3), dtype='float64')

    motion_in = NormalisedDifferenceLayer()([ct, ct1])
    motion1 = Conv2D(filters=32, kernel_size=(3, 3), input_shape=(img_height, img_width, 3), strides=(1, 1),
                     kernel_initializer='Orthogonal', padding='same', name='conv_1')(motion_in)
    motion_act1 = Activation('tanh', name='tanh_1')(motion1)
    motion2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
                     name='conv_2')(motion_act1)
    motion_act2 = Activation('tanh', name='tanh_2')(motion2)
    motion_poll1 = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None,
                                    name='average_3')(motion_act2)
    motion_dropout1 = Dropout(rate=0.25)(motion_poll1)
    motion3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
                     name='conv_4')(motion_dropout1)
    motion_act3 = Activation('tanh', name='tanh_3')(motion3)
    motion4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
                     name='conv_5')(motion_act3)
    motion_act4 = Activation('tanh', name='tanh_4')(motion4)
    motion_poll2 = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None,
                                    name='average_6')(motion_act4)
    motion_dropout2 = Dropout(rate=0.25)(motion_poll2)
    motion_flatten = Flatten(name='flatten_7')(motion_dropout2)
    motion_dense1 = Dense(128, name='fully_connected_8')(motion_flatten)
    motion_dropout3 = Dropout(rate=0.5)(motion_dense1)
    motion_dense2 = Dense(1, name='fully_connected_9')(motion_dropout3)
    out = Activation('linear', name='linear_5')(motion_dense2)
    model = models.Model(inputs=[ct, ct1], outputs=out)

    return model


if __name__ == '__main__':
    file_path = 'PURE'  # sciezka do datasetu
    set_name = 'PURE'
    img_width = 36
    img_height = 36
    save_every = 1
    nb_train_samples = 1900*40
    nb_validation_samples = 1900*19
    batch_size = 128
    epochs = 5

    save_dir = os.path.join('models', 'set_' + str(set_name))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sequence_list = [line.rstrip('\n') for line in open('sequence_all_short.txt', 'r')]
    np.random.shuffle(sequence_list)
    sequence_train_list = sequence_list[0:int(len(sequence_list)*2/3)]
    print(sequence_train_list)
    sequence_validation_list = sequence_list[int(len(sequence_list)*2/3):len(sequence_list)]
    print(sequence_validation_list)
    np.random.shuffle(sequence_train_list)

    model = motion_diff_model(img_width=img_width, img_height=img_height)

    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file='model.png')
    print(model.inputs)

    model.summary()  # wypisanie podsumowania modelu
    adam = optimizers.Adam(lr=0.0001)

    adadelta = optimizers.Adadelta(learning_rate=10**(-6), rho=0.95)
    model.compile(optimizer=adadelta, loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])

    checkpointer = ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.hdf5'), verbose=1, save_weights_only=False,
                                   period=save_every)

    csv_logger = CSVLogger(os.path.join(save_dir, 'log.csv'), append=True, separator=',')
    tensor_board = tf.keras.callbacks.TensorBoard(save_dir, histogram_freq=1)

    history_full = model.fit_generator(
        frames_data_generator(sequence_list=sequence_train_list, batch_size=batch_size, img_width=img_width, img_height=img_height),
        steps_per_epoch=nb_train_samples//batch_size,
        epochs=epochs,
        validation_data=frames_data_generator(sequence_list=sequence_validation_list, batch_size=batch_size, img_width=img_width, img_height=img_height),
        validation_steps=nb_validation_samples//batch_size,
        callbacks=[checkpointer, csv_logger])

    print(history_full.history.keys())
    plt.plot(history_full.history['mean_squared_error'])
    plt.plot(history_full.history['val_mean_squared_error'])
    plt.plot(history_full.history['mean_absolute_error'])
    plt.plot(history_full.history['val_mean_absolute_error'])
    plt.title('model metrics')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    plt.plot(history_full.history['loss'])
    plt.plot(history_full.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
