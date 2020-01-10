from keras.models import *
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
import keras
import itertools
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('--val_f1: %.4f --val_precision: %.4f --val_recall: %.4f' % (_val_f1, _val_precision, _val_recall))
        # return _val_f1, _val_precision, _val_recall
        return


def loss(y_true, y_pred):
    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 2)


def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    TN = K.sum((1 - y_pred) * (1 - y_true))
    return TN / N


def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP / P


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def unet(pretrained_weights=None, input_size=(None, None, 3), num_class=2):
    inputs = Input(input_size)
    conv1 = Conv2D(32, (7, 1), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

    conv1 = BatchNormalization(momentum=0.9)(conv1)
    conv1 = Conv2D(32, (1, 7), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BatchNormalization(momentum=0.9)(pool1)
    conv2 = Conv2D(64, (5, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization(momentum=0.9)(conv2)
    conv2 = Conv2D(64, (1, 5), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BatchNormalization(momentum=0.9)(pool2)
    conv3 = Conv2D(256, (5, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization(momentum=0.9)(conv3)
    conv3 = Conv2D(256, (1, 5), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = AveragePooling2D(pool_size=(2, 2))(drop3)

    # conv4 = BatchNormalization(momentum=0.9)(pool3)
    # conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # conv4 = BatchNormalization(momentum=0.9)(conv4)
    # conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = BatchNormalization(momentum=0.9)(pool3)
    conv5 = Conv2D(512, (5, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization(momentum=0.9)(conv5)
    conv5 = Conv2D(512, (1, 5), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))

    merge6 = concatenate([drop3, up6], axis=3)
    conv6 = BatchNormalization(momentum=0.9)(merge6)
    conv6 = Conv2D(256, (5, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization(momentum=0.9)(conv6)
    conv6 = Conv2D(256, (5, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv2, up7], axis=3)
    conv7 = BatchNormalization(momentum=0.9)(merge7)
    conv7 = Conv2D(64, (5, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization(momentum=0.9)(conv7)
    conv7 = Conv2D(64, (1, 5), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization(momentum=0.9)(conv7)

    up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv1, up8], axis=3)
    conv8 = BatchNormalization(momentum=0.9)(merge8)
    conv8 = Conv2D(32, (7, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization(momentum=0.9)(conv8)
    conv8 = Conv2D(32, (1, 7), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization(momentum=0.9)(conv8)
    # up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(conv8))
    # merge9 = concatenate([conv1, up9], axis=3)
    # conv9 = BatchNormalization(momentum=0.9)(merge9)
    # conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv9 = BatchNormalization(momentum=0.9)(conv9)
    # conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv9 = BatchNormalization(momentum=0.9)(conv9)
    # conv9 = Conv2D(num_class, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv9 = BatchNormalization(momentum=0.9)(conv9)
    if num_class == 2:
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv8)
        loss_function = 'binary_crossentropy'
    else:
        conv10 = Conv2D(num_class, 1, activation='relu')(conv8)
        loss_function = 'mse'

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss=loss_function, metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss=loss, metrics=[binary_PTA, binary_PFA])
    model.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model
unet()