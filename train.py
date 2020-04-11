import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
from keras.callbacks import TensorBoard

from simplified_unet import *
from data import *

# import tensorflow as tf
# import keras.backend.tensorflow_backend as K

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=False
# sess = tf.Session(config=config)
# K.set_session(sess)

def train(train_path, image_folder, label_folder,
          valid_path, valid_image_folder, valid_label_folder,
          flag_multi_class, num_classes,
          pretrained_model, batch_size, epoch):
    dp = data_preprocess(train_path=train_path,
                         image_folder=image_folder,
                         label_folder=label_folder,
                         valid_path=valid_path,
                         valid_image_folder=valid_image_folder,
                         valid_label_folder=valid_label_folder,
                         flag_multi_class=flag_multi_class,
                         num_classes=num_classes)

    # train your own model
    train_data = dp.trainGenerator(batch_size=batch_size)  # change batch_size and target_size in dp according to your gpu memory
    valid_data = dp.validLoad(batch_size=batch_size)
    test_data = dp.testGenerator()

    model = unet(num_class=num_classes)
    model.load_weights(pretrained_model)  # train model start over should remove this line

    tb_cb = TensorBoard(log_dir=log_filepath)
    model_checkpoint = [tb_cb,
                        keras.callbacks.ModelCheckpoint(r'./model/model_{epoch}_{val_accuracy}.hdf5',  # lower keras
                                                        monitor='val_accuracy',
                                                        # should replace val_accuracy as val_acc
                                                        verbose=1,
                                                        save_best_only=True,
                                                        mode='max')]
    # model_checkpoint = keras.callbacks.ModelCheckpoint(r'./model/model_{epoch}_{val_binary_PTA}.hdf5',
    #                                                    monitor='val_binary_PTA',
    #                                                    verbose=1,
    #                                                    save_best_only=True,
    #                                                    mode='max')

    model.fit_generator(train_data,
                        steps_per_epoch=len(os.listdir(os.path.join(train_path, image_folder))),
                        epochs=epoch,
                        validation_steps=len(os.listdir(os.path.join(valid_path, valid_image_folder))),
                        validation_data=valid_data,
                        callbacks=model_checkpoint)


if __name__ == '__main__':
    # path to images which are prepared to train a model
    train_path = "data"
    image_folder = "resize_train_img"
    label_folder = "resize_train_label"
    valid_path = "data"
    valid_image_folder = "resize_valid_img"
    valid_label_folder = "resize_valid_label"
    log_filepath = './log'
    flag_multi_class = False
    num_classes = 2
    pretrained_model = r'model/model_17_0.9887387033492799.hdf5'
    batch_size = 1
    epoch = 30
    train(train_path, image_folder, label_folder,
          valid_path, valid_image_folder, valid_label_folder,
          flag_multi_class, num_classes,
          pretrained_model, batch_size, epoch)
