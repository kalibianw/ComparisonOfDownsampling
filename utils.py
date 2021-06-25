from tensorflow.keras import models, layers, activations, optimizers, losses, callbacks, regularizers
from sklearn.model_selection import train_test_split
from natsort import natsorted
import numpy as np
import cv2
import os


class DataModule:
    def __init__(self, data_dir_path: str):
        """
        :param data_dir_path: Path of image data dir. ex) D:/data/
        """
        self.DATA_DIR_PATH = data_dir_path

    def img_to_npz(self, npz_path: str, color_mode="rgb", dsize=(300, 300), normalized=True):
        """
        :param npz_path: Path of result npz file path. ex) test.npz
        :param color_mode: Channel of image color. rgb or gray. default: "rgb'
        :param dsize: Destination size of image. default: (300, 300)
        :param normalized: Image normalization status. default: True
        :return: None
        """
        imgs = list()
        labels = list()
        flist = natsorted(os.listdir(self.DATA_DIR_PATH))
        fcnt = len(flist)
        per = fcnt / 100
        for fidx, fname in enumerate(flist):
            if color_mode == "rgb":
                img = cv2.imread(self.DATA_DIR_PATH + fname, flags=cv2.IMREAD_COLOR)
                img = cv2.resize(img, dsize=dsize)
            elif color_mode == "gray":
                img = cv2.imread(self.DATA_DIR_PATH + fname, flags=cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, dsize=dsize)

            imgs.append(img)
            if "cat" in fname:
                labels.append(0)
            elif "dog" in fname:
                labels.append(1)
            else:
                labels.append(-1)

            if fidx % int(per) == 0:
                print(f"{fidx / int(per)}% 완료")

        if normalized is True:
            imgs = np.array(imgs, dtype=np.float16)
            imgs = imgs / 255.0
        elif normalized is False:
            imgs = np.array(imgs)
        labels = np.array(labels)

        print(np.shape(imgs), np.shape(labels))
        print(np.unique(labels, return_counts=True))
        x_train_all, x_test, y_train_all, y_test = train_test_split(imgs, labels,
                                                                    stratify=labels, test_size=0.4)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all,
                                                              stratify=y_train_all, test_size=0.2)

        print(np.shape(x_train), np.shape(x_valid), np.shape(x_test),
              np.shape(y_train), np.shape(y_valid), np.shape(y_test))

        np.savez_compressed(file=npz_path,
                            x_train=x_train, y_train=y_train,
                            x_valid=x_valid, y_valid=y_valid,
                            x_test=x_test, y_test=y_test)


class TrainModule:
    def __init__(self, input_shape, output_shape, ckpt_path, model_path, log_dir_path, batch_size, epochs):
        self.INPUT_SHAPE = input_shape
        self.OUTPUT_SHAPE = output_shape
        self.CKPT_PATH = ckpt_path
        self.MODEL_PATH = model_path
        self.LOG_DIR_PATH = log_dir_path
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs

    def create_basic_cnn_model(self):
        input_layer = layers.Input(shape=self.INPUT_SHAPE)

        conv2d_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=activations.relu,
                                 kernel_initializer="he_normal", kernel_regularizer=regularizers.L2())(input_layer)
        max_pool_1 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(conv2d_1)

        conv2d_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activations.relu,
                                 kernel_initializer="he_normal", kernel_regularizer=regularizers.L2())(max_pool_1)

        batch_normalization_1 = layers.BatchNormalization()(conv2d_2)

        max_pool_2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(batch_normalization_1)

        conv2d_3 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=activations.relu,
                                 kernel_initializer="he_normal", kernel_regularizer=regularizers.L2())(max_pool_2)
        max_pool_3 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(conv2d_3)

        conv2d_4 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=activations.relu,
                                 kernel_initializer="he_normal", kernel_regularizer=regularizers.L2())(max_pool_3)

        batch_normalization_2 = layers.BatchNormalization()(conv2d_4)

        flatten = layers.Flatten()(batch_normalization_2)

        dense_1 = layers.Dense(units=1024, activation=activations.relu, kernel_initializer="he_normal",
                               kernel_regularizer=regularizers.L2())(flatten)
        dropout_1 = layers.Dropout(rate=0.5)(dense_1)
        dense_2 = layers.Dense(units=512, activation=activations.relu, kernel_initializer="he_normal",
                               kernel_regularizer=regularizers.L2())(dropout_1)
        dropout_2 = layers.Dropout(rate=0.5)(dense_2)
        dense_3 = layers.Dense(units=128, activation=activations.relu, kernel_initializer="he_normal",
                               kernel_regularizer=regularizers.L2())(dropout_2)

        output_layer = layers.Dense(units=self.OUTPUT_SHAPE, activation=activations.softmax)(dense_3)

        model = models.Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=["acc"]
        )

        return model

    def create_strided_cnn_model(self):
        input_layer = layers.Input(shape=self.INPUT_SHAPE)

        conv2d_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding="same",
                                 activation=activations.relu, kernel_initializer="he_normal",
                                 kernel_regularizer=regularizers.L2())(input_layer)

        conv2d_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding="same",
                                 activation=activations.relu, kernel_initializer="he_normal",
                                 kernel_regularizer=regularizers.L2())(conv2d_1)

        batch_normalization_1 = layers.BatchNormalization()(conv2d_2)

        conv2d_3 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding="same",
                                 activation=activations.relu, kernel_initializer="he_normal",
                                 kernel_regularizer=regularizers.L2())(batch_normalization_1)

        conv2d_4 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same",
                                 activation=activations.relu, kernel_initializer="he_normal",
                                 kernel_regularizer=regularizers.L2())(conv2d_3)

        batch_normalization_2 = layers.BatchNormalization()(conv2d_4)

        flatten = layers.Flatten()(batch_normalization_2)

        dense_1 = layers.Dense(units=1024, activation=activations.relu, kernel_initializer="he_normal",
                               kernel_regularizer=regularizers.L2())(flatten)
        dropout_1 = layers.Dropout(rate=0.5)(dense_1)
        dense_2 = layers.Dense(units=512, activation=activations.relu, kernel_initializer="he_normal",
                               kernel_regularizer=regularizers.L2())(dropout_1)
        dropout_2 = layers.Dropout(rate=0.5)(dense_2)
        dense_3 = layers.Dense(units=128, activation=activations.relu, kernel_initializer="he_normal",
                               kernel_regularizer=regularizers.L2())(dropout_2)

        output_layer = layers.Dense(units=self.OUTPUT_SHAPE, activation=activations.softmax)(dense_3)

        model = models.Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=["acc"]
        )

        return model

    def create_concat_cnn_model(self):
        input_layer = layers.Input(shape=self.INPUT_SHAPE, name="input_layer")

        conv2d_1_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=activations.relu,
                                   kernel_initializer="he_normal", name="conv2d_1_1",
                                   kernel_regularizer=regularizers.L2())(input_layer)
        max_pool_1_1 = layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_1_1")(conv2d_1_1)

        conv2d_1_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activations.relu,
                                   kernel_initializer="he_normal", name="conv2d_1_2",
                                   kernel_regularizer=regularizers.L2())(max_pool_1_1)
        batch_normalization_1_1 = layers.BatchNormalization()(conv2d_1_2)
        max_pool_1_2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same",
                                           name="max_pool_1_2")(batch_normalization_1_1)

        conv2d_1_3 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activations.relu,
                                   kernel_initializer="he_normal", name="conv2d_1_3",
                                   kernel_regularizer=regularizers.L2())(max_pool_1_2)
        max_pool_1_3 = layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_1_3")(conv2d_1_3)

        conv2d_1_4 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=activations.relu,
                                   kernel_initializer="he_normal", name="conv2d_1_4",
                                   kernel_regularizer=regularizers.L2())(max_pool_1_3)
        batch_normalization_1_2 = layers.BatchNormalization()(conv2d_1_4)
        max_pool_1_4 = layers.MaxPooling2D(pool_size=(2, 2), padding="same",
                                           name="max_pool_1_4")(batch_normalization_1_2)

        conv2d_2_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding="same",
                                   activation=activations.relu, kernel_initializer="he_normal",
                                   name="conv2d_2_1", kernel_regularizer=regularizers.L2())(input_layer)

        conv2d_2_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding="same",
                                   activation=activations.relu, kernel_initializer="he_normal",
                                   name="conv2d_2_2", kernel_regularizer=regularizers.L2())(conv2d_2_1)

        batch_normalization_2_1 = layers.BatchNormalization()(conv2d_2_2)

        conv2d_2_3 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding="same",
                                   activation=activations.relu, kernel_initializer="he_normal",
                                   name="conv2d_2_3", kernel_regularizer=regularizers.L2())(batch_normalization_2_1)

        conv2d_2_4 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same",
                                   activation=activations.relu, kernel_initializer="he_normal",
                                   name="conv2d_2_4", kernel_regularizer=regularizers.L2())(conv2d_2_3)

        batch_normalization_2_2 = layers.BatchNormalization()(conv2d_2_4)

        concat = layers.Concatenate()([max_pool_1_4, batch_normalization_2_2])

        flatten = layers.Flatten()(concat)

        dense_1 = layers.Dense(units=1024, activation=activations.relu, kernel_initializer="he_normal",
                               name="dense_1", kernel_regularizer=regularizers.L2())(flatten)
        dropout_1 = layers.Dropout(rate=0.5)(dense_1)
        dense_2 = layers.Dense(units=512, activation=activations.relu, kernel_initializer="he_normal",
                               name="dense_2", kernel_regularizer=regularizers.L2())(dropout_1)
        dropout_2 = layers.Dropout(rate=0.5)(dense_2)
        dense_3 = layers.Dense(units=128, activation=activations.relu, kernel_initializer="he_normal",
                               name="dense_3", kernel_regularizer=regularizers.L2())(dropout_2)

        output_layer = layers.Dense(units=self.OUTPUT_SHAPE, activation=activations.softmax,
                                    name="output_layer")(dense_3)

        model = models.Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=["acc"]
        )

        return model

    def model_training(self, model, x_train, y_train, x_valid, y_valid):
        callback_list = [
            callbacks.ReduceLROnPlateau(factor=0.4, patience=5, verbose=1, min_lr=1e-6),
            callbacks.EarlyStopping(min_delta=1e-6, patience=30, verbose=1),
            callbacks.ModelCheckpoint(filepath=self.CKPT_PATH, verbose=1, save_best_only=True, save_weights_only=True),
            callbacks.TensorBoard(log_dir=self.LOG_DIR_PATH)
        ]

        hist = model.fit(
            x=x_train, y=y_train,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            callbacks=callback_list,
            validation_data=(x_valid, y_valid)
        )
        model.load_weights(filepath=self.CKPT_PATH)
        model.save(filepath=self.MODEL_PATH)

        return hist
