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
        :param npz_path: Path of result npz file. ex) test.npz
        :param color_mode: Channel of image color. rgb or gray. default: "rgb'
        :param dsize: Destination size of image. default: (300, 300)
        :param normalized: Image normalization status. default: True
        :return: None
        """
        imgs = list()
        labels = list()
        folder_list = natsorted(os.listdir(self.DATA_DIR_PATH))
        for folder_idx, folder_name in enumerate(folder_list):
            flist = natsorted(os.listdir(self.DATA_DIR_PATH + folder_name))
            fcnt = len(flist)
            per = fcnt / 100
            for file_idx, fname in enumerate(flist):
                if color_mode == "rgb":
                    img = cv2.imread(self.DATA_DIR_PATH + folder_name + "/" + fname, flags=cv2.IMREAD_COLOR)
                    img = cv2.resize(img, dsize=dsize)
                elif color_mode == "gray":
                    img = cv2.imread(self.DATA_DIR_PATH + folder_name + "/" + fname, flags=cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, dsize=dsize)

                imgs.append(img)
                labels.append(folder_idx)

                if file_idx % int(per) == 0:
                    print(f"{folder_idx}번 째 폴더 {file_idx / int(per)}% 완료")

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

        conv2d_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                 kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(input_layer)
        conv2d_1_ = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                  kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(conv2d_1)
        max_pool_1 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(conv2d_1_)

        conv2d_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                 kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(max_pool_1)
        conv2d_2_ = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                  kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(conv2d_2)
        batch_normalization_1 = layers.BatchNormalization()(conv2d_2_)
        max_pool_2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(batch_normalization_1)

        conv2d_3 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                 kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(max_pool_2)
        conv2d_3_ = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                  kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(conv2d_3)
        batch_normalization_2 = layers.BatchNormalization()(conv2d_3_)
        max_pool_3 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(batch_normalization_2)

        flatten = layers.Flatten()(max_pool_3)

        dense_1 = layers.Dense(units=1024, activation=activations.selu, kernel_initializer="he_uniform",
                               kernel_regularizer=regularizers.L2())(flatten)
        dropout_1 = layers.Dropout(rate=0.5)(dense_1)
        dense_2 = layers.Dense(units=128, activation=activations.selu, kernel_initializer="he_uniform",
                               kernel_regularizer=regularizers.L2())(dropout_1)

        output_layer = layers.Dense(units=self.OUTPUT_SHAPE, activation=activations.softmax)(dense_2)

        model = models.Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=["acc"]
        )

        return model

    def create_avg_pooling_cnn_model(self):
        input_layer = layers.Input(shape=self.INPUT_SHAPE)

        conv2d_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                 kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(input_layer)
        conv2d_1_ = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                  kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(conv2d_1)
        avg_pool_1 = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(conv2d_1_)

        conv2d_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                 kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(avg_pool_1)
        conv2d_2_ = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                  kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(conv2d_2)
        batch_normalization_1 = layers.BatchNormalization()(conv2d_2_)
        avg_pool_2 = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(batch_normalization_1)

        conv2d_3 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                 kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(avg_pool_2)
        conv2d_3_ = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                  kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(conv2d_3)
        batch_normalization_2 = layers.BatchNormalization()(conv2d_3_)
        avg_pool_3 = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(batch_normalization_2)

        flatten = layers.Flatten()(avg_pool_3)

        dense_1 = layers.Dense(units=1024, activation=activations.selu, kernel_initializer="he_uniform",
                               kernel_regularizer=regularizers.L2())(flatten)
        dropout_1 = layers.Dropout(rate=0.5)(dense_1)
        dense_2 = layers.Dense(units=128, activation=activations.selu, kernel_initializer="he_uniform",
                               kernel_regularizer=regularizers.L2())(dropout_1)

        output_layer = layers.Dense(units=self.OUTPUT_SHAPE, activation=activations.softmax)(dense_2)

        model = models.Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=["acc"]
        )

        return model

    def create_strided_cnn_model(self):
        input_layer = layers.Input(shape=self.INPUT_SHAPE)

        conv2d_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same",
                                 activation=activations.selu, kernel_initializer="he_uniform",
                                 kernel_regularizer=regularizers.L2())(input_layer)
        conv2d_1_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding="same",
                                   activation=activations.selu, kernel_initializer="he_uniform",
                                   kernel_regularizer=regularizers.L2())(conv2d_1)

        conv2d_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same",
                                 activation=activations.selu, kernel_initializer="he_uniform",
                                 kernel_regularizer=regularizers.L2())(conv2d_1_2)
        conv2d_2_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding="same",
                                   activation=activations.selu, kernel_initializer="he_uniform",
                                   kernel_regularizer=regularizers.L2())(conv2d_2)

        batch_normalization_1 = layers.BatchNormalization()(conv2d_2_2)

        conv2d_3 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same",
                                 activation=activations.selu, kernel_initializer="he_uniform",
                                 kernel_regularizer=regularizers.L2())(batch_normalization_1)
        conv2d_3_2 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same",
                                   activation=activations.selu, kernel_initializer="he_uniform",
                                   kernel_regularizer=regularizers.L2())(conv2d_3)

        batch_normalization_2 = layers.BatchNormalization()(conv2d_3_2)

        flatten = layers.Flatten()(batch_normalization_2)

        dense_1 = layers.Dense(units=1024, activation=activations.selu, kernel_initializer="he_uniform",
                               kernel_regularizer=regularizers.L2())(flatten)
        dropout_1 = layers.Dropout(rate=0.5)(dense_1)
        dense_2 = layers.Dense(units=128, activation=activations.selu, kernel_initializer="he_uniform",
                               kernel_regularizer=regularizers.L2())(dropout_1)

        output_layer = layers.Dense(units=self.OUTPUT_SHAPE, activation=activations.softmax)(dense_2)

        model = models.Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=["acc"]
        )

        return model

    def create_strided_cnn_model_2(self):
        input_layer = layers.Input(shape=self.INPUT_SHAPE)

        conv2d_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same",
                                 activation=activations.selu, kernel_initializer="he_uniform",
                                 kernel_regularizer=regularizers.L2())(input_layer)
        conv2d_1_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same",
                                   activation=activations.selu, kernel_initializer="he_uniform",
                                   kernel_regularizer=regularizers.L2())(conv2d_1)
        conv2d_1_3 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding="same",
                                   activation=activations.selu, kernel_initializer="he_uniform",
                                   kernel_regularizer=regularizers.L2())(conv2d_1_2)

        conv2d_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same",
                                 activation=activations.selu, kernel_initializer="he_uniform",
                                 kernel_regularizer=regularizers.L2())(conv2d_1_3)
        conv2d_2_1 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same",
                                   activation=activations.selu, kernel_initializer="he_uniform",
                                   kernel_regularizer=regularizers.L2())(conv2d_2)
        conv2d_2_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding="same",
                                   activation=activations.selu, kernel_initializer="he_uniform",
                                   kernel_regularizer=regularizers.L2())(conv2d_2_1)

        batch_normalization_1 = layers.BatchNormalization()(conv2d_2_2)

        conv2d_3 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same",
                                 activation=activations.selu, kernel_initializer="he_uniform",
                                 kernel_regularizer=regularizers.L2())(batch_normalization_1)
        conv2d_3_1 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same",
                                   activation=activations.selu, kernel_initializer="he_uniform",
                                   kernel_regularizer=regularizers.L2())(conv2d_3)
        conv2d_3_2 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same",
                                   activation=activations.selu, kernel_initializer="he_uniform",
                                   kernel_regularizer=regularizers.L2())(conv2d_3_1)

        batch_normalization_2 = layers.BatchNormalization()(conv2d_3_2)

        flatten = layers.Flatten()(batch_normalization_2)

        dense_1 = layers.Dense(units=1024, activation=activations.selu, kernel_initializer="he_uniform",
                               kernel_regularizer=regularizers.L2())(flatten)
        dropout_1 = layers.Dropout(rate=0.5)(dense_1)
        dense_2 = layers.Dense(units=128, activation=activations.selu, kernel_initializer="he_uniform",
                               kernel_regularizer=regularizers.L2())(dropout_1)

        output_layer = layers.Dense(units=self.OUTPUT_SHAPE, activation=activations.softmax)(dense_2)

        model = models.Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=["acc"]
        )

        return model

    def create_basic_cnn_model_2(self):
        input_layer = layers.Input(shape=self.INPUT_SHAPE)

        conv2d_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                 kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(input_layer)
        conv2d_1_ = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                  kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(conv2d_1)
        max_pool_1 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(conv2d_1_)

        conv2d_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                 kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(max_pool_1)
        conv2d_2_ = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                  kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(conv2d_2)
        batch_normalization_1 = layers.BatchNormalization()(conv2d_2_)
        max_pool_2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(batch_normalization_1)

        conv2d_3 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                 kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(max_pool_2)
        conv2d_3_ = layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                  kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(conv2d_3)
        max_pool_3 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(conv2d_3_)

        conv2d_4 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                 kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(max_pool_3)
        conv2d_4_ = layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                  kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(conv2d_4)
        batch_normalization_2 = layers.BatchNormalization()(conv2d_4_)

        flatten = layers.Flatten()(batch_normalization_2)

        dense_1 = layers.Dense(units=1024, activation=activations.selu, kernel_initializer="he_uniform",
                               kernel_regularizer=regularizers.L2())(flatten)
        dropout_1 = layers.Dropout(rate=0.5)(dense_1)
        dense_2 = layers.Dense(units=128, activation=activations.selu, kernel_initializer="he_uniform",
                               kernel_regularizer=regularizers.L2())(dropout_1)

        output_layer = layers.Dense(units=self.OUTPUT_SHAPE, activation=activations.softmax)(dense_2)

        model = models.Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=["acc"]
        )

        return model

    def create_obo_cnn_model(self):
        input_layer = layers.Input(shape=self.INPUT_SHAPE)

        conv2d_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                 kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(input_layer)
        conv2d_1_ = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                  kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(conv2d_1)
        max_pool_1 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(conv2d_1_)

        conv2d_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                 kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(max_pool_1)
        conv2d_2_ = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                  kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(conv2d_2)
        batch_normalization_1 = layers.BatchNormalization()(conv2d_2_)
        max_pool_2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(batch_normalization_1)

        conv2d_obo_1 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding="same", activation=activations.relu,
                                     kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2(),
                                     name="obo")(max_pool_2)

        conv2d_3 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                 kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(conv2d_obo_1)
        conv2d_3_ = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                  kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(conv2d_3)
        max_pool_3 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(conv2d_3_)

        conv2d_4 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                 kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(max_pool_3)
        conv2d_4_ = layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=activations.selu,
                                  kernel_initializer="he_uniform", kernel_regularizer=regularizers.L2())(conv2d_4)
        batch_normalization_2 = layers.BatchNormalization()(conv2d_4_)

        flatten = layers.Flatten()(batch_normalization_2)

        dense_1 = layers.Dense(units=1024, activation=activations.selu, kernel_initializer="he_uniform",
                               kernel_regularizer=regularizers.L2())(flatten)
        dropout_1 = layers.Dropout(rate=0.5)(dense_1)
        dense_2 = layers.Dense(units=128, activation=activations.selu, kernel_initializer="he_uniform",
                               kernel_regularizer=regularizers.L2())(dropout_1)

        output_layer = layers.Dense(units=self.OUTPUT_SHAPE, activation=activations.softmax)(dense_2)

        model = models.Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.categorical_crossentropy,
            metrics=["acc"]
        )

        return model

    def model_training(self, model, x_train, y_train, x_valid, y_valid):
        callback_list = [
            callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1, min_lr=1e-8),
            callbacks.EarlyStopping(min_delta=1e-4, patience=20, verbose=1),
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
