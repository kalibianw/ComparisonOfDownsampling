from utils import TrainModule
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import numpy as np
import shutil
import os

OUTPUT_LABEL_CNT = 10
FNAME = "strided_CNN_training_2"
CKPT_PATH = f"D:/AI/ckpt/COD/{FNAME}.ckpt"
MODEL_PATH = f"D:/AI/model/COD/{FNAME}.h5"
LOG_DIR_PATH = f"logs/{FNAME}/"
if os.path.exists(LOG_DIR_PATH):
    shutil.rmtree(LOG_DIR_PATH)
    os.makedirs(LOG_DIR_PATH)
else:
    os.makedirs(LOG_DIR_PATH)
BATCH_SIZE = 32
EPOCHS = 1000

(x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()
print(np.shape(x_train_all), np.shape(y_train_all), np.shape(x_train_all), np.shape(y_test))

x_train_all, x_test = x_train_all / 255.0, x_test / 255.0
y_train_all, y_test = to_categorical(y_train_all), to_categorical(y_test)

x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2)

print(np.shape(x_train), np.shape(x_valid), np.shape(x_test),
      np.shape(y_train), np.shape(y_valid), np.shape(y_test))

tm = TrainModule(input_shape=np.shape(x_train)[1:], output_shape=OUTPUT_LABEL_CNT,
                 ckpt_path=CKPT_PATH,
                 model_path=MODEL_PATH,
                 log_dir_path=LOG_DIR_PATH,
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS)

model = tm.create_strided_cnn_model_2()
model.summary()

tm.model_training(
    model=model,
    x_train=x_train, y_train=y_train,
    x_valid=x_valid, y_valid=y_valid
)
