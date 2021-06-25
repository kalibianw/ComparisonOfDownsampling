from utils import TrainModule
from tensorflow.keras.utils import to_categorical
import numpy as np
import shutil
import sys
import os

COLOR_MODE = sys.argv[1]
OUTPUT_LABEL_CNT = 2
FNAME = f"normalized_cats_dogs_{COLOR_MODE}"
NPZ_PATH = f"npz/{FNAME}.npz"
CKPT_PATH = f"D:/AI/ckpt/DAG/{FNAME}.ckpt"
MODEL_PATH = f"D:/AI/model/DAG/{FNAME}.h5"
LOG_DIR_PATH = f"logs/{FNAME}/"
if os.path.exists(LOG_DIR_PATH):
    shutil.rmtree(LOG_DIR_PATH)
    os.makedirs(LOG_DIR_PATH)
else:
    os.makedirs(LOG_DIR_PATH)
BATCH_SIZE = 32
EPOCHS = 1000

nphandler = np.load(NPZ_PATH)
x_train, x_valid, x_test, y_train, y_valid, y_test = nphandler["x_train"], nphandler["x_valid"], nphandler["x_test"], \
                                                     nphandler["y_train"], nphandler["y_valid"], nphandler["y_test"]

if COLOR_MODE == "gray":
    x_train, x_valid, x_test = np.expand_dims(x_train, axis=-1), \
                               np.expand_dims(x_valid, axis=-1), \
                               np.expand_dims(x_test, axis=-1)

print(np.shape(x_train), np.shape(x_valid), np.shape(x_test),
      np.shape(y_train), np.shape(y_valid), np.shape(y_test))
print(np.max(x_train), np.max(x_valid), np.max(x_test),
      np.min(x_train), np.min(x_valid), np.min(x_test))
print(np.unique(y_train, return_counts=True))
print(np.unique(y_valid, return_counts=True))
print(np.unique(y_test, return_counts=True))

y_train, y_valid, y_test = to_categorical(y_train), to_categorical(y_valid), to_categorical(y_test)

tm = TrainModule(input_shape=np.shape(x_train)[1:], output_shape=OUTPUT_LABEL_CNT,
                 ckpt_path=CKPT_PATH,
                 model_path=MODEL_PATH,
                 log_dir_path=LOG_DIR_PATH,
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS)

model = tm.create_model()
model.summary()

hist = tm.model_training(
    model=model,
    x_train=x_train, y_train=y_train,
    x_valid=x_valid, y_valid=y_valid
)
