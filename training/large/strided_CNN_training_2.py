from utils import TrainModuleForLargeModel
from tensorflow.keras.utils import to_categorical
import numpy as np
import shutil
import os

NPZ_PATH = "D:/AI/npz/cats-dogs/normalized_cats_dogs_gray.npz"
OUTPUT_LABEL_CNT = 2
FNAME = "strided_CNN_large_training_2"
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

nploader = np.load(NPZ_PATH)
x_train, x_valid, y_train, y_valid = nploader["x_train"], nploader["x_valid"], nploader["y_train"], nploader["y_valid"]
if len(np.shape(x_train)) == 3:
    x_train, x_valid = np.expand_dims(x_train, axis=-1), np.expand_dims(x_valid, axis=-1)
elif len(np.shape(x_train)) == 4:
    pass
else:
    print("x data shape error.")
    exit()
y_train, y_valid = to_categorical(y_train), to_categorical(y_valid)

print(np.shape(x_train), np.shape(x_valid),
      np.shape(y_train), np.shape(y_valid))

tm = TrainModuleForLargeModel(input_shape=np.shape(x_train)[1:], output_shape=OUTPUT_LABEL_CNT,
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
