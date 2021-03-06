from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import models
import numpy as np
import os

MODEL_FOLDER_PATH = "D:/AI/model/COD/obo"
_, (x_test, y_test) = cifar10.load_data()
x_test = x_test / 255.0

folder_name = os.path.basename(MODEL_FOLDER_PATH)
fhandler = open(f"result/report_{folder_name}.txt", 'w')
model_list = os.listdir(MODEL_FOLDER_PATH)
for model_name in model_list:
    model_result_list = list()
    model = models.load_model(MODEL_FOLDER_PATH + "/" + model_name)
    pred = model.predict(
        x=x_test,
        batch_size=32,
        verbose=1
    )
    pred_ = np.argmax(pred, axis=1)
    f1 = f1_score(y_true=y_test, y_pred=pred_, average="macro")
    precision = precision_score(y_true=y_test, y_pred=pred_, average="macro")
    recall = recall_score(y_true=y_test, y_pred=pred_, average="macro")
    model_result_list.append(model_name)
    model_result_list.append(f1)
    model_result_list.append(precision)
    model_result_list.append(recall)

    report = classification_report(y_true=y_test, y_pred=pred_)
    fhandler.write(f"{model_name}\n")
    fhandler.write(report)
    fhandler.write("\n")
fhandler.close()
