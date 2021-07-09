from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import models
import numpy as np
import os

MODEL_FOLDER_PATH = "D:/AI/model/COD/"
_, (x_test, y_test) = cifar10.load_data()
x_test = x_test / 255.0

folder_list = os.listdir(MODEL_FOLDER_PATH)
result_list = list()
fhandler = open(f"report.txt", 'w')
for folder_name in folder_list:
    model_list = os.listdir(MODEL_FOLDER_PATH + folder_name)
    for model_name in model_list:
        model_result_list = list()
        model = models.load_model(MODEL_FOLDER_PATH + folder_name + "/" + model_name)
        pred = model.predict(
            x=x_test,
            batch_size=32,
            verbose=1
        )
        pred_ = np.argmax(pred, axis=1)
        f1 = f1_score(y_true=y_test, y_pred=pred_, average="micro")
        precision = precision_score(y_true=y_test, y_pred=pred_, average="micro")
        recall = recall_score(y_true=y_test, y_pred=pred_, average="micro")
        model_result_list.append(folder_name)
        model_result_list.append(model_name)
        model_result_list.append(f1)
        model_result_list.append(precision)
        model_result_list.append(recall)
        result_list.append(model_result_list)

        report = classification_report(y_true=y_test, y_pred=pred_)
        fhandler.write(f"{folder_name} - {model_name}\n")
        fhandler.write(report)
        fhandler.write("\n")
fhandler.close()
result_list = np.array(result_list)
print(result_list)
print(np.shape(result_list))
