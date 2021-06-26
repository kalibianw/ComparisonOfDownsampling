from tensorflow.keras import models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np
import os

_, (x_test, y_test) = cifar10.load_data()
print(np.unique(y_test, return_counts=True))

x_test = x_test / 255.0
categorical_y_test = to_categorical(y_test)
print(np.shape(x_test), np.shape(categorical_y_test))

MODEL_FOLDER_PATH = "D:/AI/model/DAG/"
model_folder_list = os.listdir(MODEL_FOLDER_PATH)
fhandler = open("result.txt", 'w')
for model_folder_name in model_folder_list:
    model_list = os.listdir(MODEL_FOLDER_PATH + model_folder_name)
    for model_name in model_list:
        model = models.load_model(filepath=MODEL_FOLDER_PATH + model_folder_name + "/" + model_name)
        evaluate_result = model.evaluate(x_test, categorical_y_test, batch_size=32)
        predict_result = np.argmax(model.predict(x_test, batch_size=32, verbose=1), axis=1)
        print(f"{model_name}_{model_folder_name} predict result")
        print(f"Loss: {evaluate_result[0]}; Accuracy: {evaluate_result[1]}")
        print(f1_score(y_true=y_test, y_pred=predict_result, average="micro"))
        print(precision_score(y_true=y_test, y_pred=predict_result, average="micro"))
        print(recall_score(y_true=y_test, y_pred=predict_result, average="micro"))
        print()

        fhandler.write(f"{model_name}_{model_folder_name} predict result\n")
        fhandler.write(classification_report(y_true=y_test, y_pred=predict_result, digits=10))
        fhandler.write("\n\n")

fhandler.close()
