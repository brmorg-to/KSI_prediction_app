import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def predict(X_test, y_test):
    svc = joblib.load('application/model/SVC_model_v2.pkl')
    pred = svc.predict(X_test)
    # print(classification_report(y_test, pred))

    return pred
