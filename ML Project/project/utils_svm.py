import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, make_scorer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy.linalg import norm
import math
import matplotlib.pyplot as plt

#monk_path = 'ML-project22_gsp/monk/monks-'
#usare path di sotto per seguire il path della cartella monk nella cartella SVM_team_gsp
monk_path = 'monk/monks-'

#FUNZIONE UTILITY: il path del monk è preso dalla variabile globale 'monk_path'
def monk_create_df(num, train=True):
    data_type = "train" if train else "test"
    path = monk_path+str(num)+'.'+data_type

    columns = ["id", "output", "a1", "a2", "a3", "a4", "a5", "a6", "monk_id"]
    # skipping first 7 rows as they are comments and not actual data
    df = pd.read_csv(path, names=columns, delimiter=" ")

    df = df.drop('id', axis='columns')
    df = df.drop("monk_id", axis='columns')
    return pd.get_dummies(df, columns=df.columns[1:])


def get_xy_test_by_monk(monk_n):   
    df_test = monk_create_df(monk_n, train=False)

    X_test = df_test.drop(columns=['output']) #MONK.TEST
    y_test = df_test['output']                #MONK.TEST
    return X_test,y_test