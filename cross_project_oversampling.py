# Import required libraries for performance metrics
from sklearn.metrics import accuracy_score, f1_score, fbeta_score
import sqlite3 as db
# Import required libraries for machine learning classifiers
from sklearn.neural_network import MLPClassifier
from EffortAware.calculate_metrics import *
from ProposedAlgorithm.KernelOversampling import KernelOversampling
import smote_variants as sv
from MAHAKIL.mahakil import MAHAKIL
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import utils


# Define the models evaluation function
def models_evaluation(model ,xtrain, ytrain, xtest, ytest):
    '''
    X : data set features
    y : data set target
    folds : number of cross-validation folds

    '''
    scores = []
    model.fit(xtrain, ytrain)
    prediction = model.predict(xtest)
    accuracy = accuracy_score(ytest, prediction)
    fscore = fbeta_score(ytest, prediction, beta=0.5)
    f1score = f1_score(ytest, prediction)


    # Return models performance metrics scores data frame
    return accuracy, fscore, f1score, model


# Set random seed
np.random.seed(1)

conn = db.connect('cross_project.db')
c = conn.cursor()

resampling = [
    # ('KCO', KernelOversampling()),
    # ('Borderline', sv.Borderline_SMOTE1()),
    # ('ROSE', sv.ROSE()),
    # ('ADASYN', sv.ADASYN()),
    # ('SMOTE', sv.SMOTE()),
    # ('MWMOTE', sv.MWMOTE()),
    ('MAHAKIL', MAHAKIL())
     ]

Network_model = MLPClassifier(alpha=1, max_iter=1000)


list_datasets = ['columba', 'bugzilla', 'postgres','jdt', 'platform', 'mozilla']
# list_datasets = ['jdt', 'bugzilla','platform', 'mozilla']

for training in list_datasets:
    exec(f'Performance_{training} = ' + "pd.DataFrame()")

    df_train = pd.read_csv(f'../SRC/input/{training}.csv')
    df_train.sort_values(by='commitdate', ascending=True, inplace=True, )
    df_train = df_train.reset_index(drop=True)
    df_train = df_train.drop(['transactionid', 'commitdate', 'nm', 'rexp','la','ld'], axis=1)  # drop commitId
    labels = df_train.iloc[:, -1]
    norm = StandardScaler(with_mean=False)
    features = norm.fit_transform(df_train.iloc[:, :-1])
    for name, resampler in resampling:
        if name == 'KCO' or name == 'MAHAKIL':
            X_resample, Y_resample = resampler.fit_sample(features, labels)
            X_resample, Y_resample = utils.shuffle(X_resample, Y_resample)

        else:
            X_resample, Y_resample = resampler.sample(features, labels)
            X_resample, Y_resample = utils.shuffle(X_resample, Y_resample)

        exec(f'resample_{training}_{name} = ' + f'pd.DataFrame(X_resample)')
        exec(f'resample_{training}_{name} = ' + f'resample_{training}_{name}.assign(label=Y_resample)')
        # exec(f'resample_{training}_{name}.to_sql(name="{name}{training}", con=conn)')

        resampleData = pd.DataFrame(X_resample)
        resampleData.assign(label=Y_resample)

        list_fscore = list()
        list_testData = list()
        list_f1score = list()
        list_accuracy = list()
        for testing in list_datasets:
            if testing == training:
                continue
            df_test = pd.read_csv(f'../SRC/input/{testing}.csv')
            df_test.sort_values(by='commitdate', ascending=True, inplace=True, )
            df_test = df_test.reset_index(drop=True)
            df_test = df_test.drop(['transactionid', 'commitdate', 'nm', 'rexp', 'la', 'ld'], axis=1)  # drop commitId
            targets = df_test.iloc[:, -1]
            norm = StandardScaler(with_mean=False)
            states = norm.fit_transform(df_test.iloc[:, :-1])
            accuracy, fscore, f1score, model_fitted = models_evaluation(Network_model, X_resample, Y_resample,
                                                                        states, targets)
            list_fscore.append(fscore)
            list_testData.append(testing)
            list_f1score.append(f1score)
            list_accuracy.append(accuracy)
        exec(f'Performance_{training}' + '["TestData"] = ' + f'{list_testData}')
        nameColumn = "FScore" + name
        exec(f'Performance_{training}' + '[nameColumn] = ' + f'{list_fscore}')
        nameColumn = "F1Score" + name
        exec(f'Performance_{training}' + '[nameColumn] = ' + f'{list_f1score}')
        nameColumn = "Accuracy" + name
        exec(f'Performance_{training}' + '[nameColumn] = ' + f'{list_accuracy}')
    # exec(f'Performance_{training}.to_sql(name="Performance_{training}", con=conn)')

    # new_index = [x for x in list_datasets if x != training]
    # exec (f'{training}.set_axis({new_index},axis = 0,inplace=True)')
    # c.execute(f'CREATE TABLE IF NOT EXISTS {training} (product_name text, price number)')
    # exec(f'{training}.to_sql(name={training},con = conn)')

# exec("%s = %d" % (list_datasets[1],100))

# conn = db.connect('cross_project.db')
# c = conn.cursor()
# # check if table exists
# # listOfTables = c.execute(
# #     """SELECT name FROM sqlite_master WHERE type='table'
# #     AND name='accuracy'; """).fetchall()
# #
# # if listOfTables == []:
# #     c.execute("CREATE TABLE accuracy (Fscore INTEGER PRIMARY KEY, firstname NVARCHAR(20), lastname NVARCHAR(20))")
# # else:
# #     print('Table found!')
#
conn.commit()
c.close()
conn.close()
