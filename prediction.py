import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import matplotlib
from sklearn.neural_network import MLPClassifier

font = {'family': 'normal', 'size': 14}
matplotlib.rc('font', **font)

here_path = os.path.dirname(os.path.realpath(__file__))

train_file_name = 'train_df_features_sub_standardized.csv'
train_file_path = os.path.join(here_path, train_file_name)

data = pd.read_csv(train_file_path)

y = data['churn']
X = data.drop(['customerID', 'churn'], 1)

# A function library ...................................................................................................
def find_auc_score(clf, Xin, yin, lstyle, color='g', name='LogReg', label=1, prob=1):
    '''Function to plot Receiver characteristics and find AUC'''
    if prob == 1:
        yscore = clf.predict_proba(Xin)
    else:
        yscore = clf.decision_function(Xin)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(yin, yscore[:,label],pos_label=label)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, color, linestyle=lstyle,  label='AUC' + name + ' = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle=':', color='grey')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return roc_auc


def cv_optimize(clf, parameters, X, y, n_jobs=1, n_folds=5, score_func=None):
    if score_func:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func)
    else:
        gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds)
    gs.fit(X, y)
    print("BEST", gs.best_params_, gs.best_score_, gs.grid_scores_)
    best = gs.best_estimator_
    return best


def pre_process_Xy(Xarray,yarray,test_dev_split_size=0.2):
    '''Function to split given data into test and (train, dev) set'''
    Xtr,Xdev,ytr,ydev = train_test_split(Xarray,yarray,test_size=test_dev_split_size,random_state=42,stratify=yarray)
    return Xtr,Xdev,ytr,ydev


def do_classify(clf, parameters, Xtr,ytr,Xdev,ydev, score_func='roc_auc', n_folds=5, n_jobs=2):
    if parameters:
        clf = cv_optimize(clf, parameters, Xtr, ytr, n_jobs=n_jobs, n_folds=n_folds, score_func=score_func)
    clf=clf.fit(Xtr, ytr)
    find_auc_score(clf,Xtr,ytr, lstyle='-', color='#ff6666', name="_Train", label=1, prob=1)
    find_auc_score(clf,Xdev,ydev, lstyle='--', color='#1f77b4', name="_Test", label=1, prob=1)
    return clf


def threshold_fscore(y, ypred_probability):
    '''function to find optimal threshold'''
    from sklearn.metrics import f1_score
    threshold_list = np.arange(100) / 100
    flist = []
    for threshold in threshold_list:
        pred_test_data = [1 if item > threshold else 0 for item in ypred_probability]
        f = f1_score(y, pred_test_data)
        flist.append(f)

    plt.figure()
    plt.plot(threshold_list, flist)
    plt.xlabel('Probability Threshold')
    plt.ylabel('F1 Score')
    plt.grid(linestyle=':', color='grey')
    plt.axis('equal')
    return threshold_list[np.argmax(np.asarray(flist))]
# ......................................................................................................................

Xtrain, Xdev, ytrain, ydev = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

plt.figure()
mlp = do_classify(MLPClassifier(hidden_layer_sizes=(6, 4)), None, Xtrain, ytrain, Xdev, ydev, score_func=None, n_folds=5, n_jobs=2)

ypred_probability_dev = mlp.predict_proba(Xdev)
threshold_optimal = threshold_fscore(ydev, ypred_probability_dev[:,1])

print('end')
plt.show()


