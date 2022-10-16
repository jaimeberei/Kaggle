import pandas as pd
import numpy as np
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix, accuracy_score, explained_variance_score, roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
import tensorflow as tf

train = pd.read_csv('C:/Users/jaime/OneDrive/Escritorio/train.csv')
test = pd.read_csv('C:/Users/jaime/OneDrive/Escritorio/test.csv')
print(train.info())
print(train.isnull().sum())

### Preprocessing

X = train.drop(columns=['Transported'])
y = train['Transported']


imp_ord = IterativeImputer(estimator=RandomForestRegressor(),
                           initial_strategy='most_frequent',
                           max_iter=20, random_state=0)


imputer = SimpleImputer(strategy='most_frequent')

for column in X:
    if X.dtypes[column] =='float64':
        X[column] = imp_ord.fit_transform(X[[column]])
    elif X.dtypes[column] ==object:
        X[column] = imputer.fit_transform(X[[column]])

for column in test:
    if test.dtypes[column] =='float64':
        test[column] = imp_ord.fit_transform(test[[column]])
    elif test.dtypes[column] ==object:
        test[column] = imputer.fit_transform(test[[column]])

X[['Deck', 'Num', 'Side']] = pd.DataFrame(X['Cabin'].str.split('/', expand=True))
X = X.drop(columns=['Cabin', 'Name'], axis=1)

columns_trans = make_column_transformer(
    (OneHotEncoder(), ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']),
    remainder='passthrough')
X = columns_trans.fit_transform(X)
test = columns_trans.fit_transform(test)

### ML ###

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Decision Tree
dt_pipe = Pipeline([('clf', DecisionTreeClassifier())])

# Random Forest
rf_pipe = Pipeline([('clf', RandomForestClassifier())])

# Support Vector Machine
svm_pipe = Pipeline([('clf', SVC())])

# Stochastic Gradient Descent:
kn_pipe = Pipeline([('clf', KNeighborsClassifier())])

# Kernel Ridge Regression:
gnb_pipe = Pipeline([('clf', GaussianNB())])


param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
param_range_fl = [0.005, 0.001, 0.05, 0.1, 0.5, 1]
param_range_est = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
param_range_mlp = [100, 200, 300, 400, 500, 600, 700, 800]

grid_params_dt = [{'clf__criterion': ['gini', 'entropy'],
                   'clf__min_samples_leaf': param_range,
                   'clf__max_depth': param_range,
                   'clf__min_samples_split': param_range[1:]}]

grid_params_rf = [{'clf__criterion': ['gini', 'entropy'],
                   'clf__min_samples_leaf': param_range,
                   'clf__max_depth': param_range,
                   'clf__min_samples_split': param_range[1:]}]

grid_params_svm = [{'clf__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                    'clf__C': param_range}]

grid_params_kn = [{'clf__n_neighbors': param_range,
                   'clf__weights': ['uniform', 'distance']}]

grid_params_gnb = [{'clf__var_smoothing': [1e-9, 1e-10,1e-5]}]


# KFold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=2022)

gs_dt = RandomizedSearchCV(estimator= dt_pipe,
                           param_distributions=grid_params_dt,
                           scoring = "accuracy",
                           cv= kf, n_jobs=-1) #n_jobs to run faster


# Radomized Search
gs_rf = RandomizedSearchCV(estimator=rf_pipe,
                           param_distributions=grid_params_rf,
                           scoring="accuracy",
                           cv=kf, n_jobs=-1)



gs_svm = RandomizedSearchCV(estimator= svm_pipe,
                            param_distributions=grid_params_svm,
                            scoring = "accuracy",
                            cv= kf, n_jobs=-1)

gs_kn = RandomizedSearchCV(estimator= kn_pipe,
                           param_distributions=grid_params_kn,
                           scoring = "accuracy",
                           cv= kf, n_jobs=-1)

gs_gnb = RandomizedSearchCV(estimator= gnb_pipe,
                            param_distributions= grid_params_gnb,
                            cv= kf, n_jobs=-1)

# Pipelines:
grids = [gs_dt, gs_rf, gs_svm, gs_kn, gs_gnb]

# Dictionary of pipelines and classifier types for reference
grid_dict = {0: 'Decision Tree', 1: 'Random Forest', 2:'Support Vector Machine',
             3: 'KNN Classifier', 4: 'Gaussian Naive-Bayes'}

# Fit the grid search
print("Training model...")

best_acc = 0.0
best_clf = 0
best_gs = ''
for idx, gs in enumerate(grids):
    print('\nEstimator: %s' % grid_dict[idx])
    # Fit grid search
    gs.fit(X_train, y_train)
    # Best params
    print('Best params: %s' % gs.best_params_)
    # Best training data accuracy
    print('Best training accuracy: %.3f' % gs.best_score_)
    # Predict on test data with best params
    y_pred = gs.predict(X_test)
    # Test data accuracy of model with best params
    print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))
    # Track best (highest test accuracy) model
    if accuracy_score(y_test, y_pred) > best_acc:
        best_acc = accuracy_score(y_test, y_pred)
        best_gs = gs
        best_clf = idx

print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])

print('classifiers trained!')







