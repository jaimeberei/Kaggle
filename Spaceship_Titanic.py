import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold, RandomizedSearchCV

train_export = pd.read_csv('C:/Users/jaime/OneDrive/Escritorio/train.csv')
test_export = pd.read_csv('C:/Users/Jaime/OneDrive/Escritorio/test.csv')
train = pd.read_csv('C:/Users/Jaime./OneDrive/Escritorio/train.csv', index_col='PassengerId')

test = pd.read_csv('C:/Users/Jaime/OneDrive/Escritorio/test.csv', index_col='PassengerId')

print(train.columns)
print(train.dtypes)
print(train.head())
print(train.isnull().sum())

print("STEP 2: Preprocessing ....")

# TRAIN
# Impute on Numerical values

train = train.drop(columns='Transported')
# y = train_export.iloc[:, -1]
y = train_export['Transported'].replace('False', 0)
y = train_export['Transported'].replace('True', 1)

imp_ord = IterativeImputer(estimator=RandomForestRegressor(),
                           initial_strategy='most_frequent',
                           max_iter=20, random_state=0)

train['Age'] = imp_ord.fit_transform(train[['Age']])
train['RoomService'] = imp_ord.fit_transform(train[['RoomService']])
train['FoodCourt'] = imp_ord.fit_transform(train[['FoodCourt']])
train['ShoppingMall'] = imp_ord.fit_transform(train[['ShoppingMall']])
train['Spa'] = imp_ord.fit_transform(train[['Spa']])
train['VRDeck'] = imp_ord.fit_transform(train[['VRDeck']])

# Impute on categorical values
imputer = SimpleImputer(strategy='most_frequent')

train['HomePlanet'] = imputer.fit_transform(train[['HomePlanet']])
train['CryoSleep'] = imputer.fit_transform(train[['CryoSleep']])
train['Cabin'] = imputer.fit_transform(train[['Cabin']])
train['Destination'] = imputer.fit_transform(train[['Destination']])
train['VIP'] = imputer.fit_transform(train[['VIP']])
train['Name'] = imputer.fit_transform(train[['Name']])

train[['Deck', 'Num', 'Side']] = pd.DataFrame(train['Cabin'].str.split('/', expand=True))
train = train.drop(columns=['Cabin', 'Name'], axis=1)

columns_trans = make_column_transformer(
    (OneHotEncoder(), ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']),
    remainder='passthrough')
train = columns_trans.fit_transform(train)
print(columns_trans.get_feature_names())

# TEST

imp_ord = IterativeImputer(estimator=RandomForestRegressor(),
                           initial_strategy='most_frequent',
                           max_iter=20, random_state=0)

test['Age'] = imp_ord.fit_transform(test[['Age']])
test['RoomService'] = imp_ord.fit_transform(test[['RoomService']])
test['FoodCourt'] = imp_ord.fit_transform(test[['FoodCourt']])
test['ShoppingMall'] = imp_ord.fit_transform(test[['ShoppingMall']])
test['Spa'] = imp_ord.fit_transform(test[['Spa']])
test['VRDeck'] = imp_ord.fit_transform(test[['VRDeck']])


# Impute on categorical values
imputer = SimpleImputer(strategy='most_frequent')

test['HomePlanet'] = imputer.fit_transform(test[['HomePlanet']])
test['CryoSleep'] = imputer.fit_transform(test[['CryoSleep']])
test['Cabin'] = imputer.fit_transform(test[['Cabin']])
test['Destination'] = imputer.fit_transform(test[['Destination']])
test['VIP'] = imputer.fit_transform(test[['VIP']])
test['Name'] = imputer.fit_transform(test[['Name']])

test[['Deck', 'Num', 'Side']] = pd.DataFrame(test['Cabin'].str.split('/', expand=True))
test = test.drop(columns=['Cabin', 'Name'], axis=1)

columns_trans = make_column_transformer(
    (OneHotEncoder(), ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']),
    remainder='passthrough')
test = columns_trans.fit_transform(test)

print("Step:3 Fitting Model ...")

X_train = train
y_train = y
X_test = test

dt_pipe = Pipeline([('clf', DecisionTreeClassifier())])
rf_pipe = Pipeline([('clf', RandomForestClassifier(n_jobs=-1))])
svm_pipe = Pipeline([('clf', SVC())])
kn_pipe = Pipeline([('clf', KNeighborsClassifier())])
mlp_pipe = Pipeline([('clf', MLPClassifier(max_iter=100))])

param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
param_range_fl = [0.005, 0.001, 0.05, 0.1, 0.5, 1]
param_range_est = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
param_range_mlp = [100, 200, 300, 400, 500, 600, 700, 800]
print(3)
grid_params_mlp = [{'clf__hidden_layer_sizes': param_range_mlp,
                    'clf__activation': ['identity', 'logistic', 'relu'],
                    'clf__solver': ['lbfgs', 'sgd', 'adam'],
                    'clf__batch_size': param_range_mlp}]
print(MLPClassifier.get_params(mlp_pipe))

grid_params_rf = [{'clf__criterion': ['gini', 'entropy'],
                   'clf__min_samples_leaf': param_range,
                   'clf__max_depth': param_range,
                   'clf__min_samples_split': param_range[1:],
                   'clf__random_state': param_range_est,
                   'clf__n_estimators': param_range_est,
                   'clf__bootstrap': ['True', 'False'],
                   'clf__oob_score': ['True', 'False'],
                   'clf__class_weight': ['balanced', 'balanced_subsample']}]

grid_params_kn = [{'clf__n_neighbors': param_range,
                   'clf__weights': ['uniform', 'distance']}]

kf = KFold(n_splits=10, shuffle=True, random_state=2022)

gs_rf = RandomizedSearchCV(estimator=rf_pipe,
                           param_distributions=grid_params_rf,
                           scoring='accuracy',
                           cv=kf, verbose=1)

gs_mlp = RandomizedSearchCV(estimator= mlp_pipe,
                            param_distributions=grid_params_mlp,
                            scoring = "accuracy",
                            cv=kf, n_jobs=-1)


grids = [gs_mlp]
#
grid_dict = {0: 'MLP'}
#
print("Training model...")
# #
best_acc = 0.0
best_clf = 0
best_gs = ''
for idx, gs in enumerate(grids):
    print('\nEstimator: %s' % grid_dict[idx])
    # Fit grid search
    gs.fit(X_train, y_train)
    # Best params
    print('Best params: %s' % gs.best_params_)
#     # Best training data accuracy
    print('Best training accuracy: %.3f' % gs.best_score_)
    # Predict on test data with best params
    y_pred = gs.predict(X_test)


print ("STEP 4: ASSESSING THE MODEL...")

passenger_id = test_export['PassengerId']
output = pd.DataFrame({'PassengerId': passenger_id, 'Transported': y_pred})
output['Transported'] = output['Transported'].apply(lambda x: True if x == 1 else False)
output.to_csv(r'C:/Users/Jaime/OneDrive/Escritorio/submission.csv', index=False)
