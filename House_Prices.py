import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor


def fill_na(dataframe):

    imputer_ordinal = IterativeImputer(estimator=RandomForestRegressor(), initial_strategy='most_frequent', max_iter=20,
                                       random_state=0)

    imputer = SimpleImputer(strategy='most_frequent')

    for column in dataframe.columns:
        if dataframe[column].isnull().any():
            if dataframe.dtypes[column] == 'float64' or dataframe.dtypes[column] == 'int64':
                dataframe[column] = imputer_ordinal.fit_transform(dataframe[[column]])
            else:
                dataframe[column] = imputer.fit_transform(dataframe[[column]])

    return dataframe


def list_encoder(dataframe):

    l_ohe = []
    for column in dataframe.columns:
        if dataframe.dtypes[column] == 'float64' or dataframe.dtypes[column] == 'int64':
            pass
        else:
             l_ohe.append(column)

    return l_ohe


def encoder(dataframe, columns):

    cat_tranformer = Pipeline(
        steps=[('cat_cat', OrdinalEncoder(),),])
    transformer = ColumnTransformer(
        transformers=[('cat_transformer', cat_tranformer, columns)])
    transformer_pipeline = Pipeline(steps=[('tranformer', transformer)])
    df = pd.DataFrame(transformer_pipeline.fit_transform(dataframe), columns=columns)
    dataframe = dataframe.iloc[:, ~dataframe.columns.isin(columns)]
    dataframe = pd.concat([df, dataframe], axis=1)

    return dataframe


test = pd.read_csv('C:/Users/jaime/OneDrive/Escritorio/Kaggle/test.csv', index_col=None)
test = fill_na(test)


train = pd.read_csv('C:/Users/jaime/OneDrive/Escritorio/Kaggle/train.csv', index_col=None)
train = fill_na(train)

encoder_list = list_encoder(test)

test = encoder(test, encoder_list)
train = encoder(train, encoder_list)

y_train = train['SalePrice'].copy()
X_train = train.drop(columns={'SalePrice'})
X_test = test.copy()

X_train = StandardScaler().fit_transform(X_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)

dt_pipe = Pipeline([('clf', DecisionTreeRegressor())])
rf_pipe = Pipeline([('clf', RandomForestRegressor(n_jobs=-1, random_state=0))])
mlp_pipe = Pipeline([('clf', MLPRegressor(max_iter=100, random_state=0))])

param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
param_range_est = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
param_range_mlp = [100, 200, 300, 400, 500, 600, 700, 800]

grid_params_dt = [{'clf__splitter': ['best', 'random'],
                    'clf__max_features': ['auto', 'sqrt', 'log2'],
                    'clf__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']}]

grid_params_mlp = [{'clf__hidden_layer_sizes': param_range_mlp,
                    'clf__activation': ['identity', 'logistic', 'relu', 'tanh'],
                    'clf__solver': ['lbfgs', 'sgd', 'adam'],
                    'clf__batch_size': param_range_mlp}]

grid_params_rf = [{'clf__criterion': ['squared_error', 'absolute_error', 'poisson'],
                   'clf__min_samples_leaf': param_range,
                   'clf__max_depth': param_range,
                   'clf__min_samples_split': param_range[1:],
                   'clf__random_state': param_range_est,
                   'clf__n_estimators': param_range_est}]


kf = KFold(n_splits=10, shuffle=True, random_state=1)

gs_rf = RandomizedSearchCV(estimator=rf_pipe,
                           param_distributions=grid_params_rf,
                           scoring='neg_mean_absolute_error',
                           cv=kf, verbose=1)
#
gs_mlp = RandomizedSearchCV(estimator= mlp_pipe,
                            param_distributions=grid_params_mlp,
                            scoring = 'neg_mean_absolute_error',
                            cv=kf, n_jobs=-1)

gs_dt = RandomizedSearchCV(estimator= dt_pipe,
                            param_distributions=grid_params_dt,
                            scoring = 'neg_mean_absolute_error',
                            cv=kf, n_jobs=-1)

grids = [gs_rf, gs_mlp, gs_dt]
#
grid_dict = {0: 'Random Forest', 1: 'Multi Layer Perceptor', 2: 'Decision Tree'}
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
#     # Lowest training data error
    print('Best training accuracy: %.3f' % gs.best_score_)
    # Predict on test data with best params
    y_pred = gs.predict(X_test)
    print('Test set mean squared error for best params: %.3f ' % mean_squared_error(y_test, y_pred))
    # Track lowest (error) model
    if mean_squared_error(y_test, y_pred) < best_acc:
        best_acc = mean_squared_error(y_test, y_pred)
        best_gs = gs
        best_clf = idx

id = test['Id']
# predictions = round(prediction_test)
output = pd.DataFrame({'Id': id, 'SalePrice': y_pred})
output.to_csv(r'C:/Users/Jaime/OneDrive/Escritorio/submission.csv', index=False)
