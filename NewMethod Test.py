import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from pandas.api.types import CategoricalDtype
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR


columns = ["Instance", "Year of Record", "Gender", "Age", "Country",
           "Size of City", "Profession", "University Degree", "Wears Glasses", "Hair Color",
           "Body Height [cm]", "Income in EUR"]

input_file = "tcd ml 2019-20 income prediction training (with labels).csv"
df = pd.read_csv(input_file, header=0, names=columns, na_values='#N/A')
#df = df.drop('Size of City', 1)
#df = df.drop('Gender', 1)

#df = df.dropna()
df = df.fillna(df.mode().iloc[0])
train, test = train_test_split(df, test_size=0.2)

actual_file = "tcd ml 2019-20 income prediction test (without labels).csv"
finalData = pd.read_csv(actual_file, na_values='#N/A')

#finalData = finalData.dropna()
finalData = finalData.fillna(finalData.mode().iloc[0])
finalData = finalData.drop('Instance', 1)
#finalData = finalData.drop('Size of City', 1)
#finalData = finalData.drop('Gender', 1)

final_copy = finalData.copy()
final_copy_x = final_copy.drop('Income', axis=1)
# final_copy_x = final_copy_x.drop('Body Height [cm]', axis=1)
# final_copy_x = final_copy_x.drop('Wears Glasses', axis=1)
# final_copy_x = final_copy_x.drop('Hair Color', axis=1)



class ColumnsSelector(BaseEstimator, TransformerMixin):

    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert(isinstance(X, pd.DataFrame))
        return X.select_dtypes(include=[self.dtype])


num_pipeline = Pipeline(steps=[("num_attr_selector", ColumnsSelector(np.number)), ("scaler", StandardScaler())],
                        verbose=False)


class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, strategy='most_frequent'):
        self.columns = columns
        self.strategy = strategy

    def fit(self, X, y=None):
        print(self)
        print(X.columns)
        if self.columns is None:
            self.columns = X.columns
        if self.strategy is 'most_frequent':
            self.fill = {column: X[column].value_counts().index[0] for column in self.columns}
        else:
            self.fill = {column: '0' for column in self.columns}

        return self

    def transform(self, X):
        X_copy = X.copy()
        for column in self.columns:
            X_copy[column] = X_copy[column].fillna(self.fill[column])
        return X_copy


class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, dropFirst=True):
        self.categories = dict()
        self.dropFirst = dropFirst

    def fit(self, X, y=None):
        join_df = pd.concat([train, test])
        join_df = join_df.select_dtypes(include=['object'])
        for column in join_df.columns:
            self.categories[column] = join_df[column].value_counts().index.tolist()
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.select_dtypes(include=['object'])
        for column in X_copy.columns:
            X_copy[column] = X_copy[column].astype({column: CategoricalDtype(self.categories[column])})
        return pd.get_dummies(X_copy, drop_first=self.dropFirst)


cat_pipeline = Pipeline(verbose=False, steps=[
    ("cat_attr_selector", ColumnsSelector('object')),
    ("cat_imputer", CategoricalImputer(columns=
                                       ['Gender', 'Country', 'Profession', 'University Degree', 'Hair Color'])),
    ("encoder", CategoricalEncoder(dropFirst=True))])
full_pipeline = FeatureUnion([("num_pipe", num_pipeline), ("cat_pipeline", cat_pipeline)])

train.drop('Instance', axis=1, inplace=True)
test.drop('Instance', axis=1, inplace=True)

# test = test.drop('Body Height [cm]', axis=1)
# test = test.drop('Wears Glasses', axis=1)
# test = test.drop('Hair Color', axis=1)
#
# train = train.drop('Body Height [cm]', axis=1)
# train = train.drop('Wears Glasses', axis=1)
# train = train.drop('Hair Color', axis=1)


# train.info()
# test.info()

train_copy = train.copy()
#print(np.shape(train_copy))

X_train = train_copy.drop('Income in EUR', axis=1)
Y_train = train_copy['Income in EUR']

#print(X_train)
print("Training:")
print(X_train)
print("Testing:")
print(final_copy_x)
X_train_processed = full_pipeline.fit_transform(X_train)
final_x_processed = full_pipeline.fit_transform(final_copy_x)
# print("Training:")
# print(X_train_processed)
# print("Testing:")
# print(final_x_processed)

#model = GradientBoostingRegressor(verbose=2)
# model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1, verbose=True)
# feature_importances = pd.DataFrame(model.feature_importances_, index=X_train_processed.columns,
# columns=['importance'])\
#     .sort_values('importance', ascending=False)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(feature_importances)
#
# print(feature_importances)

# params = np.linspace(10, 32, 32, endpoint=True)
# for param in params:
#model = GradientBoostingRegressor(verbose=2, n_estimators=3000, learning_rate=0.01)
# verbose=2, max_features=400, min_samples_split=1000, max_depth=32,
#                                   min_samples_leaf=50, n_estimators=100,
#                                   subsample=0.8, learning_rate=0.5
# model = RandomForestRegressor(n_estimators=2000, random_state=0, verbose=2, n_jobs=4)

param_test1 = {
 #'max_depth': [12, 24, 36],
 # 'min_child_weight': [6, 7, 8],
 #'gamma': [i/10.0 for i in range(0, 4)],
 #'subsample': [i/10.0 for i in range(7, 9)],
 #'colsample_bytree': [i/10.0 for i in range(6, 10)],
 #'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
    'learning rate': [0.01, 0.05, 0.1]
}

# gsearch1 = GridSearchCV(estimator=xgb.XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=24,
#                         min_child_weight=6, gamma=0.0, subsample=0.7, colsample_bytree=0.9,
#                         nthread=4, scale_pos_weight=1, seed=27, verbosity=0, tree_method='gpu_hist', reg_alpha=1e-5),
#                         param_grid=param_test1, n_jobs=1, iid=False, cv=5, verbose=3)
# gsearch1.fit(X_train_processed, Y_train)
# print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
model = KNeighborsRegressor()
# model = xgb.XGBRegressor(booster='gbtree', nthread=4, eta=0.1, verbosity=2, max_depth=8, n_estimators=500)
#                         min_child_weight=6, gamma=0.0, subsample=0.7, scale_pos_weight=1, colsample_bytree=0.9,
#                        seed=27, reg_alpha=1e-5)
model.fit(X_train_processed, Y_train)
test_copy = test.copy()
X_test = test_copy.drop('Income in EUR', axis=1)
Y_test = test_copy['Income in EUR']
X_test_processed = full_pipeline.fit_transform(X_test)
pred = model.predict(X_test_processed)
#print("Coefficients: ")
#print(model.coef_)
print("Root Mean squared error: %.2f"
      % sqrt(mean_squared_error(Y_test, pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, pred))
# print('Internal score: %.2f' % regr.score(X, y))
# print('Internal score: %.2f' % model.score(X_test_processed, Y_test))

results = model.predict(final_x_processed)
output_file = "tcd ml 2019-20 income prediction submission file.csv"
output = pd.read_csv(output_file, header=0, index_col=False)
output["Income"] = results
output.to_csv(output_file, index=False)

# cross_val_model = RandomForestRegressor(n_estimators=10, random_state=0, verbose=2, n_jobs=1)
# scores = cross_val_score(cross_val_model, X_train_processed, Y_train, cv=5, verbose=2)
# print(scores)
# print(np.mean(scores))
#
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300, 1000]
# }
#
# clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
# best_model = clf.fit(X_train_processed, Y_train)
# print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
# print('Best C:', best_model.best_estimator_.get_params()['C'])
#
# best_predicted_values = best_model.predict(X_test_processed)
# print(best_predicted_values)
#
# print('Variance score: %.2f' % r2_score(Y_test, best_predicted_values))





