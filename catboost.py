import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from catboost import CatBoostRegressor, Pool

training = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')

X = training.drop('Instance', axis=1)
X = X.drop('Income in EUR', axis=1)
y = training['Income in EUR']

pred_x = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
pred_x = pred_x.drop('Income', axis=1)
pred_x = pred_x.drop('Instance', axis=1)

print(pred_x)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.1)

ct = ColumnTransformer(transformers=[('num_imp', SimpleImputer(strategy='median'), [0, 2, 4, 9]), ('cat_imp', SimpleImputer(strategy='most_frequent'), [1, 3, 5, 6, 7, 8])], remainder='passthrough')

ct.fit(X_train, y_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)
pred_x = ct.transform(pred_x)

pool_train = Pool(X_train, label=y_train, cat_features = [4, 5, 6, 7, 8, 9])

pool_test = Pool(X_test, label=y_test, cat_features = [4, 5, 6, 7, 8, 9])

# pred_pool = Pool(X_pred, label=y_pred, cat_features = [4, 5, 6, 7, 8, 9, 10])

print("Starting model creation")
model = CatBoostRegressor(cat_features= [4, 5, 6, 7, 8, 9], eval_metric='RMSE', od_type='Iter', od_wait=10,
                          one_hot_max_size=40, task_type="GPU", devices='0', use_best_model=True,iterations=10000,
                          learning_rate=0.01, depth=10, l2_leaf_reg=3, random_strength=4, bagging_temperature=10,
                          border_count=255)
model.fit(pool_train, eval_set=pool_test, use_best_model=True)

print("Passed Model Creation")


predicted_scores = model.predict(X_test)

print("Root Mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, predicted_scores)))

print(predicted_scores)
print(X_test)
print(pred_x)

output_scores = model.predict(pred_x)

output_file = "tcd ml 2019-20 income prediction submission file.csv"
output = pd.read_csv(output_file, header=0, index_col=False)
output["Income"] = output_scores
output.to_csv(output_file, index=False)

