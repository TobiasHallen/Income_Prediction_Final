import pandas as pd
import numpy as np
from math import sqrt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool

# read in training data from csv
training = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')

# get "Income" column on its own, separate from rest of table
training_x = training.drop('Instance', axis=1)
training_x = training_x.drop('Income in EUR', axis=1)
training_y = training['Income in EUR']

# read in test data from csv, drop instance and income columns for formatting
pred_x = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
pred_x = pred_x.drop('Income', axis=1)
pred_x = pred_x.drop('Instance', axis=1)

# print(pred_x)

# create imputer for missing values with different strategies for numerical vs categorical data,
# numerical takes the mean, categorical the mode
ct = ColumnTransformer(transformers=[('cat_imp', SimpleImputer(strategy='most_frequent'), [1, 3, 5, 6, 7, 8]),
                                     ('num_imp', SimpleImputer(strategy='median'), [0, 2, 4, 9])],
                       remainder='passthrough')

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(training_x, training_y, shuffle=True, test_size=0.2)

# apply imputer to data
ct.fit(X_train, y_train)
pred_x = ct.transform(pred_x)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

# create catboost pool data structures specifying categorical features for both train and test data
pool_train = Pool(X_train, label=y_train, cat_features=[4, 5, 6, 7, 8, 9])
pool_test = Pool(X_test, label=y_test, cat_features=[4, 5, 6, 7, 8, 9])

print("Starting model creation")
# create catboostmodel
model = CatBoostRegressor(cat_features=[4, 5, 6, 7, 8, 9], eval_metric='RMSE', od_type='Iter', od_wait=10,
                          one_hot_max_size=40, task_type="GPU", devices='0', use_best_model=True,iterations=10000,
                          learning_rate=0.01, depth=10, l2_leaf_reg=3, random_strength=4, bagging_temperature=10,
                          border_count=255)
#fit model to data
model.fit(pool_train, eval_set=pool_test, use_best_model=True)

# grid search for best parameters
# params = {'depth': [6, 8, 10],
# 		  'learning_rate': [0.01,0.05,0.1],
#         'l2_leaf_reg': [2, 4, 6, 8, 10],
#         'bagging_temperature' : [0, 1, 6, 12],
#         'random_strength' : [0, 3, 6, 9]}
# 
# grid_search = model.grid_search(params, X=pool_train, verbose=2) 
# print(grid_search)

# bayesian optimization
# pds = {'depth': (6, 8),
#        'bagging_temperature': (3,10)
# 		 'learning_rate': (0.01, 1)
#       }
# optimizer = BayesianOptimization(cat_hyp, pds)
# optimizer.maximize(init_points=3, n_iter=50)
# params = optimizer.max['params']
# params['depth'] = int(params['depth'])

print("Passed Model Creation & Fitting")

# predict test data
predicted_scores = model.predict(X_test)

# compare to real data for rmse
print("Root Mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, predicted_scores)))

print(predicted_scores)
print(X_test)
print(pred_x)

# predict actual output data
output_scores = model.predict(pred_x)

# read csv file, replace income column with predicted data, write back to same file
output_file = "tcd ml 2019-20 income prediction submission file.csv"
output = pd.read_csv(output_file, header=0, index_col=False)
output["Income"] = output_scores
output.to_csv(output_file, index=False)


# bayesian optimizer function
# def cat_hyp(depth, bagging_temperature, learning_rate): 
#   params = {"iterations": 1000,
#             "learning_rate": learning_rate,
#             "eval_metric": "R2",
#             "verbose": False} # Default Parameters
#   params[ "depth"] = int(round(depth)) 
#   params["bagging_temperature"] = bagging_temperature
#    
#   scores = cgb.cv(pool_train,
#               params,
#               fold_count=5)
#   return np.min(scores['test-RMSE-mean'])*-1              
