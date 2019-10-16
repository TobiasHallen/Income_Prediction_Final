# Income Prediction 

Contains a solution for the TCD ML Comp. 2019/20 - Income Prediction (Ind.) competition on Kaggle.

Implemented a CatBoost model, which has support for the type of categorical data some of the columns contained. Hyperparameters may or may not be equal to what they were when the best result was achieved, used gridsearch and bayesian optimization to find reasonable parameters before hand-testing, but leaving them in the codebase commented out would have been messy. This runs well out of the box, if all libs are installed and one has the training, test and submission files.   
Also added another .py script containing a large amount of the testing work that went into this assignment, as the finished result doesn't really tell the whole story. This script is not properly commented and very messy.

Best Root mean squared error: ~60k

