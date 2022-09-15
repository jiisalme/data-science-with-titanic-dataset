# data-science-with-titanic-dataset
This repository contains exploratory data analysis, feature engineering, imputation and model comparison parts for 
Kaggle titanic dataset. Focus in this repository is to develop a ML pipeline with scikit-learn which can process 
(create features, impute and preprocess) raw Kaggle titanic data and form predictions.

In order to run the notebooks you should first download and install Python. 
At least Python version __3.8.13__ worked fine. 

To download and install required Python packages run

__pip install -r requirements.txt__

It is recommended to go through the notebooks in the following order:

1) data_analysis.ipynb
2) feature_engineering.ipynb
3) imputation.ipynb
4) training.ipynb

If you are interested in implementation details, take a look into __utils__ folder, which contains:
- feature_engineering_utils.py
- imputation_utils.py
- training_utils.py

The remaining folders are:

- __data__ folder contains raw and transformed data files. Especially, train.csv and test.csv are important.

- __results__ folder contains the final predictions that were sent to Kaggle. 

- __pipeline__ folder contains the final pipeline that can make predictions from raw data.

Feel free to contact me in case of questions or comments!





