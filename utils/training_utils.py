import numpy as np
import pandas as pd

import os

from ortools.linear_solver import pywraplp

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, log_loss
from sklearn.calibration import CalibrationDisplay

from joblib import dump, load

def show_results(best_pipeline, search, X_train, y_train, X_test, y_test, visualize=False):
    best_params = search.best_params_
    print('############ Parameters ################\n')
    print("Best parameters: \n")
    for key, val in best_params.items():
        print(f"{key}: {val}")
    print("\n")

    def print_metrics(pipeline, X, y):
        accuracy = pipeline.score(X, y)
        y_pred = pipeline.predict(X)
        predict_proba = pipeline.predict_proba(X)
        logloss = log_loss(y, predict_proba)
        precision, recall, fscore, _ = precision_recall_fscore_support(y, y_pred, pos_label=1,
                                                                       average='binary')
        confusions = pd.DataFrame(confusion_matrix(y, y_pred), index=['True_0', 'True_1'],
                                  columns=['Pred_0', 'Pred_1'])

        print(f"Accuracy: {accuracy}\n")
        print(f"Log loss: {logloss}\n")
        print(f"Precision: {precision}, Recall: {recall}, F-score: {fscore}\n")
        print(f"Confusion matrix: \n\n {confusions}\n")

    def visualize_metrics(pipeline, X, y):
        # plot ROC and calibration curves
        predict_proba = pipeline.predict_proba(X)
        fig, ax = plt.subplots(2, 1, figsize=(15, 10))
        fpr, tpr, thresholds = roc_curve(y_test, predict_proba[:, 1], pos_label=1)
        sns.lineplot(x=fpr, y=tpr, ax=ax[0])
        sns.lineplot(x=[0, 1], y=[0, 1], linestyle='--', color='gray', ax=ax[0])
        ax[0].set_ylabel('True positive rate')
        ax[0].set_xlabel('False positive rate')
        disp = CalibrationDisplay.from_predictions(y, predict_proba[:, 1], n_bins=10, ax=ax[1])

    print(f"Best accuracy in validation sets: {search.best_score_}\n")
    print('############ Results in training set (refit after cv) ################\n')
    print_metrics(best_pipeline, X_train, y_train)

    print('############ Results in test set ################\n')
    print_metrics(best_pipeline, X_test, y_test)
    if visualize:
        visualize_metrics(best_pipeline, X_test, y_test)


def save_pipeline(estimator, file_path):
    os.path.join('./models', 'pipeline.joblib')
    dump(estimator, file_path)
    print(f'Pipeline saved to {file_path}!')



def get_failed_predictions(estimator, X, y):
    y_pred = estimator.predict(X)
    return X[y!=y_pred]

def show_slice_results(estimator, X, y, groupby_cols):
    data = X.copy(deep=True)
    y_pred = estimator.predict(X)
    data['Predictions'] = y_pred
    failed_data = get_failed_predictions(estimator, X, y)
    data['Failure'] = 0
    data.loc[failed_data.index, 'Failure'] = 1

    groups = data.groupby(groupby_cols)
    def mean_poscount_count(x):
        return np.round(pd.Series.mean(x), 2), pd.Series.mean(x)*pd.Series.count(x), pd.Series.count(x)

    results = groups[['Survived', 'Predictions', 'Failure']].agg(mean_poscount_count)

    display(results)

def show_group_predictions(estimator, X, groupby_cols):
    data = X.copy(deep=True)
    y_pred = estimator.predict(X)
    data['Predictions'] = y_pred
    groups = data.groupby(groupby_cols)
    def mean_poscount_count(x):
        return np.round(pd.Series.mean(x), 2), pd.Series.mean(x)*pd.Series.count(x), pd.Series.count(x)

    results = groups[['Predictions']].agg(mean_poscount_count)

    display(results)


def create_data_model(target_data, source_data, test_data_size, groupby_cols, print_output=False):
    target_groups = target_data.groupby(groupby_cols)
    desired_group_sizes = np.round(target_groups['Name'].count() / target_data.shape[0] * test_data_size, 2)
    target_data_keys, target_data_counts = zip(*desired_group_sizes.items())

    source_data_row_keys = {index: tuple(row[groupby_col] for groupby_col in groupby_cols) for index, row in
                            source_data.iterrows()}
    source_data_key_matrix = pd.DataFrame(index=source_data.index, columns=target_data_keys)
    for index, key in source_data_row_keys.items():
        source_data_key_matrix.loc[index, :] = [int(elem == key) for elem in target_data_keys]

    K = len(target_data_keys)
    N = source_data.shape[0]

    if print_output:
        print(f'Number of keys in target data: {K}')
        print(f'Rows to choose from: {N}')
    # Importance weights of keys
    weights = np.ones(K)

    data = {}

    data['num_train_samples'] = N
    data['num_keys'] = K

    num_vars = 2 * K + N
    data['num_vars'] = num_vars

    num_constraints = 2 * K + 2
    data['num_constraints'] = num_constraints

    # Form constraint coeff matrix (Matrix A in matrix form constraints Ax <= b)
    constraint_coeffs = np.zeros((num_constraints, num_vars))
    for k, key in enumerate(target_data_keys):
        constraint_coeffs[k, :N] = weights[k] * source_data_key_matrix.loc[:, [key]].values.flatten()
        constraint_coeffs[k, N + k] = 1
        constraint_coeffs[k, N + K + k] = -1
        constraint_coeffs[K + k, :N] = -weights[k] * source_data_key_matrix.loc[:, [key]].values.flatten()
        constraint_coeffs[K + k, N + k] = -1
        constraint_coeffs[K + k, N + K + k] = 1
    constraint_coeffs[2 * K, :N] = 1
    constraint_coeffs[2 * K + 1, :N] = -1
    data['constraint_coeffs'] = constraint_coeffs

    bounds = np.zeros(2 * K + 2)
    bounds[:K] = weights * np.array(target_data_counts)
    bounds[K:2 * K] = -weights * np.array(target_data_counts)
    bounds[2 * K] = test_data_size
    bounds[2 * K + 1] = -test_data_size
    data['bounds'] = bounds
    # Objective function coefficients (vector c in objective function cx)
    objective_coeffs = [0 for _ in range(N)] + [1 for _ in range(2 * K)]
    data['obj_coeffs'] = objective_coeffs
    return data


def create_test_set(target_data, source_data, test_data_size, groupby_cols, print_output=False):
    data = create_data_model(target_data, source_data, test_data_size, groupby_cols)

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print('NO solver')

    infinity = solver.infinity()
    x = {}
    for j in range(data['num_vars']):
        if j < data['num_train_samples']:
            x[j] = solver.IntVar(0, 1, 'x[%i]' % j)
        elif data['num_train_samples'] <= j < data['num_train_samples'] + data['num_keys']:
            x[j] = solver.NumVar(0, infinity, 'e_plus[%i]' % (j - data['num_train_samples']))
        else:
            x[j] = solver.NumVar(0, infinity, 'e_minus[%i]' % (j - data['num_train_samples'] - data['num_keys']))



    for i in range(data['num_constraints']):
        constraint = solver.RowConstraint(-infinity, data['bounds'][i], '')
        for j in range(data['num_vars']):
            constraint.SetCoefficient(x[j], data['constraint_coeffs'][i, j])


    objective = solver.Objective()
    for j in range(data['num_vars']):
        objective.SetCoefficient(x[j], data['obj_coeffs'][j])
    objective.SetMinimization()

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        if print_output:
            print('Number of variables =', solver.NumVariables())
            print('Number of constraints =', solver.NumConstraints())
            print('Objective value =', solver.Objective().Value())

            print('Problem solved in %f milliseconds' % solver.wall_time())
            print('Problem solved in %d iterations' % solver.iterations())
            print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

        train_indices = [i for i in range(data['num_train_samples'])
                         if x[i].solution_value() == 0.0]
        test_indices = [i for i in range(data['num_train_samples'])
                        if x[i].solution_value() == 1.0]

        return np.array(train_indices), np.array(test_indices)
    else:
        print('The problem does not have an optimal solution.')
        return


class CustomCVSplitter:
    def __init__(self, n_splits, val_data_size, target_data, groupby_cols):
        self.n_splits = n_splits
        self.val_data_size = val_data_size
        self.target_data = target_data
        self.groupby_cols = groupby_cols

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y, groups):
        assert self.val_data_size <= 1 / self.n_splits, "Forbidden input values: val_data_size > 1/n_splits"

        stratified_splits = StratifiedKFold(n_splits=self.n_splits).split(X, y)
        new_splits = (self.form_optimized_split(X, train, val) for train, val in stratified_splits)
        return new_splits

    def form_optimized_split(self, X, train_indices, val_indices):
        source_data = X.iloc[val_indices]
        val_set_size = int(self.val_data_size * X.shape[0])
        source_train_indices, source_val_indices = create_test_set(self.target_data, source_data,
                                                                   val_set_size, self.groupby_cols)
        split_train_indices = np.append(train_indices, val_indices[source_train_indices])
        split_val_indices = val_indices[source_val_indices]

        return split_train_indices, split_val_indices

class GenderClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.array([1 if sex=='female' else 0 for sex in X['Sex']])


class GroupByClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, groupby_cols):
        self.groupby_cols = groupby_cols

    def fit(self, X, y):
        self.group_pred = np.round(X.groupby(self.groupby_cols)['Survived'].mean())
        return self

    def predict(self, X):
        if len(self.groupby_cols) > 1:
            pred = [self.group_pred.get(tuple(row[col] for col in self.groupby_cols), 0) for _, row in X.iterrows()]
        else:
            pred = [self.group_pred.get(row[self.groupby_cols[0]], 0) for _, row in X.iterrows()]
        return np.array(pred)

class ModifiedGroupByClassifier(GroupByClassifier):
    def __init__(self, groupby_cols, special_group_clfs, special_group_features):
        super().__init__(groupby_cols)
        self.special_group_clfs = special_group_clfs
        self.special_group_features = special_group_features

    def fit(self, X, y):
        super().fit(X, y)
        for group in self.special_group_clfs:
            indices = self.form_group_indices(X, group)
            X_group, y_group = X[indices].loc[:, self.special_group_features[group]], y[indices]

            if not X_group.empty:
                self.special_group_clfs[group].fit(X_group, y_group)
            else:
                print('X_group empty')

    def predict(self, X):
        groupby_pred = super().predict(X)
        for group in self.special_group_clfs:
            indices = self.form_group_indices(X, group)
            X_group = X[indices].loc[:, self.special_group_features[group]]
            if not X_group.empty:
                pred = self.special_group_clfs[group].predict(X_group)
                groupby_pred[indices] = pred

        return groupby_pred



    def form_group_indices(self, X, group):
        indices = X['Parch'] > -1  # all trues
        for k, col in enumerate(self.groupby_cols):
            indices &= (X[col] == group[k])  # add desired value of each column to condition
        return indices


