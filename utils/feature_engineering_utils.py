import numpy as np
import pandas as pd
from collections import Counter
from functools import partial
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer, OrdinalEncoder

from imblearn.pipeline import Pipeline


def grouper(elem, group_names, group_limits):
    """ Assign elem (float/int) into a group based on group_limits of groups.
        Group limits are upper bounds for groups, i.e. group can contain only elements that are less than
        its group limit. Group limit of previous group (left in the list) is the lower bound for group."""
    return group_names[np.argmax(elem < np.array(group_limits))]


class CreateNewFeatures(BaseEstimator, TransformerMixin):
    """Pipeline component that is used to create new features. New features are specified
       in transform() method.

       Input, i.e. X, must be data frame.
       Output is a new dataframe with new features added as columns.
    """

    def __init__(self, group_size_limits=[2, 5, np.inf], group_rate_limits=[0.5, np.inf],
                 family_size_limits=[2, 5, np.inf], family_rate_limits=[0.5, np.inf], split_other_title=False):
        self.group_size_limits = group_size_limits
        self.group_rate_limits = group_rate_limits
        self.family_size_limits = family_size_limits
        self.family_rate_limits = family_rate_limits
        self.split_other_title = split_other_title

    def fit(self, X, y=None):
        self.names = X['Name']
        self.familykeys = [(ticket, name.split(',')[0]) for ticket, name in X[['Ticket', 'Name']].values]

        self.ticket_counter = Counter(X['Ticket'])
        self.familykey_counter = Counter(self.familykeys)

        X_no_family = X[(X['Parch'] == 0) & (X['SibSp'] == 0)]
        self.gender_survival_rates = X_no_family.groupby(['Sex'])['Survived'].mean()

        self.group_survival_rates = {ticket: np.mean(X.loc[X['Ticket'] == ticket, 'Survived'])
                                     for ticket in self.ticket_counter
                                     if self.ticket_counter[ticket] > 1}

        self.family_survival_rates = {
            family_key: np.mean(X.loc[(np.array(self.familykeys) == family_key).all(axis=1), 'Survived'])
            for family_key in self.familykey_counter
            if self.familykey_counter[family_key] > 1
        }

        return self

    def transform(self, X, y=None):
        new_X = X.copy(deep=True)
        assert isinstance(new_X, pd.core.frame.DataFrame), 'Input must be pandas Data Frame!'

        # Feature that tells if passenger has cabin or not
        new_X['HasCabin'] = ~new_X['Cabin'].isna()

        # Feature that tells passenger's cabin type
        new_X['CabinType'] = [cabin[0] if not pd.isnull(cabin) else 'None' for cabin in new_X['Cabin']]

        # Feature that tells surname of passenger
        new_X['Surname'] = [name.split(',')[0] for name in new_X['Name']]

        # Feature for title of passenger
        p = re.compile(r'(?P<title>\b\w+)\.')  # regular expression for searching title (ends always to dot)
        new_X['Title'] = [p.search(name).group('title') for name in new_X['Name']]

        # Feature for more grouped title of passenger
        title_mapping = {
            'Mrs': 'Mrs',
            'Mme': 'Mrs',
            'Mlle': 'Miss',
            'Miss': 'Miss',
            'Mr': 'Mr',
            'Master': 'Master',
            'Rev': 'Religious'
        }
        new_X['Title_grouped'] = new_X['Title'].map(title_mapping)
        new_X['Title_grouped'].fillna('Other', inplace=True)
        if self.split_other_title:
            mr = new_X[(new_X['Title_grouped'] == 'Other') & (new_X['Sex'] == 'male')]
            miss = new_X[(new_X['Title_grouped'] == 'Other') & (new_X['Sex'] == 'female') &
                         (new_X['SibSp'] == 0)]
            mrs = new_X[(new_X['Title_grouped'] == 'Other') & (new_X['Sex'] == 'female') &
                        (new_X['SibSp'] > 0)]
            new_X.loc[mr.index, 'Title_grouped'] = 'Mr'
            new_X.loc[miss.index, 'Title_grouped'] = 'Miss'
            new_X.loc[mrs.index, 'Title_grouped'] = 'Mrs'


        # Feature for family size
        new_X['FamilySize'] = new_X['Parch'] + new_X['SibSp'] + 1

        # Group into categories
        grouper_familysize = partial(grouper, group_names=list(range(len(self.family_size_limits))),
                                     group_limits=self.family_size_limits)
        new_X['FamilySize_grouped'] = new_X['FamilySize'].map(grouper_familysize)

        # Feature for group size
        is_in_train_data = np.isin(new_X['Name'], self.names)
        not_fitted_rows = new_X.loc[~is_in_train_data, :]
        ticket_counter_test = Counter(not_fitted_rows['Ticket'])

        new_X['GroupSize'] = [self.ticket_counter.get(ticket, 0) + ticket_counter_test.get(ticket, 0)
                              for ticket in new_X['Ticket']]

        # take family size if it's higher than group number based on tickets
        new_X['GroupSize'] = np.maximum(new_X['GroupSize'], new_X['FamilySize'])

        # Group into categories
        grouper_groupsize = partial(grouper, group_names=list(range(len(self.group_size_limits))),
                                    group_limits=self.group_size_limits)
        new_X['GroupSize_grouped'] = new_X['GroupSize'].map(grouper_groupsize)

        # Feature for group survival rates
        new_X['GroupRate'] = [self.group_survival_rates.get(row['Ticket'],
                                                            self.gender_survival_rates[row['Sex']])
                              for _, row in new_X.iterrows()]

        # GroupRate into categories
        grouper_grouprate = partial(grouper, group_names=list(range(len(self.group_rate_limits))),
                                    group_limits=self.group_rate_limits)
        new_X['GroupRate_grouped'] = new_X['GroupRate'].map(grouper_grouprate)

        # Feature for family survival rates
        new_X['FamilyRate'] = [self.family_survival_rates.get((row['Ticket'], row['Surname']),
                                                              self.gender_survival_rates[row['Sex']])
                               for _, row in new_X.iterrows()]

        grouper_familyrate = partial(grouper, group_names=list(range(len(self.family_rate_limits))),
                                     group_limits=self.family_rate_limits)
        new_X['FamilyRate_grouped'] = new_X['FamilyRate'].map(grouper_familyrate)

        # Feature for adjusted Fare
        new_X['Fare_adjusted'] = new_X['Fare'] / new_X['GroupSize']

        return new_X


class Preprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, label='Survived', categorical_cols=[], numerical_cols=[], ordinal_cols=[]):
        self.label = label
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.ordinal_cols = ordinal_cols

    def fit(self, X, y=None):
        X_features = X.copy(deep=True)

        if len(self.categorical_cols) == len(self.numerical_cols) == len(self.ordinal_cols) == 0:
            X_features = X.drop(columns=[self.label]) if self.label in X.columns else X.copy(deep=True)
            cat_cols = X_features.dtypes[X_features.dtypes == 'object'].index.tolist()
            self.categorical_cols = [col for col in cat_cols if len(set(X_features[col])) <= 8]

            num_cols = X_features.dtypes[np.isin(X_features.dtypes, ['float64'])].index.tolist()
            self.numerical_cols = [col for col in num_cols if len(set(X_features[col])) > 8]

            ord_cols = X_features.dtypes[np.isin(X_features.dtypes, ['int64'])].index.tolist()
            self.ordinal_cols = [col for col in ord_cols if len(set(X_features[col])) <= 8]

        fn_numerical = Pipeline(steps=[
            ('scaling', StandardScaler()),
        ])
        fn_categorical = Pipeline(steps=[
            ('encoding', OneHotEncoder(drop='if_binary', handle_unknown='infrequent_if_exist', sparse=False))
        ])
        fn_ordinal = Pipeline(steps=[
            ('encoding', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan))
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', fn_numerical, self.numerical_cols),
            ('cat', fn_categorical, self.categorical_cols),
            ('ord', fn_ordinal, self.ordinal_cols)
        ])
        self.preprocessor.fit(X_features, y)

        return self

    def get_categorical(self, X):
        return self.categorical_cols

    def get_numerical(self, X):
        return self.numerical_cols

    def get_ordinal(self, X):
        return self.ordinal_cols

    def transform(self, X, y=None):
        self.new_colnames = self.preprocessor.get_feature_names_out()
        return pd.DataFrame(self.preprocessor.transform(X), columns=self.new_colnames, index=X.index)
