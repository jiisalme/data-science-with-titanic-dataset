import numpy as np
import pandas as pd
from collections import Counter


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin


def create_nan_dataframe_row(columns):
    return pd.DataFrame([[np.nan for i in columns]], columns=columns).iloc[0]


def store_to_dict_of_lists(dictionary, key, value):
    dictionary.setdefault(key, []).append(value)


def find_partner(person, partner_candidates, possible_titles):
    surname, rest_of_name = person['Name'].split(', ')[:2]
    first_name = ' '.join(rest_of_name.split(' ')[1:])
    possible_partner_names = [f'{surname}, {title}. {first_name}' for title in possible_titles]
    for index, partner_candidate in partner_candidates.iterrows():
        for possible_partner_name in possible_partner_names:
            if possible_partner_name in partner_candidate['Name']:
                return partner_candidate
    return create_nan_dataframe_row(partner_candidates.columns)


def find_children(parent1, parent2, family):
    surname = parent1['Surname']

    children = family[(family['Surname'] == surname) &
                      (family['Name'] != parent2['Name']) &
                      (family['Age'] < 50) &
                      (family['Parch'] > 0)]
                      #(np.isin(family['Title_grouped'], ['Master', 'Miss']))]


    if not np.isnan(parent1['Age']):
        children = children[children['Age'] < parent1['Age'] - 15]

    if not children.empty:
        max_sibsp_count = Counter(children['SibSp']).most_common(1)[0][0]
        return children[children['SibSp'] == max_sibsp_count]
    return children


def find_parents(person, family, partner, sex, possible_titles):
    surname = person['Surname']
    parents = family[(family['Parch'] > 0) &
                     (np.isin(family['Title_grouped'], possible_titles)) &
                     (family['Sex'] == sex) &
                     (family['Surname'] == surname) &
                     (family['Name'] != partner['Name']) &
                     (family['Age'] >= 14)]

    # Master turn into Mr at age 14, Mrs are usually >14 years old and usually <15 years old don't get childs
    if person['Title'] in ['Mr', 'Mrs']:
        return parents[parents['Age'] > 14 + 15]

    return parents


def find_siblings(row, family, age_estimates):
    siblings = family[(family['SibSp'] > 0) &
                  (family['SibSp'] == row['SibSp']) &
                  (family['Surname'] == row['Surname']) &
                  (family['Parch'] == row['Parch'])]
    if age_estimates:
        age = np.nanmean(age_estimates)
        siblings = siblings[(siblings['Age'] > age - 15) & (siblings['Age'] < age + 15)]

    return siblings

class FamilyAgeImputer(BaseEstimator, TransformerMixin):
    """Custom imputation for missing age values using passenger family.

    base_imputation_method: regressor method with .fit() and .predict() to make base estimates for Age

    Return value of transform method is the imputed Age column and mask telling which values were imputed.
    """
    def __init__(self, base_imputation_method):
        self.husband_wife_avg_age_diffs = {}
        self.mom_child_avg_age_diffs = {}
        self.dad_child_avg_age_diffs = {}

        neutral_titles = ['Rev', 'Major', 'Capt', 'Col', 'Dr',]
        self.possible_husband_titles = ['Mr', 'Don', 'Sir'] + neutral_titles
        self.possible_wife_titles = ['Mrs', 'Countess', 'Lady', 'Mme'] + neutral_titles

        numerical = []
        ordinal = []
        categorical = ['Title_grouped', 'Pclass', 'FamilySize_grouped']

        self.base_imputation_method = base_imputation_method

    def fit(self, X, y=None):

        self.mom_ch_diffs, self.dad_ch_diffs, self.husband_wife_diffs = {}, {}, {}

        self.complete_data = X.dropna(subset=['Age'])
        self.base_imputation_method.fit(self.complete_data, self.complete_data['Age'])

        for ticket in set(self.complete_data['Ticket']):
            group = self.complete_data[self.complete_data['Ticket'] == ticket]
            family = group[group['FamilySize'] > 1]
            if not family.empty:
                possible_husbands = family[np.isin(family['Title_grouped'], self.possible_husband_titles) &
                                           (family['Sex'] == 'male')]
                possible_wives = family[np.isin(family['Title_grouped'], self.possible_wife_titles) &
                                        (family['Sex'] == 'female')]
                # Search first for husband-wife pairs
                for index, husband_candidate in possible_husbands.iterrows():
                    pclass = husband_candidate['Pclass']
                    wife = find_partner(husband_candidate, possible_wives, self.possible_wife_titles)
                    if not wife.dropna().empty:
                        possible_wives = possible_wives[possible_wives['Name'] != wife['Name']]

                    store_to_dict_of_lists(self.husband_wife_diffs, pclass, husband_candidate['Age'] - wife['Age'])
                    if husband_candidate['Parch'] > 0:  # Search for dad-child and wife-child relations
                        children = find_children(husband_candidate, wife, family)
                        avg_children_age = np.mean(children['Age'])
                        store_to_dict_of_lists(self.mom_ch_diffs, pclass, wife['Age'] - avg_children_age)
                        store_to_dict_of_lists(self.dad_ch_diffs, pclass, husband_candidate['Age'] - avg_children_age)

                husband = create_nan_dataframe_row(self.complete_data.columns)
                for index, mom_candidate in possible_wives.iterrows():
                    pclass = mom_candidate['Pclass']
                    # Search for mom-child relations (no husband onboard or in training data with age value)
                    if mom_candidate['Parch'] > 0:
                        children = find_children(mom_candidate, husband, family)
                        age_diff = mom_candidate['Age'] - np.mean(children['Age'])
                        store_to_dict_of_lists(self.mom_ch_diffs, pclass, age_diff)

        self.husband_wife_avg_age_diffs = {key: np.nanmean(val) for key, val in self.husband_wife_diffs.items()}
        self.mom_child_avg_age_diffs = {key: np.nanmean(val) for key, val in self.mom_ch_diffs.items()}
        self.dad_child_avg_age_diffs = {key: np.nanmean(val) for key, val in self.dad_ch_diffs.items()}

        return self

    def transform(self, X):
        new_rows = X[~np.isin(X['Name'], self.complete_data['Name'])].dropna(subset=['Age'])
        self.complete_data = pd.concat((self.complete_data, new_rows))  # update complete data
        new_X = X.copy(deep=True)
        nan_age_indices = new_X['Age'].isna()
        base_estimates = self.base_imputation_method.predict(new_X[nan_age_indices])

        for i, (index, row) in enumerate(new_X[nan_age_indices].iterrows()):
            row['Age'] = self.impute_age(row, base_estimates[i])
            self.complete_data.loc[index, :] = row  # add row to complete data
            new_X.loc[index, :] = row  # add row also to transformed data

        return new_X['Age'], nan_age_indices

    def impute_age(self, row, default_age):
        group = self.complete_data[self.complete_data['Ticket'] == row['Ticket']]
        family = group[group['FamilySize'] > 1]

        # use predictions of base_imputation_method as default age values
        age = default_age

        if not family.empty:  # some family members do have age stored -> utilize them and prioritize
            family_age = self.get_age_from_family(row, family, age)
            if not np.isnan(family_age):
                return family_age
            else:
                print(f'NaN produced! {row["Name"]}')

        return age

    def get_age_from_family(self, row, family, default_age):
        surname = row['Surname']
        partner = create_nan_dataframe_row(family.columns)
        age_estimates = []

        if (row['Title'] in self.possible_wife_titles) & (row['Sex'] == 'female'):
            possible_husbands = family[np.isin(family['Title'], self.possible_husband_titles) &
                                       (family['Sex'] == 'male')]
            partner = find_partner(row, possible_husbands, self.possible_husband_titles)
            children = find_children(row, partner, family)
            if not children.empty and row['Parch'] > 0:
                age_estimates.append(np.mean(children['Age']) + self.mom_child_avg_age_diffs[row['Pclass']])
            if not partner.dropna().empty:
                age_estimates.append(partner['Age'] - self.husband_wife_avg_age_diffs[row['Pclass']])

        elif (row['Title'] in self.possible_husband_titles) & (row['Sex'] == 'male'):

            possible_wives = family[np.isin(family['Title'], self.possible_wife_titles) &
                                    (family['Sex'] == 'female')]
            partner = find_partner(row, possible_wives, self.possible_wife_titles)
            children = find_children(row, partner, family)
            moms = find_parents(row, family, partner, 'female', self.possible_wife_titles)

            if not children.empty and row['Parch'] > 0 and moms.empty:
                age_estimates.append(np.mean(children['Age']) + self.dad_child_avg_age_diffs[row['Pclass']])
            if not partner.dropna().empty:
                age_estimates.append(partner['Age'] + self.husband_wife_avg_age_diffs[row['Pclass']])


        moms = find_parents(row, family, partner, 'female', self.possible_wife_titles)
        if not moms.empty:
            age_estimates.append(max(0, np.mean(moms['Age']) - self.mom_child_avg_age_diffs[row['Pclass']]))

        dads = find_parents(row, family, partner, 'male', self.possible_husband_titles)
        if not dads.empty:
            age_estimates.append(max(0, np.mean(dads['Age']) - self.dad_child_avg_age_diffs[row['Pclass']]))

        siblings = find_siblings(row, family, age_estimates)
        if not siblings.empty:
            age_estimates.append(np.mean(siblings['Age']))

        return default_age if not age_estimates else np.nanmean(age_estimates)


class GroupByImputer(BaseEstimator, RegressorMixin):
    """Imputer that does imputation to target_col by grouping data by groupby_cols and applying
       the desired method for groupby object.

       target_col: column that needs imputation
       groupby_cols: columns that are used in pandas groupby
       method: method to be used for grouby object, for example pd.Series.median

       Transform returns transformed column as well as mask that indicates which values were imputed."""

    def __init__(self, target_col, groupby_cols, method):
        self.target_col = target_col
        self.groupby_cols = groupby_cols
        self.method = method

    def fit(self, X, y=None):
        self.average_age = np.nanmean(X['Age'])
        self.groupby_objects = X.groupby(self.groupby_cols)[self.target_col].agg(self.method)

        return self

    def transform(self, X):
        new_X = X.copy(deep=True)
        nan_rows = new_X[self.target_col].isna()
        new_X.loc[nan_rows, self.target_col] = self.predict(new_X[nan_rows])

        return new_X, nan_rows

    def predict(self, X):
        y_pred = list(np.zeros(X.shape[0]))
        for i, (index, row) in enumerate(X.iterrows()):
            age_key = tuple(row[col] for col in self.groupby_cols) if len(self.groupby_cols) > 1 \
                else row[self.groupby_cols[0]]
            y_pred[i] = self.groupby_objects[age_key] if age_key in self.groupby_objects else self.average_age

        return np.array(y_pred)

class Imputer(BaseEstimator, TransformerMixin):
    """Custom imputation for missing values.

    col_imputation_methods: dict of form {col: imputation_method} defines imputation method for each column
    imputation_method should have .fit() and .predict() methods defined.

    """

    def __init__(self, col_imputation_methods):

        self.col_imputation_methods = col_imputation_methods
        self.imputed_vals = {}

    def fit(self, X, y=None):

        for col, method in self.col_imputation_methods.items():
            method.fit(X, y)

        return self

    def transform(self, X):
        new_X = X.copy(deep=True)
        incomplete_cols = new_X.columns[new_X.isna().any(axis=0)]
        impute_cols = incomplete_cols[np.isin(incomplete_cols, list(self.col_imputation_methods.keys()))]

        for col in impute_cols:
            impute_values, mask = self.col_imputation_methods[col].transform(new_X)
            new_X.loc[:, col] = impute_values
            self.imputed_vals[col] = mask

        return new_X

    def get_last_imputed_values(self):
        return self.imputed_vals