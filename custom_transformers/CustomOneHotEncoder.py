from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Pretreats other object columns
    """
    def __init__(self, rare_values = {'person_attributes' : [], 'seat': [], 'other_person_location' : []}):
        self.all_values = {}
        self.rare_values = rare_values

    def fit(self, X, y=None):

        dataset = X.copy()
        self.all_values['m_or_f'] = ['m', 'f']
        self.all_values['person_attributes'] = list(dataset['person_attributes'].unique())
        self.all_values['person_attributes'] = [x for x in self.all_values['person_attributes'] if x not in [np.nan, 'N/A or Unknown']]
        self.all_values['seat'] = list(dataset['seat'].unique())
        self.all_values['seat'] = [x for x in self.all_values['seat'] if x not in [np.nan, 'N/A or Unknown']]
        self.all_values['other_person_location'] = list(dataset['other_person_location'].unique())
        self.all_values['other_person_location'] = [x for x in self.all_values['other_person_location'] if x not in [np.nan, 'N/A or Unknown']]
        return self

    def transform(self, X):

        transformed = X.copy()

        transformed['sex'] = 0
        transformed.loc[transformed.m_or_f == 'f', 'sex'] = 1

        for factor in [x for x in self.all_values['person_attributes'] if x not in self.rare_values['person_attributes']]:
            transformed['atrib_' + factor] = 0
            transformed.loc[transformed.person_attributes == factor, 'atrib_' + factor] = 1

        for factor in self.rare_values['person_attributes']:
            transformed['atrib_misc'] = 0
            transformed.loc[transformed.person_attributes == factor, 'atrib_misc'] = 1

        for factor in [x for x in self.all_values['seat'] if x not in self.rare_values['seat']]:
            transformed['seat_' + factor] = 0
            transformed.loc[transformed.seat == factor, 'seat_' + factor] = 1

        for factor in self.rare_values['seat']:
            transformed['seat_misc'] = 0
            transformed.loc[transformed.seat == factor, 'seat_misc'] = 1

        for factor in [x for x in self.all_values['other_person_location'] if x not in self.rare_values['other_person_location']]:
            transformed['other_loc_' + str(factor)] = 0
            transformed.loc[transformed.other_person_location == factor, 'other_loc_' + factor] = 1

        for factor in self.rare_values['other_person_location']:
            transformed['other_loc_misc'] = 0
            transformed.loc[transformed.other_person_location == factor, 'other_loc_misc'] = 1

        transformed = transformed.drop(["m_or_f", "person_attributes", "seat", "other_person_location"], axis=1)

        return transformed
