from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class MultiColumnOneHotEncoder(BaseEstimator, TransformerMixin):
    """Transformer for applying label encoder on multiple columns.

    This transformer applies label encoding to columns in a dataset.
    """

    def __init__(self, excluded_factors = []):
        self.factors = []
        self.excluded_factors = excluded_factors

    def transform(self, X, **transform_params):
        """Transforms X to have columns label encoded.

        Args:
            X (obj): The dataset to transform. Can be dataframe or matrix.
            transform_params (kwargs, optional): Additional params.

        Returns:
            The transformed dataset with the label encoded columns.
        """
        X = X.fillna('NaN')  # fill null values with 'NaN'
        transformed = X.copy()
        factors_to_use = [x for x in self.factors if x not in self.excluded_factors]

        for factor in factors_to_use:
            transformed['factor_' + factor] = 0
            transformed.loc[transformed.other_factor_1 == factor, 'factor_' + factor] = 1
            transformed.loc[transformed.other_factor_2 == factor, 'factor_' + factor] = 1
            transformed.loc[transformed.other_factor_3 == factor, 'factor_' + factor] = 1

        for factor in self.excluded_factors:
            transformed["factor_misc"] = 0
            transformed.loc[transformed.other_factor_1 == factor, "factor_misc"] = 1
            transformed.loc[transformed.other_factor_2 == factor, "factor_misc"] = 1
            transformed.loc[transformed.other_factor_3 == factor, "factor_misc"] = 1

        transformed = transformed.drop(["other_factor_1", "other_factor_2", "other_factor_3"], axis=1)

        return transformed

    def fit(self, X, y=None, **fit_params):
        """Fits transfomer over X.

        """
        dataset = X.copy()
        self.factors = set(np.concatenate([dataset.other_factor_1.unique(), dataset.other_factor_2.unique(), dataset.other_factor_3.unique()]))
        self.factors = list(self.factors.difference([np.nan, 'N/A or Unknown']))
        return self
