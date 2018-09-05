import os
import json
import pickle
from sklearn.externals import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict

#########################################
# Custom transformers

from sklearn.base import BaseEstimator, TransformerMixin

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
        factors_to_use = self.factors.difference(self.excluded_factors)

        for factor in factors_to_use:
            transformed[factor] = 0
            transformed.loc[transformed.other_factor_1 == factor, factor] = 1
            transformed.loc[transformed.other_factor_2 == factor, factor] = 1
            transformed.loc[transformed.other_factor_3 == factor, factor] = 1

        transformed = transformed.drop(["other_factor_1", "other_factor_2", "other_factor_3"], axis=1)

        return transformed

    def fit(self, X, y=None, **fit_params):
        """Fits transfomer over X.

        """
        dataset = X.copy()
        self.factors = set(np.concatenate([dataset.other_factor_1.unique(), dataset.other_factor_2.unique(), dataset.other_factor_3.unique()]))
        self.factors = self.factors.difference([np.nan, 'N/A or Unknown'])
        return self


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]






########################################
# Begin database stuff

if 'DATABASE_URL' in os.environ:
    db_url = os.environ['DATABASE_URL']
    dbname = db_url.split('@')[1].split('/')[1]
    user = db_url.split('@')[0].split(':')[1].lstrip('//')
    password = db_url.split('@')[0].split(':')[2]
    host = db_url.split('@')[1].split('/')[0].split(':')[0]
    port = db_url.split('@')[1].split('/')[0].split(':')[1]
    DB = PostgresqlDatabase(
        dbname,
        user=user,
        password=password,
        host=host,
        port=port,
    )
else:
    DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # flask provides a deserialization convenience function called
    # get_json that will work if the mimetype is application/json
    obs_dict = request.get_json()
    _id = obs_dict['id']
    observation = obs_dict['observation']
    # now do what we already learned in the notebooks about how to transform
    # a single observation into a dataframe that will work with a pipeline
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    # now get ourselves an actual prediction of the positive class
    proba = pipeline.predict_proba(obs)[0, 1]
    response = {'proba': proba}
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = 'Observation ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
        return jsonify({'error': error_msg})


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
