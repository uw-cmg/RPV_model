import os
import numpy as np
import pandas as pd
from copy import copy
import joblib
import tensorflow as tf
from mastml.models import EnsembleModel
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def rebuild_model(n_features, model_folder):

    # We need to define the function that builds the network architecture
    def keras_model(n_features):
        model = Sequential()
        model.add(Dense(1024, input_dim=n_features, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    model_keras = KerasRegressor(build_fn=keras_model, epochs=250, batch_size=100, verbose=0)
    model_bagged_keras_rebuild = EnsembleModel(model=model_keras, n_estimators=10)

    num_models = 10
    models = list()
    for i in range(num_models):
        models.append(tf.keras.models.load_model(os.path.join(model_folder, 'keras_model_' + str(i))))

    model_bagged_keras_rebuild.model.estimators_ = models
    model_bagged_keras_rebuild.model.estimators_features_ = [np.arange(0, n_features) for i in models]

    return model_bagged_keras_rebuild

def get_preds_ebars(model, df_featurized, preprocessor, return_ebars=True):
    preds_each = list()
    ebars_each = list()

    df_featurized_scaled = preprocessor.transform(pd.DataFrame(df_featurized))

    if return_ebars == True:
        for i, x in df_featurized_scaled.iterrows():
            preds_per_data = list()
            for m in model.model.estimators_:
                preds_per_data.append(m.predict(pd.DataFrame(x).T)) #pd.DataFrame(x).T
            preds_each.append(np.mean(preds_per_data))
            ebars_each.append(np.std(preds_per_data))

    else:
        preds_each = model.predict(df_featurized_scaled)
        ebars_each = [np.nan for i in range(preds_each.shape[0])]

    if return_ebars == True:
        a = -0.041
        b = 2.041
        c = 3.124
        ebars_each_recal = a*np.array(ebars_each)**2 + b*np.array(ebars_each) + c
    else:
        ebars_each_recal = ebars_each


    return np.array(preds_each).ravel(), np.array(ebars_each_recal).ravel()

def make_predictions_DNN(df_featurized, model_folder):

    # Rebuild the saved model
    n_features = df_featurized.shape[1]
    model = rebuild_model(n_features, model_folder)

    # Normalize the input features
    preprocessor = joblib.load(os.path.join(model_folder, 'StandardScaler.pkl'))
    
    # Get predictions and error bars from model
    preds, ebars = get_preds_ebars(model, df_featurized, preprocessor, return_ebars=True)

    pred_dict = {'preds':preds,
                 'ebars':ebars}

    return pd.DataFrame(pred_dict)

def test(df):
    #pred_dict = {'preds': ['This is a test'], 'ebars': ['This is a test']}
    pred_arr = np.array([['here is some data'], ['here are some ebars']])
    model = keras.models.load_model('keras_model_0')
    return pred_arr
    #return pd.DataFrame(pred_dict)
    #return np.sqrt(x)
