# imports
import pandas as pd
import numpy as np
import pickle
import xmltodict
import os
import time
import lxml.etree

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.layers import concatenate, Input, BatchNormalization, Activation
from keras.utils import normalize
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import Nadam
from keras import backend as K

import logging
logging.getLogger('tensorflow').disabled = True

def preprocess_file(file, ev_keys):
    result = pd.DataFrame()
    
    doc = xmltodict.parse(lxml.etree.tostring(file))

    data = [[x[key] if key in x.keys() else '' for key in ev_keys
             ] for x in doc['Games']['Game']['Event']]
    
    events = pd.DataFrame(data, columns=ev_keys)

    events['game_id'] = doc['Games']['Game']['@id']
    events['@player_id'] = [int(x) if x != '' else 0 for x in events['@player_id']]

    return events

# Model for player_id identification
def model_playerid(file_weight, sequence_shape):
   model_lstm_in = Input(shape=(sequence_shape[0], sequence_shape[1]-1))
   model_lstm_out = LSTM(300, input_shape=(sequence_shape[0], sequence_shape[1]-1),
                         name='layer_1',
                         return_sequences=True,
                         recurrent_activation='sigmoid'
                         )(model_lstm_in)
   model_lstm_out2 = LSTM(200, input_shape=(sequence_shape[0], sequence_shape[1]-1),
                          name='layer_1_1',
                          recurrent_activation='sigmoid'
                          )(model_lstm_out)
   model_lstm = Model(model_lstm_in, model_lstm_out2)

   model_embed_in = Input(shape=(sequence_shape[0], 1))
   conv_1D = Conv1D(filters=64,
                    kernel_size=3, 
                    padding='same',
                    activation='relu'
                    )(model_embed_in)
   model_embed_inter = MaxPooling1D(pool_size=2)(conv_1D)

   model_embed_out = LSTM(300, input_shape=(sequence_shape[0], 1),
                          name='layer_2',
                          return_sequences=True,
                          recurrent_activation='sigmoid'
                          )(model_embed_inter)
   model_embed_out2 = LSTM(200, input_shape=(sequence_shape[0], sequence_shape[1]-1),
                           name='layer_2_1',
                           recurrent_activation='sigmoid'
                          )(model_embed_out)


   model_embed = Model(model_embed_in, model_embed_out2)

   merged = concatenate([model_lstm_out2, model_embed_out2])

   out_dense = Dense(232, name='output_layer', use_bias=False)(merged)
   out_bn = BatchNormalization()(out_dense)
   out = Activation('softmax')(out_bn)
   model = Model(inputs = [model_lstm.input, model_embed.input], outputs = [out])

   model.load_weights(file_weight)
   return model


# Model for team_id prediction
def model_teamid(file_weight, sequence_shape):
    model_lstm_in = Input(shape=(sequence_shape[0], sequence_shape[1]))
    model_lstm_out = LSTM(256, input_shape=(10, 1),
                      name='layer_1',
                      return_sequences=True,
                      recurrent_dropout=0.2
                      )(model_lstm_in)
    model_lstm_out_2 = LSTM(128,
                       name='layer_1_1',
                       recurrent_dropout=0.2
                      )(model_lstm_out)

    out_dense = Dense(2, name='output_layer', activation='sigmoid')(model_lstm_out_2)

    model = Model(inputs = model_lstm_in, outputs = out_dense)
    model.load_weights(file_weight)
    return model


# Model for x,y prediction
def model_xy(file_weight, sequence_shape):
   model_lstm_in = Input(shape=(sequence_shape[0], sequence_shape[1]))
   model_lstm_out = LSTM(256, input_shape=(sequence_shape[0], sequence_shape[1]),
                      	 name='layer_1',
                      	 return_sequences=True
                      	 )(model_lstm_in)

   out_lstm = Dropout(0.1)(model_lstm_out)

   model_lstm_out_2 = LSTM(128, name='layer_1_1')(out_lstm)

   out_dense = Dense(2, name='output_layer',
                  	  activation='linear'
                  	  )(model_lstm_out_2)

   model = Model(inputs = model_lstm_in, outputs = out_dense)
   model.load_weights(file_weight)
   return model


# Result function
def Result(xml_file):
    start = time.time()

    # Column from xml file
    event_keys = ['@type_id', '@period_id', '@min', '@sec',
                  '@player_id', '@team_id', '@outcome', '@x', '@y', '@keypass',
                  'Q']
    
    # Load data from xml file
    df = preprocess_file(xml_file, event_keys)

    # Preprocess data
    convert_dict = {'@type_id': int, '@period_id': int, '@team_id': int,
		    '@x': float, '@y': float}    
 
    df.astype(convert_dict, copy=False)
    
    df['minutes'] = (df['@min'].astype(int) * 60 + df['@sec'].astype(int)) / 60
    df['delta_time'] = df['minutes'].diff()
    df['@period_id'] = df['@period_id'].apply(lambda x: int(x))
    df['@keypass'] = [int(x) if x != '' else 0 for x in df['@keypass']]
    df['@outcome'] = [int(x) if x != '' else 0 for x in df['@outcome']]
    
    dummy_period = pd.get_dummies(df['@period_id'], prefix='period')
    period = int(dummy_period.columns[0][7:])
    
    dummy_team = pd.get_dummies(df['@team_id'], prefix='team')
    dummy_outcome = pd.get_dummies(df['@outcome'], prefix='outcome')
    feat_to_drop = ['@period_id', '@team_id', '@outcome', '@min', '@sec',
                    'Q', 'game_id', 'minutes']
    df.drop(feat_to_drop, inplace=True, axis=1)

    if period == 1:
    	dummy_period_2 = pd.DataFrame.from_dict({'period_2': np.zeros(len(dummy_period)) })

    	df = pd.concat((df, dummy_period, dummy_period_2, dummy_team, dummy_outcome),
              	       axis=1)
    else:
    	dummy_period_1 = pd.DataFrame({'period_1': np.zeros(len(dummy_period)) })

    	df = pd.concat((df, dummy_period_1, dummy_period, dummy_team, dummy_outcome),
              	       axis=1)
    
    df['@type_id'] = df['@type_id'].astype('int')
    df['@x'] = df['@x'].astype('float')
    df['@y'] = df['@y'].astype('float')

    example = df[df['@player_id'] == 1].to_numpy()
    example = np.reshape(example, (1, example.shape[0], example.shape[1]))

    example_padded = pad_sequences(example, maxlen=45, value=-1)

    ex_norm = normalize(example_padded)
    # Load model to predict players
    weight_model_file = 'model_playerid.h5'
    model_player = model_playerid(weight_model_file, (45, 12))

    # Predict player
    player_pred = model_player.predict([ex_norm[:, :, 1:],
                                        ex_norm[:, :, 0:1]
                                        ])

    predict_class = player_pred.argmax()
    value_class = np.max(player_pred)
    player_dict = pickle.load(open('player_labels_mapping_reverse.p', 'rb'))
    
    player_id = player_dict[predict_class]
    
    # Inputs for model to predict x,y, team_id
    inputs_xyid = df[['team_0', 
    		      'team_1',
    		      '@x',
    		      '@y',
    		      'delta_time']].tail(10).to_numpy()
    
    inputs_xyid = np.reshape(inputs_xyid, (1, inputs_xyid.shape[0], inputs_xyid.shape[1]))

    # Load model for team_id
    weight_model_file = 'model_teamid.h5'
    model_tid = model_teamid(weight_model_file, (10, 5))
    
    # Predict team id
    teamid_prob = model_tid.predict(normalize(inputs_xyid))
    teamid_pred = teamid_prob.argmax()
    
    # Load model for x,y
    weight_modelxy_file = 'model_xy.h5'
    modelxy = model_xy(weight_modelxy_file, (10, 5))
    
    # Predict x,y
    xy_pred = modelxy.predict(inputs_xyid)

    # Create csv file
    df = pd.DataFrame(data={"playerid": [float(player_id)],
    			    "teamid": [float(teamid_pred)],
    			    "y": [xy_pred[0][1]],
    			    "x": [xy_pred[0][0]] })
    
    df.to_csv("res_psgx.csv", sep=',', index=False, header=False)
     
    K.clear_session()
    end = time.time()

    return print('Prediction result in res_psgx.csv file. Total time: {} seconds.'.format(end-start))

