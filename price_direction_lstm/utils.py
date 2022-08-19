from sklearn.preprocessing import MinMaxScaler
import numpy as np
# Import tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Conv2D, Bidirectional
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.regularizers import L1L2
import yfinance as yf
from datetime import datetime, timedelta
import sys
from minisom import MiniSom
import matplotlib.pyplot as plt

sys.path.append('../src/')
import technical_indicators as ti
import yfinance as yf
def prep_ticker_data(symbol, start_date = None, end_date = None, \
                     if_features_delta = True, if_target_delta = True, \
                     if_features_binary = True, if_target_binary = True, shift_days = 1, target_variable = 'label'):
    """
    The data prep procedure
    -------------------------------
    symbol : string, the target symbol, the asset code in yfinance
    start_date: string, format 'YYYY-MM-DD'
    end_date: string,  format 'YYYY-MM-DD'
    if_features_delta: bool, if use delta for features
    if_target_delta: bool, if use delta for target variable
    if_features_binary: bool, if the features are binary
    if_target_binary: bool, if the target variable is binary
    """
    df = yf.download(symbol, start=start_date, end=end_date)
    df.reset_index(inplace=True)

    #delete as forex doesn't provide the volume information
    del df['Volume']
    del df['Adj Close']

    df['temp_Close'] = df['Close']

    df = ti.commodity_channel_index(df, 5)
    df = ti.commodity_channel_index(df, 10)
    df = ti.commodity_channel_index(df, 21)#select
    df = ti.commodity_channel_index(df, 50)
    
    df = ti.average_true_range(df, 5)#select
    df = ti.average_true_range(df, 10)
    df = ti.average_true_range(df, 21)
    df = ti.average_true_range(df, 50)
    
    df = ti.bollinger_bands(df, 5)
    df = ti.bollinger_bands(df, 10)
    df = ti.bollinger_bands(df, 21)
    df = ti.bollinger_bands(df, 50)

    df = ti.exponential_moving_average(df, 20)
    df = ti.exponential_moving_average(df, 5)
    df = ti.exponential_moving_average(df, 10)
    df = ti.exponential_moving_average(df, 21)
    df = ti.exponential_moving_average(df, 50)

    df = ti.moving_average(df, 5)
    df = ti.moving_average(df, 10)
    df = ti.moving_average(df, 21)
    df = ti.moving_average(df, 50)

    df = ti.momentum(df, 5)
    df = ti.momentum(df, 10)
    df = ti.momentum(df, 21)
    df = ti.momentum(df, 50)

    df = ti.rate_of_change(df, 1)
    df = ti.rate_of_change(df, 5)
    df = ti.rate_of_change(df, 10)
    df = ti.rate_of_change(df, 21)
    df = ti.rate_of_change(df, 50)

    df = ti.relative_strength_index(df, 5)
    df = ti.relative_strength_index(df, 10)
    df = ti.relative_strength_index(df, 21) #select
    df = ti.relative_strength_index(df, 50)
    
    df = ti.williams_ad(df) #select
    
    df = ti.standard_deviation(df, 5)
    df = ti.standard_deviation(df, 10)
    df = ti.standard_deviation(df, 21)
    df = ti.standard_deviation(df, 50)

    delta_columns = list(df.columns[(df.columns!='temp_Close') & (df.columns!='Date')])

    if if_features_delta == True:
        for col in delta_columns:
            df[col] = df[col] - df[col].shift(1)
    if if_features_binary == True:
        for col in delta_columns:
            df[col] = (df[col]>=0).astype(int)

    df['Date'] = df['Date'].shift(-1)
    
    if if_target_delta == True:
        df[target_variable] = df['temp_Close'].shift(-1) - df['temp_Close']
    else:
        df[target_variable] = df['temp_Close'].shift(-1)
    if if_target_binary == True:
        df[target_variable] = (df[target_variable]>=0).astype(int)
    df['return'] = df['temp_Close'].shift(-1) - df['temp_Close']
    del df['temp_Close']
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method = 'ffill')
    df = df.dropna()
    df = df.set_index('Date')
    return df

def load_external_data(df_result,
                       gold_path = 'Data/commedity/gold.csv',\
                       oil_path = 'Data/commedity/wti-crude-oil.csv',\
                       ffr_path = 'Data/fundamental/Federal Fund Rate.csv', \
                       usdi_path = 'Data/fundamental/USD Index.csv', \
                       if_delta = True, if_features_binary = True, window = 30):
    """
    load external commedity file and index file
    -------------------------------
    df_result: orginal DataFrame
    *_path: string, file paths
    if_delta: bool, if use delta for features
    if_features_binary: bool, if the features are binary
    window: the running window
    """
    symbol_lst = ['GOLD', 'OIL', 'Federal_Fund_Rate', 'USD_Index']
    for symbol in symbol_lst:
        if symbol == 'GOLD':
            df = pd.read_csv(gold_path)
        if symbol == 'OIL':
            df = pd.read_csv(oil_path)
        if symbol == 'Federal_Fund_Rate':
            df = pd.read_csv(ffr_path)
        if symbol == 'USD_Index':
            df = pd.read_csv(usdi_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        x_col = symbol+'_Close'
        df[x_col] = df['Price'].rolling(window).mean()
        df = df[['Date', x_col]]
        #shift the data
        df['Date'] = df['Date'].shift(-1)
        df = df.dropna()
        df = df.set_index('Date')
        df_result = df_result.join(df[x_col])
    for symbol in symbol_lst:
        x_col = symbol+'_Close'
        df_result[x_col] = df_result[x_col].fillna(method='ffill')
        df_result[x_col] = df_result[x_col].astype(float)
        if if_delta == True:
            df_result[x_col] = df_result[x_col] - df_result[x_col].shift(1)
        if if_features_binary == True:
             df_result[x_col] = (df_result[x_col] >= 0).astype(int)
    df_result = df_result.dropna()
    return df_result

class StackedAutoencoderModel():
    def __init__(self, key, batch_size = 64, epochs = 200, verbose = 1, \
                opt = tf.keras.optimizers.Adam(learning_rate=0.0005), \
                 loss_fn = 'mean_squared_error'):
        self.symbol_key = key
        self.params = {}
        self.params['batch_size'] = batch_size
        self.params['epochs'] = epochs
        self.params['verbose'] = verbose
        self.params['optimizer'] = opt
        self.params['loss'] = loss_fn
    def fit(self, X_train_encode, X_validate_encode, hidden_layers = [30,15,15], target_feature_number=10, metric = 'val_loss', bottom_epochs = 20):
        model = Sequential()
        #encoder
        model.add(Dense(X_train_encode.shape[1],  activation='elu', input_shape=(X_train_encode.shape[1],)))
        model.add(Dense(hidden_layers[0],  activation='elu'))
        model.add(Dense(hidden_layers[1],  activation='elu'))
        model.add(Dense(hidden_layers[2],  activation='elu'))
        #central
        model.add(Dense(target_feature_number, activation='elu', name="bottleneck"))
        #decoder
        model.add(Dense(hidden_layers[2],  activation='elu'))
        model.add(Dense(hidden_layers[1],  activation='elu'))
        model.add(Dense(hidden_layers[0],  activation='elu'))
        model.add(Dense(X_train_encode.shape[1],  activation='elu'))
        model.compile(loss=self.params['loss'], optimizer = self.params['optimizer'])
        history = model.fit(X_train_encode, X_train_encode, batch_size=self.params['batch_size'], \
                            epochs=self.params['epochs'], verbose=self.params['verbose'], 
                            validation_data=(X_validate_encode, X_validate_encode))
        encoder = Model(model.input, model.get_layer('bottleneck').output)
        self.model = model
        self.encoder = encoder
        self.history = history
        #average the bottom epochs result
        bottom_vals_loss = history.history[metric][-bottom_epochs:]
        average_val_loss = sum(bottom_vals_loss)/len(bottom_vals_loss)
        self.validation_history = average_val_loss
    def get_encoder(self, X_encode):
        X_encoded = self.encoder.predict(X_encode)  # bottleneck representation
        return X_encode
    def get_reconstruction(self, X_encode):
        X_decoded = self.model.predict(X_encode)
        return X_decoded
    def save(self, folder, version = 1):
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_model(self.model,\
                   '{folder}/{symbol}_{version}.h5'.format(folder = folder, symbol = self.symbol_key, version=version), \
                   overwrite=True, include_optimizer=True)
#         self.model.save('{folder}/{symbol}_{version}.h5'.format(folder = folder, symbol = self.symbol_key, \
#                                                                 version=version))
    def load(self, folder, version = 1):
        self.model = load_model('{folder}/{symbol}_{version}.h5'.format(folder = folder, symbol = self.symbol_key, version=version))
        print(r'Model loaded from {folder}/{symbol}_{version}.h5'.format(folder = folder, symbol = self.symbol_key, version=version))
        return model
    
def train_test_split_scale(dataset, x_indices, y_indices, train_size = 0.8):
    """
    The data prep procedure
    -------------------------------
    dataset : np.array, input data
    x_indices: range(i,j), e.g. range(0,29), range(0, dataset.shape[1]-1)
    y_indices:range(i,j) or int, e.g. 29, dataset.shape[1]-1
    timesteps: int, lookback steps, 3rd dimension of tf tensor
    """
    TRAIN_SIZE = train_size
    TEST_SIZE = 1 - train_size
    # Splitting the datasets into training and testing data.
    train_data, test_data = train_test_split(dataset, train_size=TRAIN_SIZE, test_size=TEST_SIZE, shuffle=False)
    # Output the train and test data size
    print(f"Train and Test Size {len(train_data)}, {len(test_data)}")
    scaled_train_data, scaler = normalize(train_data, x_indices, y_indices, inv = False, scaler = None, norm_range = (0,1))
    scaled_test_data, _ = normalize(test_data, x_indices, y_indices, inv = False, scaler = scaler, norm_range = (0,1))
 
    X_train_encode, y_train_encode = generate_tf_data(scaled_train_data, x_indices, y_indices, timesteps = None)
    X_test_encode, y_test_encode = generate_tf_data(scaled_test_data, x_indices, y_indices, timesteps = None)
    return X_train_encode, y_train_encode, X_test_encode, y_test_encode, scaler
    
    
def train_split_scale_encode(dataset, x_indices, y_indices, sae_model):
    #feature engineering entry point, every single step in feature engineering is performed here
    pass
    
class Simple_LSTM_Model():
    def __init__(self, key, time_step=30, batch_size = 512, epochs = 400, verbose = 1, \
                opt = tf.keras.optimizers.Adam(learning_rate=0.0005), \
                loss_fn = 'mean_squared_error', metrics = ['mse'], if_binary = False):
        self.symbol_key = key
        self.params = {}
        self.params['time_step'] = time_step
        self.params['batch_size'] = batch_size
        self.params['epochs'] = epochs
        self.params['verbose'] = verbose
        self.params['optimizer'] = opt
        self.params['loss'] = loss_fn
        self.params['mertics'] = metrics
        self.params['if_binary'] = if_binary
    def fit(self, X_train, y_train, layers=[100, 1], earlystopping = True, patience = 50, \
            validation_split = 0.15, shuffle = True):
        model = Sequential()
        # Add the first layer with 100 neurons and inputs shape for 1st set, return sequence set False as no additional
        #LSTM Layer to input
        model.add(LSTM(units=layers[0], input_shape = X_train.shape[-2:], return_sequences=False)) 
        # Add the output layer
        if self.params['if_binary'] == True:
            model.add(Dense(1, activation = 'sigmoid'))
        else:
            model.add(Dense(layers[1]))
        model.compile(loss=self.params['loss'], optimizer=self.params['optimizer'], metrics = self.params['mertics'])
        self.model = model
        #early stopping
        if earlystopping == True:
            self.history = self.model.fit(X_train, y_train, verbose=self.params['verbose'], \
                                          epochs=self.params['epochs'], \
                                          batch_size=self.params['batch_size'], \
                                          validation_split=validation_split, \
                                          shuffle=shuffle, \
                                          callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]
                                         )
        else:
            self.history = self.model.fit(X_train, y_train, verbose=self.params['verbose'], epochs=self.params['epochs'], \
                                          batch_size=self.params['batch_size'], validation_split=validation_split, shuffle=shuffle)
        print('Model fitted successfully')
        self.model.summary()
    def predict(self, X_test):
        result = self.model.predict(X_test, verbose=self.params['verbose'])
        return result

class LSTMModel():
    def __init__(self, key, time_step=30, batch_size = 512, epochs = 400, verbose = 1, \
                opt = tf.keras.optimizers.Adam(learning_rate=0.0005),\
                loss_fn = 'mean_squared_error', metrics = ['mse'], if_binary = False):#batch_size=128
        self.symbol_key = key
        self.params = {}
        self.params['time_step'] = time_step
        self.params['batch_size'] = batch_size
        self.params['epochs'] = epochs
        self.params['verbose'] = verbose
        self.params['optimizer'] = opt
        self.params['loss'] = loss_fn
        self.params['mertics'] = metrics
        self.params['if_binary'] = if_binary
    def fit(self, X_train, y_train, layers=[256, 256, 64, 1], dropout_rate=0.5, earlystopping = True, patience = 50, \
            validation_split = 0.15, shuffle = True):
        model = Sequential()
        # Add first layer with dropout regularisation with 100 neurons and inputs shape for 1st set
        reg = L1L2(l1=0.01, l2=0.01)
        model.add(LSTM(units=layers[0], input_shape = X_train.shape[-2:], return_sequences=True, bias_regularizer = reg)) 
        model.add(Dropout(0.5))#0.5, 0.01
        # Add second layer with dropout
        model.add(LSTM(units=layers[1], return_sequences=False, bias_regularizer = reg))
        model.add(Dropout(0.5))#0.5, 0.01
        # Add a Dense layer
        model.add(Dense(layers[2],  activation = 'relu'))
        # Add the output layer
        if self.params['if_binary'] == True:
            model.add(Dense(layers[3], activation = 'sigmoid'))
        else:
            model.add(Dense(layers[3]))
        model.compile(loss=self.params['loss'], optimizer=self.params['optimizer'], metrics = self.params['mertics'])
        self.model = model
        
        #early stopping
        if earlystopping == True:
            self.history = self.model.fit(X_train, y_train, verbose=self.params['verbose'], \
                                          epochs=self.params['epochs'], \
                                          batch_size=self.params['batch_size'], \
                                          validation_split=validation_split, \
                                          shuffle=shuffle, \
                                          callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]
                                         )
        else:
            self.history = self.model.fit(X_train, y_train, verbose=self.params['verbose'], epochs=self.params['epochs'], \
                                          batch_size=self.params['batch_size'], validation_split=0.15, shuffle=False)
        print('Model fitted successfully')
        self.model.summary()
    def predict(self, X_test):
        result = self.model.predict(X_test, verbose=self.params['verbose'])
        #self.results = pd.DataFrame(result, columns=['yhat_test'])
        return result
    def save(self, folder, version = 1):
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_model(self.model, \
                   '{folder}/{symbol}_{version}.h5'.format(folder = folder, symbol = self.symbol_key, version=version), \
                   overwrite=True, include_optimizer=True)
#         self.model.save('{folder}/{symbol}_{version}.h5'.format(folder = folder, symbol = self.symbol_key, \
#                                                                 version=version))
    def load(self, folder, version = 1):
        self.model = load_model('{folder}/{symbol}_{version}.h5'.format(folder = folder, symbol = self.symbol_key, \
                                                                        version=version))
        print(r'Model loaded from {folder}/{symbol}_{version}.h5'.format(folder = folder, symbol = self.symbol_key, \
                                                                         version=version))
        return model
#Learn rate: 0.0100 Dropout: 0.25 Epoch: 100 Training error: 0.0047 Testing error: 0.0060
class BidirectionalModel():
    def __init__(self, key, time_step=30, batch_size = 64, epochs = 400, verbose = 1, \
                opt = tf.keras.optimizers.RMSprop(learning_rate=0.01), loss_fn = 'mean_squared_error'):#batch_size=128
        self.symbol_key = key
        self.params = {}
        self.params['time_step'] = time_step
        self.params['batch_size'] = batch_size
        self.params['epochs'] = epochs
        self.params['verbose'] = verbose
        self.params['optimizer'] = opt
        self.params['loss'] = loss_fn
    def fit(self, X_train, y_train, layers=[256, 256, 64, 1], dropout_rate=0.25):
        model = Sequential()
        model.add(Bidirectional(LSTM(X_train.shape[1],return_sequences=False),input_shape=(X_train.shape[-2:])))
        model.add(Dense(X_train.shape[1]))
        model.add(Dropout(dropout_rate))
        model.add(Dense(y_train.shape[1],activation='relu'))
        model.compile(loss=self.params['loss'],optimizer=self.params['optimizer'])
        self.model = model
        self.history = self.model.fit(X_train, y_train, verbose=self.params['verbose'], epochs=self.params['epochs'], \
                                      batch_size=self.params['batch_size'], validation_split=0.15, shuffle=False)
        print('Model fitted successfully')
        self.model.summary()
    def predict(self, X_test):
        result = self.model.predict(X_test, verbose=self.params['verbose'])
        return result
    def save(self, folder, version = 'bidirectional'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_model(self.model, \
                   '{folder}/{symbol}_{version}.h5'.format(folder = folder, symbol = self.symbol_key, version=version), \
                   overwrite=True, include_optimizer=True)
#         self.model.save('{folder}/{symbol}_{version}.h5'.format(folder = folder, symbol = self.symbol_key, \
#                                                                 version=version))
    def load(self, folder, version = 1):
        self.model = load_model('{folder}/{symbol}_{version}.h5'.format(folder = folder, symbol = self.symbol_key, \
                                                                        version=version))
        print(r'Model loaded from {folder}/{symbol}_{version}.h5'.format(folder = folder, symbol = self.symbol_key, \
                                                                         version=version))
        return model

def hit_ratio(y_test_predict, y_test, multiple_ys = True):
    if multiple_ys == True:
        assert y_test_predict.shape == y_test.shape, 'The shape of y_test_predict and y_test are not aligned!'
        observation_number, feature_number = y_test.shape[0], y_test.shape[1]
        Ns = [observation_number for j in range(feature_number)]
        Hs = [0 for j in range(feature_number)]
        for j in range(feature_number):
            H = 0
            for i in range(observation_number):
                if y_test_predict[i,j]*y_test[i,j] > 0:
                    H += 1
            Hs[j] = H
        HRs = []
        print(len(Hs))
        print(len(Ns))
        for x in range(len(Hs)):
            HRs.append(Hs[x]/Ns[x])
        return HRs
    else:
        assert len(y_test_predict) == len(y_test), 'lengths of y_test_predict and y_test are not aligned!'
        y_test_predict_lst = list(y_test_predict)
        y_test_lst = list(y_test)
        l_len = len(y_test_predict_lst)
        N = l_len - 1
        H = 0
        for i in range(l_len):
            if y_test_predict_lst[i]*y_test_lst[i] > 0:
                H += 1
        hr = float(H/N)
        return hr
    
def fit_som(df, size = 15,  sigma=1.5, random_seed=1):
    #train steps
    size = 15
    dataset = df.to_numpy()
    #print(dataset.shape)
    som = MiniSom(size, size, len(dataset[0]),
                  neighborhood_function='gaussian', sigma=sigma,
                  random_seed=random_seed)
    som.pca_weights_init(dataset)
    som.train_random(dataset, 10000, verbose=True)
    return som

def plt_som(som, feature_names, size = 15):
    W = som.get_weights()
    plt.figure(figsize=(10, 10))
    for i, f in enumerate(feature_names):
        plt.subplot(5, 5, i+1)
        plt.title(f)
        plt.pcolor(W[:,:,i].T, cmap='coolwarm')
        plt.xticks(np.arange(size+1))
        plt.yticks(np.arange(size+1))
    plt.tight_layout()
    plt.show()
    
def som_feature_selection(W, labels, target_index = 0, a = 0.04):
    """ Performs feature selection based on a self organised map trained with the desired variables

    INPUTS: W = numpy array, the weights of the map (X*Y*N) where X = map's rows, Y = map's columns, N = number of variables
            labels = list, holds the names of the variables in same order as in W
            target_index = int, the position of the target variable in W and labels
            a = float, an arbitary parameter in which the selection depends, values between 0.03 and 0.06 work well

    OUTPUTS: selected_labels = list of strings, holds the names of the selected features in order of selection
             target_name = string, the name of the target variable so that user is sure he gave the correct input
    """


    W_2d = np.reshape(W, (W.shape[0]*W.shape[1], W.shape[2])) #reshapes W into MxN assuming M neurons and N features
    target_name = labels[target_index]


    Rand_feat = np.random.uniform(low=0, high=1, size=(W_2d.shape[0], W_2d.shape[1] - 1)) # create N -1 random features
    W_with_rand = np.concatenate((W_2d,Rand_feat), axis=1) # add them to the N regular ones
    W_normed = (W_with_rand - W_with_rand.min(0)) / W_with_rand.ptp(0) # normalize each feature between 0 and 1

    Target_feat = W_normed[:,target_index] # column of target feature

    # Two conditions to check against a
    Check_matrix1 = abs(np.vstack(Target_feat) - W_normed)
    Check_matrix2 = abs(np.vstack(Target_feat) + W_normed - 1)
    S = np.logical_or(Check_matrix1 <= a, Check_matrix2 <= a).astype(int) # applie "or" element-wise in two matrices

    S[:,target_index] = 0 #ignore the target feature so that it is not picked

    selected_labels = []
    while True:

        S2 = np.sum(S, axis=0) # add all rows for each column (feature)

        if not np.any(S2 > 0): # if all features add to 0 kill
            break

        selected_feature_index = np.argmax(S2) # feature with the highest sum gets selected first

        if selected_feature_index > (S.shape[1] - (Rand_feat.shape[1] + 1)): # if random feature is selected kill
            break


        selected_labels.append(labels[selected_feature_index])

        # delete all rows where selected feature evaluates to 1, thus avoid selecting complementary features
        rows_to_delete = np.where(S[:,selected_feature_index] == 1)
        S[rows_to_delete, :] = 0
#     selected_labels = [label for i, label in enumerate(labels) if i in feature_indeces]
    return selected_labels, target_name

def single_run(key, step_dict, X_train, X_test, y_train, y_test, \
               opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07), \
               loss_fn = tf.keras.losses.MeanSquaredError(), \
               metrics = ['mse'], if_binary = False):
    if step_dict['sae'] == True:
        sae_model = StackedAutoencoderModel(key, verbose=0)
        sae_model.fit(X_train, X_test, target_feature_number=4)
        X_train = sae_model.get_encoder(X_train)
        X_test = sae_model.get_encoder(X_test)
    X_train_shaped, y_train_shaped = multivariate_data(X_train, y_train, timesteps=step_dict['timestep'])
    X_test_shaped, y_test_shaped = multivariate_data(X_test, y_test, timesteps=step_dict['timestep'])
    if step_dict['lstm'] == '1layer':
        #print(metrics)
        model = Simple_LSTM_Model(key, \
                                  batch_size = 512, epochs = 500, verbose = 0, \
                                  opt = opt, loss_fn = loss_fn, metrics = metrics, if_binary = if_binary)
    elif step_dict['lstm'] == '2layer':
        model = LSTMModel(key, time_step = timesteps, \
                          batch_size = 512, epochs = 500, verbose = 0, \
                          opt = opt, loss_fn = loss_fn, metrics = metrics, if_binary = if_binary)

    model.fit(X_train_shaped, y_train_shaped)
    return model, model.model.evaluate(X_test_shaped, y_test_shaped, verbose=1)[1]

def bulk_run(key, X_train, X_test, y_train, y_test, \
    timsteps_lst=[5, 10, 30], \
    algorithm_lst=['1layer LSTM', '2layer LSTM', 'sae + 1layer LSTM', 'sae + 2layer LSTM'], \
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07), \
    loss_fn = tf.keras.losses.MeanSquaredError(),\
    metrics = ['mse'], if_binary = False):
    step_dict = {}
    o_lst = []
    m_lst = []
    for timstep in timsteps_lst:
        step_dict['timestep'] = timstep
        for algorithm in algorithm_lst:
            if algorithm == '1layer LSTM':
                step_dict['sae'] = False
                step_dict['lstm'] = '1layer'
            if algorithm == '2layer LSTM':
                step_dict['sae'] = False
                step_dict['lstm'] = '2layer' 
            if algorithm == 'sae + 1layer LSTM':
                step_dict['sae'] = True
                step_dict['lstm'] = '1layer' 
            if algorithm == 'sae + 2layer LSTM':
                step_dict['sae'] = True
                step_dict['lstm'] = '2layer' 
            m, o = single_run(key, step_dict, X_train, X_test, y_train, y_test, \
                              opt = opt, loss_fn = loss_fn, metrics = metrics, if_binary = if_binary)
            o_lst.append({'timestep':timstep, 'algorithm':algorithm, metrics[0]: o})
            m_lst.append({'timestep':timstep, 'algorithm':algorithm, 'model': m})
    return m_lst, o_lst