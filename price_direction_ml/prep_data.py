from sklearn.model_selection._split import _BaseKFold, StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff()
import matplotlib as mpl
from scipy import interp
import yfinance as yf
from utils import *
from technical_indicators import *

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

#CONST
CLASSIFICATION_MODELS = ['logit', 'nb', 'svm', 'knn', 'rf']
CLASSFICATION_FUNCS = [LogisticRegression, GaussianNB, SVC, KNeighborsClassifier, RandomForestClassifier]
CLASSFICATION_PARAMS = [{'penalty':'l2'}, {'priors' : None}, {'C' : 1.0}, {'n_neighbors': 5}, {'max_depth' : 2, 'n_estimators' : 100}]
cls_name_funcs_dict = dict(zip(CLASSIFICATION_MODELS, CLASSFICATION_FUNCS))
cls_name_params_dict = dict(zip(CLASSIFICATION_MODELS, CLASSFICATION_PARAMS))

class AssetData:

    def __init__(self, symbol, base_symbol = '^GSPC', start_date = None, end_date = None, \
        if_bin = False, y_column = 'direction', y_return = 'return', test_size = 0.2, shuffle = False):
        #
        """
        construct the assetdata object:
        -------------------------------
        symbol : string, the asset code in yfinance
        base_symbol : string, placeholder to retrieve index data's features, it's not used.
        start_date : string, argument start of yfinance.download
        end_date : string, argument end of yfinance.download
        if_bin : boolean, whether to binarize the feature
        y_column : string, the naming convention for target variable
        y_return : string, the naming convention for actual return
        test_size : int,  test_size of train_test_split
        shuffle : boolean, shuffle of train_test_split
        -------------------------------
        attribute
        self.symbol : string, the asset code in yfinance
        self.base_symbol : string, placeholder to retrieve index data's features
        self.start_date : string, argument start of yfinance.download
        self.end_date : string, argument end of yfinance.download
        self.if_bin : string, placeholder for further use the binarized flag of features
        self.df : pd.DataFrame, dumped dataset after prep_data
        self.X : pd.DataFrame, X features
        self.y : pd.Series, y feature
        self.X_train : pd.DataFrame, X features in train dataset
        self.X_test : pd.DataFrame, X features in test dataset
        self.y_train : pd.DataFrame, y feature in train dataset
        self.y_test : pd.DataFrame, y feature in test dataset
        """
        #
        self.symbol = symbol
        self.base_symbol = base_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.if_bin = if_bin
        self.df, _x_columns, _y_columns, _y_columns_dict =  AssetData.prep_data(self.symbol, base_symbol = self.base_symbol, \
            start_date = self.start_date, end_date = self.end_date, if_bin = self.if_bin)
        self.X = self.df[_x_columns]
        assert y_column in _y_columns_dict.keys()
        assert y_return in _y_columns_dict.keys()
        self.y = self.df[_y_columns_dict[y_column]]
        self.y_return = self.df[_y_columns_dict[y_return]]
        #split the train test dataset
        self.X_train, self.X_test, self.y_train, self.y_test = AssetData.train_test_split(self.X, self.y, test_size=test_size, shuffle=shuffle)
        #
        self.clf_dict = AssetData.models_fit(self.X_train, self.y_train, models = cls_name_funcs_dict, params_lst = cls_name_params_dict)
        self.y_train_predict_dict = AssetData.models_predict(self.clf_dict, self.X_train)
        self.y_test_predict_dict = AssetData.models_predict(self.clf_dict, self.X_test)
        self.y_train_predict_df = pd.DataFrame(self.y_train_predict_dict)
        self.y_test_predict_df = pd.DataFrame(self.y_test_predict_dict)
        self.y_train_predict_proba_dict = AssetData.models_predict_proba(self.clf_dict, self.X_train, exclude_lst = ['svm'])
        self.y_test_predict_proba_dict = AssetData.models_predict_proba(self.clf_dict, self.X_test, exclude_lst = ['svm'])

        self.train_confusion_matrix_dict = AssetData.confusion_matrix_dict(self.y_train, self.y_train_predict_dict)
        self.test_confusion_matrix_dict = AssetData.confusion_matrix_dict(self.y_test, self.y_test_predict_dict)

        self.train_auc_plot_dict = AssetData.auc_plot_bulk(self.y_train, self.y_train_predict_proba_dict, title = 'ROC Curve', traintest_str = 'train')
        self.test_auc_plot_dict = AssetData.auc_plot_bulk(self.y_test, self.y_test_predict_proba_dict, title = 'ROC Curve', traintest_str = 'test')

    @staticmethod
    def train_test_split(X, y, test_size = 0.2, shuffle = True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def models_fit(X_train, y_train, models = cls_name_funcs_dict, params_lst = cls_name_params_dict):
        '''
        models : dict, {'logit' : LogisticRegression, 'nb' : GaussianNB, 'svm' : SVC, 'knn' : KNeighborsClassifier, 'rf' : RandomForestClassifier}
        params_lst : dict, {'logit' : [], }
        '''
        clf_dict = {}
        for model_name in models.keys():
            clf_dict[model_name] = AssetData._model_fit(X_train, y_train, model_name, models[model_name], params = cls_name_params_dict[model_name])
        return clf_dict
    
    @staticmethod 
    def models_predict(clf_dict, X):
        y_predict_dict = {}
        for model_name, clf in clf_dict.items():
            y_predict_dict['y_'+model_name] = clf.predict(X)
        #df_predict = pd.DataFrame(y_predict_dict)
        #self.y_predict = df_predict
        return y_predict_dict

    @staticmethod
    def models_predict_proba(clf_dict, X, exclude_lst = ['svm']):
        y_predict_proba_dict = {}
        for model_name, clf in clf_dict.items():
            if model_name not in exclude_lst:
                y_predict_proba = clf.predict_proba(X)
                y_predict_df = pd.DataFrame(y_predict_proba, columns = clf.classes_)
                y_predict_proba_dict['y_'+model_name] = y_predict_df 
        #df_predict_proba = pd.DataFrame(y_predict_proba_dict)
        #self.y_predict_proba = df_predict_proba
        return y_predict_proba_dict#y_predict_proba_dict,df_predict_proba

    @staticmethod
    def _model_fit(X_train, y_train, model_name, model_func, params = {}, f_scale = True):
        pipeline_steps = []
        if f_scale == True:
            pipeline_steps.append(('scale', StandardScaler()))
        clf_estimator = model_func()
        clf_estimator.set_params(**params)
        pipeline_steps.append((model_name, clf_estimator))
        clf = Pipeline(pipeline_steps)
        clf.fit(X_train, y_train)
        return clf

    @staticmethod
    def confusion_matrix_dict(y, y_pred_dict):
        confusion_matrix_dict = {}
        for y_pred_name, y_pred in y_pred_dict.items():
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            ac = (tp + tn) / (tn + fp + fn + tp)
            p, r= tp / (tp + fp), tp / (tp + fn)
            f1 = (2 * p * r) / (p + r)
            temp_df = pd.DataFrame({'model_name': [y_pred_name], 'tn':[tn], 'fp':[fp], 'fn':[fn], 'tp':[tp], \
                'accuracy' : [ac], \
                'precision':[p], 'recall':[r], 'f1':[f1]})
            temp_df.set_index('model_name', inplace=True)
            confusion_matrix_dict[y_pred_name] = temp_df
        return confusion_matrix_dict

    @staticmethod
    def auc_plot_bulk(y, y_predict_proba_dict, title = 'ROC Curve', traintest_str = 'test'):
        auc_plot_dict = {}
        for model_name, y_predict_proba in y_predict_proba_dict.items():
            auc_plot_dict[model_name] = AssetData.auc_plot(y, y_predict_proba, model_name, title = title +\
             ' using ' +  model_name + ' on ' + traintest_str + ' dataset')
        return auc_plot_dict

    @staticmethod
    def auc_plot(y, y_predict_proba, model_name, title = None):
        y = np.asarray(y)
        f, ax = plt.subplots(figsize=(10,7))
        fpr, tpr, thresholds = roc_curve(y, np.asarray(y_predict_proba[1]))
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC (AUC = %0.2f) - %s' % (roc_auc, model_name))
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1,1))
        plt.close()
        return f

    @staticmethod
    def prep_data(symbol, base_symbol, start_date = None, end_date = None, if_bin = False):
        #
        """
        The data prep procedure
        -------------------------------
        symbol : string, the asset code in yfinance
        base_symbol : string, placeholder to retrieve index data's features, it's not used.
        """
        df = yf.download(symbol, start=start_date, end=end_date)
        df_base = yf.download(base_symbol, start=start_date, end = end_date)
        df_base['base_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df_base = create_lags(df_base, lags = 1, columns = ['base_return'])
        #df = df.join(df_base['base_return_lag_1'])
        del(df['Adj Close'])
            
        # #absolute value of features
        # df['ma5'] = df['Close'].rolling(5).mean()
        # df['ewma7'] = df['Close'].ewm(span = 7).mean()
        # df['ewma21'] = df['Close'].ewm(span = 21).mean()
        
        #log change of features
        df['return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        for window in [5, 13, 34, 55, 100]:
            df['mom'+str(window)] = df['Close'].pct_change(window)
            df['std'+str(window)] = df['return'].rolling(window=window, min_periods=window, center=False).std()
            df = create_lags(df, lags = 1, columns = ['mom'+str(window), 'std'+str(window)])
            df.drop(columns = ['mom'+str(window),'std'+str(window)], inplace = True)

            df['ma'+str(window)] = df['Close'].rolling(window).mean()
            df['ewma'+str(window)] = df['Close'].ewm(span = window).mean()
            df = create_lags(df, lags = 1, columns = ['ma'+str(window), 'ewma'+str(window)])
            df.drop(columns = ['ma'+str(window),'ewma'+str(window)], inplace = True)

        
        df['change_open'] = np.log(df['Open'] / df['Open'].shift(1))
        df['change_high'] = np.log(df['High'] / df['High'].shift(1))
        df['change_low'] = np.log(df['Low'] / df['Low'].shift(1))
        df['change_volume'] = np.log(df['Volume'] / df['Volume'].shift(1))
        
        
        #create lags
        df = create_lags(df, lags = 7, columns = ['return'])
        #df = create_lags(df, lags = 1, columns = ['Open', 'High', 'Low', 'Volume'])
        df = create_lags(df, lags = 1, columns = ['change_open', 'change_high', 'change_low', \
                                                  'change_volume'])


        #target label
        df['label_return'] = df['return']
        df['label_direction'] = (df['Close'] >= df.shift(1)['Close']).astype(int)
        #df['label_return'] = df['return'].shift(-4).rolling(5, min_periods = 0).sum()
        #df['label_direction'] = df['label_return'].apply(lambda x : 1 if x>=0 else 0)
        #df['label_direction'] = (df['Close'] >= df.shift(1)['Close']).astype(int)
        
        y_columns_dict = {}
        y_columns_dict['return'], y_columns_dict['direction']  = 'label_return', 'label_direction'
        df.drop(columns = ['change_volume_lag_1'], inplace = True)
        #df['label_direction'] = np.sign(df['return']).astype(int)
        df.drop(columns = ['return', 'Open', 'High', 'Low', 'Close', 'Volume', \
                           'change_open', 'change_high', 'change_low', \
                           'change_volume'], inplace = True) #'ma5', 'ewma7', 'ewma21'
        df.dropna(inplace=True)
        x_columns, x_bin_columns, y_columns  = [], [], []
        for column in df.columns:
            if 'label' in column:
                y_columns.append(column)
            else:
                x_columns.append(column)
        if if_bin == True:
            for column in x_columns:
                column_bin = column + '_bin'
                df[column_bin] = np.digitize(df[column], bins = [0])
                x_bin_columns.append(column_bin)
            return df[x_bin_columns + y_columns], x_bin_columns, y_columns, y_columns_dict
        return df[x_columns + y_columns], x_columns, y_columns, y_columns_dict