"""
    DATA8001 - Assignment 1 2021

    description: All Assignment 1 functions required to reproduce results

    version   1.0       ::  2021-03-02  ::  started version control

                                            ----------------------------------
                                                GENERIC FUNCTIONS
                                            ----------------------------------

                                            + data_etl
                                            + load_run_model

                                            ----------------------------------
                                                USER DEFINED FUNCTIONS
                                            ----------------------------------
                                            +
"""

##########################################################################################
##########################################################################################
#
#   IMPORTS
#
##########################################################################################
##########################################################################################

import re
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime as dt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

##########################################################################################
##########################################################################################
#
#   GENERIC FUNCTIONS
#
##########################################################################################
##########################################################################################

def data_etl(student_id):
    """
        Load original dataset and clean

        :param str student_id:
            The student_id to load the original dataset as provided in assignment zip file.

        :return: Processed (cleaned) pandas dataframe.
        :rtype: pandas.core.frame.DataFrame
    """
    try:

        print(f'cleaning data for {student_id} ...')

        #
        # 1. load the dataset
        #
        df_original = pd.read_csv(f'data/{student_id}_original.csv')

        #
        # 2. CLEAN THE DATA
        #
        # ALL: replace null values with 0
        df_original = df_original.fillna(0)
        # ALL: convert all columns letters to uppercase
        df_original = df_original.apply(lambda x: x.astype(str).str.upper())

        before_county = pd.value_counts(df_original[['county']].values.flatten())
        before_make = pd.value_counts(df_original[['make']].values.flatten())
        before_model = pd.value_counts(df_original[['model']].values.flatten())
        before_type = pd.value_counts(df_original[['type']].values.flatten())


        # purchase_date : get missing year from car_reg and convert dates to single format
        df_original['purchase_date'] = df_original.apply(lambda row: clean_dates(row['purchase_date'],row['car_reg']), axis=1)
        # year : create year (int) for each row - get year from purchase_date
        df_original.insert(loc=2, column='year', value=df_original['purchase_date'].str[6:].astype(int))
        # month: create month (int) for each row - get month from purchase_date
        df_original.insert(loc=3, column='month', value=df_original['purchase_date'].str[3:5].astype(int))
        # purchase_date : save as datetime having manipulated purchase_date, year, month, car_reg as strings
        df_original['purchase_date'] = pd.to_datetime(df_original['purchase_date'])
        # car_reg : county code = 'X', get missing county codes
        df_original['car_reg'] = df_original.apply(lambda row: clean_car_reg_county_code(row['car_reg'], row['county']), axis=1)
        # car_reg : year/period = 'XXX', get missing year and period from year and month
        df_original['car_reg'] = df_original.apply(lambda row: clean_car_reg_year_month(row['car_reg'], row['year'], row['month']), axis=1)
        # county : county = none, get missing county values from car_reg
        df_original['county'] = df_original.apply(lambda row: clean_county(row['county']), axis=1)
        # type : type = none, get type from make or model
        df_original['type'] = df_original.apply(lambda row: clean_type(row['make'],row['model'],row['type']), axis=1)
        # model : model = none, get model from make and remove type if exists leaving just model
        df_original['model'] = df_original.apply(lambda row: clean_model(row['make'],row['model']), axis=1)
        # make : remove make and/or model if exists leaving just make
        df_original['make'] = df_original['make'].apply(lambda row: clean_make(row))
        # colour : convert hex values (eg #FFFFFF) or values with tags (eg <colour>Silver</colour>) to colours (eg RED)
        df_original['colour'] = df_original['colour'].apply(lambda row: clean_colour(row))
        # tax_band : convert numeric tax bands (1,2,3,4) to alphanumeric (A,B,C,D)
        df_original['tax_band'] = df_original['tax_band'].apply(lambda row: clean_tax_band(row))
        # price : change price to float data type
        df_original['price'] = df_original['price'].astype(float)

        df_processed = df_original.copy()
        df_processed.to_csv(f'data/{student_id}_processed.csv', index=False)
        #
        # 3. return processed (clean) data
        #
        return (df_processed)

    except Exception as ex:
        raise Exception(f'data_etl :: {ex}')


def load_run_model(student_id, df_test):
    """
        Load a Linear Regression pickle model and run the ML model on unseen data.

        :param str student_id:
            The student_id to load the model file path of the pickled model.
        :param df_test pandas.core.frame.DataFrame:
            The test data to predict.

        :return: Model predictions and accuracy
        :rtype: list, float
    """
    try:

        print(f'loading and running the linear regression model for {student_id} ...')

        predictions = []
        accuracy = 0.0

        #
        # 1. load the pickled model
        #
        new_model_trans = pickle.load(open(f'model/{student_id}.pkl', 'rb'))
        #
        # 2. Pre-Process the unseen test data
        #
        df_pickle_test = df_test.copy()
        df_pickle_test[['make_model_tax_band']] = df_pickle_test['make'] + '-' + df_pickle_test['model'] + '-' + df_pickle_test['tax_band']
        input_features = ['make', 'model', 'tax_band', 'make_model_tax_band']
        output_feature = 'price'

        #
        # 3. run the model on the pre-processed data and return predictions and model accuracy
        #
        for feature in input_features:
            label_count = input_features.index(feature)
            df_pickle_test[feature] = new_model_trans.get_transformations()[label_count][feature].transform(
                df_pickle_test[feature])
        # make predictions
        y_pred = new_model_trans.get_model().predict(df_pickle_test[input_features])
        y = df_pickle_test[output_feature].values
        # what is the accuracy
        accuracy = r2_score(y_pred=y_pred, y_true=y)
        print(f'Logistic Regression R Squared Accuracy: Score = {accuracy:.2f}')

        return (y_pred, accuracy)

    except Exception as ex:
        raise Exception(f'load_run_model :: {ex}')


class Data8001():
    """
        Data8001 model & transformation class
    """

    #############################################
    #
    # Class Constructor
    #
    #############################################

    def __init__(self, transformations:list, model:object):
        """
            Initialse the objects.

            :param list transformations:
                A list of any data pre-processing transformations required
            :param object model:
                Model object
        """
        try:

            # validate inputs
            if transformations is None:
                transformations = []
            if type(transformations) != list:
                raise Exception('invalid transformations object type')
            if model is None:
                raise Exception('invalid model, cannot be none')

            # set the class variables
            self._transformations = transformations
            self._model = model

            print('data8001 initialised ...')

        except Exception as ex:
            raise Exception(f'Data8001 Constructor :: {ex}')

    #############################################
    #
    # Class Properties
    #
    #############################################

    def get_transformations(self) -> list:
        """
            Get the model transformations

            :return: A list of model transformations
            :rtype: list
        """
        return self._transformations

    def get_model(self) -> object:
        """
            Get the model object

            :return: A model object
            :rtype: object
        """
        return self._model

##########################################################################################
##########################################################################################
#
#   USER DEFINED FUNCTIONS
#
#   provide your custom functions below this box.
#
##########################################################################################
##########################################################################################

#%%
#######################################################################################################################
########################################### Modelling Functions #######################################################
#######################################################################################################################

def show_dataset_plots(df_copy):
    # create boxplot
    df_skewed = df_copy['price']
    fig = plt.figure(figsize =(15, 3))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='x', colors='green')
    plt.boxplot(df_skewed, vert=False)
    plt.xlabel('Prices', fontsize = 16)
    plt.title('Car prices', fontsize=16)
    plt.show()
    # create histogram
    counts, bins = np.histogram(df_skewed)
    plt.hist(bins[:-1], bins, weights=counts, color = "magenta")
    plt.ylabel('Frequency', fontsize = 12)
    plt.xlabel('Prices', fontsize = 12)
    plt.title('Car prices', fontsize=12)
    all_data_points = df_copy['price'].count().item()
    plt.show()
    return

def remove_outliers(df_copy):
    maxprice = (np.median(df_copy['price']) + 2 * np.std(df_copy['price'])).item()
    df_normal = df_copy[df_copy['price'] <= maxprice]
    return(df_normal)

def create_model(df_train, df_test, input_features, output_feature):
    # create X & y np arrays
    X_train = df_train[input_features].values
    y_train = df_train[[output_feature]].values
    X_test = df_test[input_features].values
    y_test = df_test[[output_feature]].values
    # create a linear regression model
    lin_model = LinearRegression()
    # fit the model
    lin_model.fit(X=X_train, y=y_train)
    return(lin_model,X_test,y_test)

def test_model(lin_model,X_test,y_test):
    # generate predictions
    y_pred = lin_model.predict(X=X_test)
    # test the model predictions
    mse = int(mean_squared_error(y_pred=y_pred, y_true=y_test))
    rmse = int(math.sqrt(mse))
    r_sq = round(r2_score(y_pred=y_pred, y_true=y_test),3)
    return (y_pred, mse, rmse, r_sq)

def label_train_transform_features(df_train, df_test, input_feature):
    label = 'label_' + input_feature
    # create the encoder object
    label = LabelEncoder()
    transformations = []
    transformations += [{input_feature: label}]
    # train the encoder on training data only
    label.fit(df_train[input_feature].values.reshape(-1, 1))
    # transform the training and test data
    df_train[input_feature] = label.transform(df_train[input_feature].values.reshape(-1, 1))
    df_test[input_feature] = label.transform(df_test[input_feature].values.reshape(-1, 1))
    return(df_train, df_test, transformations)

def scaler_train_transform_features(df_train, df_test, input_feature):
    scaler = 'scaler_' + input_feature
    # create the encoder object
    scaler = StandardScaler()
    transformations = []
    transformations += [{input_feature: scaler}]
    # train the encoder on training data only
    scaler.fit(df_train[input_feature].values.reshape(-1, 1))
    # transform the training and test data
    df_train[input_feature] = scaler.transform(df_train[input_feature].values.reshape(-1, 1))
    df_test[input_feature] = scaler.transform(df_test[input_feature].values.reshape(-1, 1))
    return(df_train, df_test, transformations)

def train_transform_model(df, input_features, label_on, scaler_on):
    output_feature = 'price'
    print('y variable (dependent) - output_feature =',output_feature)
    print('X variables (independent) - input_features =',input_features)
    print('Label encoding =', label_on, '\ Scaler encoding =', scaler_on, '\n')
    print('Splitting dataframe into training and test dataframes 80:20...')
    df_train, df_test = train_test_split(df[input_features + [output_feature]].copy(), test_size=0.2,
                                         random_state=8001)
    print('Encoding all categorical features used in model to numeric labels...')
    transformations = []
    if label_on:
        for i in range(0,len(input_features)) :
            df_train, df_test, transformation = label_train_transform_features(df_train, df_test, input_features[i])
            transformations += transformation
    if scaler_on:
        for i in range(0, len(input_features)):
            df_train, df_test, transformation = scaler_train_transform_features(df_train, df_test, input_features[i])
            transformations += transformation
    print('Creating a linear regression model...')
    lin_model, X_test, y_test = create_model(df_train, df_test, input_features, output_feature)
    print('Testing model - calculating RMSE and RSq accuracy score...\n\nRESULTS')
    y_pred, mse, rmse, r_sq = test_model(lin_model, X_test, y_test)
    print('- Overall Model Accuracy :: RMSE :', rmse, 'RSq :', r_sq,'\n')
    # test individual feature contributions with accuracy scores
    get_best_features(df_train, df_test, input_features, output_feature)
    df_train.to_csv(f'data/processed.csv', index=False)
    return(lin_model, transformations, rmse, r_sq)

def get_best_features(df_train, df_test, input_features, output_feature):
    # test indiviual feature contributions
    print('- Accuracy by Feature :')
    features = []
    for feature in input_features:
        input_features = [feature]
        # create a linear regression model
        lin_model, X_test, y_test = create_model(df_train, df_test, input_features, output_feature)
        # test model - get RMSE and RSq accuracy score
        y_pred, mse, rmse, r_sq = test_model(lin_model, X_test, y_test)
        features.append({'feature': feature, 'rmse': rmse, 'r_sq': r_sq})
    features = sorted(features, key=lambda i: i['r_sq'], reverse=True)
    for feature in features:
        print('  RMSE: {1}, R Sq: {2} - {0}'.format(feature['feature'], feature['rmse'], feature['r_sq']))
    print('\n- Best Feature by Accuracy : RMSE: {1}, R Sq: {2} - {0}\n'.format(features[0]['feature'], features[0]['rmse'], features[0]['r_sq']))
    return

def compare_models(df,model2,input_features,output_feature):
    df_train, df_test = train_test_split(df[input_features + [output_feature]].copy(), test_size=0.2, random_state=8001)
    for i in range(0, len(input_features)):
        df_train, df_test, transformation = label_train_transform_features(df_train, df_test, input_features[i])
    lin_model,X_test,y_test = create_model(df_train, df_test, input_features, output_feature)
    y_pred = model2.predict(X=X_test)
    mse = int(mean_squared_error(y_pred=y_pred, y_true=y_test))
    rmse = int(math.sqrt(mse))
    r_sq = round(r2_score(y_pred=y_pred, y_true=y_test), 3)
    return (y_pred, mse, rmse, r_sq)

#%%
#######################################################################################################################
############################################ Cleaning Functions #######################################################
#######################################################################################################################

def clean_dates(date, reg):
    # purchase_date : get missing year from car_reg and convert dates to single format
    cleaned_date = ''
    if re.search("\s", date): # eg 01 Feb
        date = '20' + reg[:2] + ' ' + date # add car_reg year to date
        cleaned_date = dt.strptime(date, '%Y %d %b').strftime('%d %m %Y')
    elif re.search("[A-Z]", date): # eg 2018-Feb-27
        cleaned_date = dt.strptime(date, '%Y-%b-%d').strftime('%d %m %Y')
    else: # eg 2018-01-25
        cleaned_date = dt.strptime(date, '%Y-%m-%d').strftime('%d %m %Y')
    return cleaned_date


def clean_car_reg_county_code(reg, county):
    # car_reg : county code = 'X', get missing county codes
    county_codes_dict = {'D': 'DUBLIN', 'C': 'CORK', 'L': 'LIMERICK', 'W': 'WATERFORD', 'G': 'GALWAY'}
    for key, value in county_codes_dict.items():
        if county == value:
            code = key # assign code for given county
    if reg[4:5] == 'X':
        cleaned_reg = reg[:4] + '-' + code + reg[5:] # replace car_reg 'X' with county code
    else:
        cleaned_reg = reg
    return cleaned_reg


def clean_car_reg_year_month(reg, year, month):
    # car_reg : year/period = 'XXX', get missing year and period from year and month
    cleaned_reg = reg
    if 'XXX' in reg:
        if month < 7:
            cleaned_reg = str(year)[:2] + '1' + reg[4:]
        else:
            cleaned_reg = str(year)[:2] + '2' + reg[4:]
    return cleaned_reg


def clean_county(county):
    # county : county = none, get missing county values from car_reg
    cleaned_county = county
    county_codes = {'D': 'DUBLIN', 'C': 'CORK', 'L': 'LIMERICK', 'W': 'WATERFORD', 'G': 'GALWAY'}
    if county == '0':
        for key, value in county_codes.items():
            cleaned_county = value
    return cleaned_county


def clean_type(make, model, type):
    # type : type = none, get type from make or model
    #make = row[0]; model = row[1]; type = row[2]
    cleaned_type = type
    if (type == '0') & (model == '0'):
        cleaned_type = str(make).split(' (')[-1][:-1]
    elif (type == '0') & (model != '0'):
        cleaned_type = str(model).split(' : ')[-1]
    return cleaned_type


def clean_model(make, model):
    # model : model = none, get model from make and remove type if exists leaving just model
    cleaned_model = model
    if model == '0':
        cleaned_model = str(make).split(' (')[0].split(' : ')[1]
    elif ' : ' in model:
        cleaned_model = str(model).split(' : ')[0]
    return cleaned_model


def clean_make(make):
    # make : remove make and/or model if exists leaving just make
    cleaned_make = make
    if ' : ' in make:
        cleaned_make = str(make).split(' : ')[0]
    return cleaned_make


def clean_colour(colour):
    # colour : convert hex values (eg #FFFFFF) or values with tags (eg <colour>Silver</colour>) to colours (eg RED)
    colours = {'#FFA500': 'ORANGE', '#FFFFFF': 'WHITE', '#C0C0C0': 'SILVER', '#0000FF': 'BLUE', '#FF0000': 'RED'}
    cleaned_colour = colour
    if '>' in colour:
        cleaned_colour = colour.split('>')[1].split('<')[0]
    elif colour in colours.keys():
        for key, value in colours.items():
            cleaned_colour = value
    return cleaned_colour


def clean_tax_band(tax):
    # tax_band : convert numeric tax bands (1,2,3,4) to alphanumeric (A,B,C,D)
    tax_bands = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
    cleaned_tax = tax
    if tax.isdigit():
        cleaned_tax = tax_bands.get(tax)
    return cleaned_tax
