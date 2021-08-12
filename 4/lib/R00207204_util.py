"""
    DATA8001 - Assignment 2 2021
    
    description: All Assignment 2 functions required to reproduce results
    
    version   1.0       ::  2021-03-22  ::  started version control

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

import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') # Load English stop words
from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import MultinomialNB
import pickle



##########################################################################################
##########################################################################################
#
#   GENERIC FUNCTIONS
#
##########################################################################################
##########################################################################################

def data_etl(student_id):
    """
        Load original data from news files and clean
        
        :param str student_id:
            The student_id to unzip the news files.

        :return: Processed (cleaned) pandas dataframe.
        :rtype: pandas.core.frame.DataFrame
    """
    try:
        print(f'cleaning data for {student_id} ...')
        df_processed = None
        
        #
        # 1. TODO - load the news files into a dataframe with the columns: ['news_headline', 'news_article']
        #
        files = os.listdir('data/files/')
        all_records = []
        for file in files:
            record = open(f'data/files/{file}', 'r').read().split('\n')
            all_records.append(record[2:5])

        #
        # 2. clean the data
        #
        cleaned_records = [];
        # remove html tags
        tags = ['<CATEGORY>', '</CATEGORY>', '<HEADLINE>', '</HEADLINE>', '<ARTICLE>', '</ARTICLE>']
        for record in all_records:
            cleaned_record = []
            for item in record:
                for tag in tags:
                    if tag in item:  # print('found',tag)
                        item = item.replace(tag, '')
                cleaned_record.append(item)
            if cleaned_record[0] != '# DATA8001 - ASSIGNMENT 2':
                cleaned_records.append(cleaned_record)
        # remove '# DATA8001 - ASSIGNMENT 2' row



        #
        # 3. return processed (clean) data with columns: ['news_headline', 'news_article']
        #
        df_processed = pd.DataFrame.from_records(cleaned_records)
        df_processed.columns = ['news_category', 'news_headline', 'news_article']

        return (df_processed)
        
    except Exception as ex:
        raise Exception(f'data_etl :: {ex}')


def load_run_model(model_id, student_id, news_headline, news_article):
    """
        Load the specified student pickle model and use the ML model to predict a new category based on the news headline and article provided.

        :param str model_id:
            The model_id to load the model file path of the pickled model.
            ['model_1', 'model_2', 'model_3']
        :param str student_id:
            The student_id to load the model file path of the pickled model.
        :param str news_headline:
            The news headline
        :param str news_article:
            The news article
            
        :return: Model object and model predicted category (i.e., news category)
        :rtype: object, str
    """
    try:
        if model_id not in ['model_1', 'model_2', 'model_3']:
            raise Exception('invalid model id')
            
        print(f'Loading and running the {model_id} for {student_id}...')

        model = None
        news_category = ''

        #
        # 1. load the correct pickled model
        model_trans = pickle.load(open(f'model/{student_id}_{model_id}.pkl', 'rb'))
        transformations = model_trans.get_transformations()[0]
        model = model_trans.get_model()
        #
        # 2. Pre-Process the news_headline and news_article values if required
        group_method = 'Stem'
        df = pd.DataFrame({'news_article': [news_article], 'news_headline': [news_headline], 'news_category': ''})
        for i in range (2000):
            df.append({'news_article': [news_article], 'news_headline': [news_headline]}, ignore_index=True)
        df, cat_dic = pre_process_text(df, group_method)
        features = df['news_article'] # + df['news_headline']
        #
        # 3. run the model to predict the new category
        print('Transforming text to vectors...')
        features = transformations.transform(features).toarray()
        print('Predicting category...')
        #news_category = model.predict(features)
        predictions = model.predict(features)[0]
        cat_dic = {0 : 'WORLD', 1 : 'ENTERTAINMENT', 2 : 'SPORTS', 3 : 'POLITICS'}
        news_category = cat_dic[predictions]
        return (model, news_category)


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

def pre_process_text(df, group_method):
    print('Pre-processing data...')
    # average word count before stop words removal
    before_word_count = df['news_article'].str.split().str.len().mean().astype(int)
    # change all characters to lower case
    df['parsed_news_headline'] = df['news_headline'].str.lower()
    df['parsed_news_article'] = df['news_article'].str.lower()
    # remove stop words from news article and headline
    remove_stopwords(df)
    # remove punctuation characters, whitespace, digits, pronoun terminations
    cleaning_list = list("?:!.,;:%€$₹-/[]-*()\'\"")
    whitespace = "  "; cleaning_list.append(whitespace)
    digits = "\d+"; cleaning_list.append(digits)
    pronoun_terminations = "\'s"; cleaning_list.append(pronoun_terminations)
    for char in cleaning_list:
        df['parsed_news_headline'] = df['parsed_news_headline'].str.replace(char, '', regex=True)
        df['parsed_news_article'] = df['parsed_news_article'].str.replace(char, '', regex=True)
    # average word count before stop words removal
    # after_word_count = df['parsed_news_article'].str.split().str.len().mean().astype(int)
    # print('Mean word count', before_word_count, '-->', after_word_count)
    # lemmatise dataframe
    f = open('data/results.txt', 'a')
    f.write('PRE-PROCESSING WORD GROUP METHOD : ' + group_method + '\n' + '*****************************************************' + '\n\n')
    f.close()
    if group_method == 'Lemmatise':
        lemmatise(df)
    else:
        stem(df)
    # Category mapping
    df['news_category_code'] = df['news_category']
    cat_list = df['news_category'].drop_duplicates().tolist()
    cat_dic = {cat_list[i]: i for i in range(0, len(cat_list))}
    df = df.replace({'news_category_code':cat_dic})
    return(df, cat_dic)

def remove_stopwords(df):
    stop_words = list(stopwords.words('english'))
    # loop through stop words list for each word and remove word from news article and headline
    for word in stop_words:
        regex_stopword = r"\b" + word + r"\b"
        df['parsed_news_headline'] = df['parsed_news_headline'].str.replace(regex_stopword, '', regex=True)
        df['parsed_news_article'] = df['parsed_news_article'].str.replace(regex_stopword, '', regex=True)
    # remove double spaces created by stop word removal
    while df['parsed_news_headline'].str.contains('  ').any():
        df['parsed_news_headline'] = df['parsed_news_headline'].str.replace('  ', ' ', regex=True)
    while df['parsed_news_article'].str.contains('  ').any():
        df['parsed_news_article'] = df['parsed_news_article'].str.replace('  ', ' ', regex=True)
    return(df)

def lemmatise(df):
    # Saving the lemmatizer into an object
    wordnet_lemmatizer = WordNetLemmatizer()
    feature_list = ['parsed_news_headline','parsed_news_article']
    for feature in feature_list:
        lemmatized_text_list = []
        for row in range(0, len(df)):
            # Create an empty list containing lemmatized words
            lemmatized_list = []
            # Save the text and its words into an object
            text = df.loc[row][feature]
            text_words = text.split(" ")
            # Iterate through every word to lemmatize
            for word in text_words:
                lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
            # Join the list
            lemmatized_text = " ".join(lemmatized_list)
            # Append to the list containing the texts
            lemmatized_text_list.append(lemmatized_text)
        df[feature] = lemmatized_text_list
    return(df)

def stem(df):
    # Saving the Lancaster stemmer into an object
    lancaster = LancasterStemmer()
    feature_list = ['parsed_news_headline', 'parsed_news_article']
    for feature in feature_list:
        stemmed_text_list = []
        for row in range(0, len(df)):
            # Create an empty list containing stemmed words
            stemmed_list = []
            # Save the text and its words into an object
            text = df.loc[row][feature]
            text_words = text.split(" ")
            # Iterate through every word to lemmatize
            for word in text_words:
                stemmed_list.append(lancaster.stem(word))
            # Join the list
            stemmed_text = " ".join(stemmed_list)
            # Append to the list containing the texts
            stemmed_text_list.append(stemmed_text)
        df[feature] = stemmed_text_list
    return(df)

def explore(df):
    # how many articles of each category?
    df['word_count'] = df['news_article'].str.split().str.len()
    df_by_cat = df.groupby(['news_category'])['word_count']
    print('Summary Statistics :')
    print(df_by_cat.describe(),'\n')
    # histogram of each category by news_article word count
    df['word_count'].hist(by=df['news_category'])
    # boxplot of each category by news_article word count
    colors = ['b', 'y', 'm', 'c', 'g', 'b', 'r', 'k', ]
    bp_dict = df.boxplot(by="news_category", column=['word_count'],layout=(4,1),figsize=(10,15), return_type='both', patch_artist = True, showfliers=False)
    for row_key, (ax,row) in bp_dict.iteritems():
        ax.set_xlabel('')
        for i,box in enumerate(row['boxes']):
            box.set_facecolor(colors[i])
    return

def vectorise(df):
    # TfidfVectorizer will tokenize documents, learn the vocabulary and inverse document frequency weightings, in preparation for encoding text
    # Set TfidfVectorizer parameters
    ngram_range = (1,2) # ngram_range: for both unigrams and bigrams
    #min_df = 10 # min_df: ignore terms with document frequency lower than threshold
    #max_df = 1. # max_df: ignore terms with document frequency higher than threshold max_df=max_df, min_df=min_df,
    max_features = 300 # max_features: build vocabulary with only top max_features ordered by term frequency across the corpus
    # Scale text when representing it as TF-IDF features with the argument 'norm'
    tfidf = TfidfVectorizer(encoding='utf-8', ngram_range=ngram_range, stop_words=None, lowercase=False, max_features=max_features, norm='l2', sublinear_tf=True)
    return(tfidf)

def plot_c_matrix(df, labels_test, model):
    # Confusion matrix
    aux_df = df[['news_category', 'news_category_code']].drop_duplicates().sort_values('news_category_code')
    conf_matrix = confusion_matrix(labels_test, model)
    plt.figure(figsize=(12.8,6))
    sns.heatmap(conf_matrix, annot=True, xticklabels=aux_df['news_category'].values, yticklabels=aux_df['news_category'].values, cmap="Blues")
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion matrix')
    plt.show()
    return

def show_transform_summary(features_train, features_test, labels_train, labels_test, tfidf, cat_dic):
    # Show Transformation Summary Statistics
    print('Transformation Summary Statistics'.upper(),'\n')
    print('1. Train Features Shape', features_train.shape,'\n')
    print('2. Test Features Shape', features_test.shape,'\n')
    # each word is assigned unique integer index in the output vector
    print('3. Vocabulary : Index (first 5 only)', list(tfidf.vocabulary_.items())[:5],'\n')
    # inverse document frequencies are calculated for each word in the vocabulary
    print('4. IDF word scores (first 5 only)\n', tfidf.idf_[:5])
    # Chi squared test shows unigrams and bigrams most correlated with each category - compare results to expectations
    print('\n5. Chi squared test for unigrams and bigrams most correlated with each category - compare results to category expectations')
    for Article, category_id in sorted(cat_dic.items()):
        features_chi2 = chi2(features_train, labels_train == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}'".format(Article))
        print("  {}".format(', '.join(unigrams[-5:])))
        print("  {}".format(', '.join(bigrams[-2:])))
    print('\n')
    return

def create_grid_search_base_model(cv, model, param_grid, features_train, labels_train):
    # Create a base model
    f = open('data/results.txt', 'a')
    f.write('GRID SEARCH CV (folds) : ' + str(cv) + '\n')
    f.close()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=cv, verbose=1)
    # Fit the grid search to the data
    print('Fitting GridSearch to data...')
    start = time.time()
    grid_search.fit(features_train, labels_train)
    end = time.time()
    print('Duration :',round((end - start),1),'(s)')
    # Get Accuracy and best hyperparameters using Grid Search
    print('Model Mean Accuracy :',round(grid_search.best_score_*100,3),'%')
    print('Best hyperparameters :',grid_search.best_params_)
    return (grid_search)

def fit_test_model(model, features_train, labels_train, features_test, labels_test, df):
    print('\nFitting best model to training data...')
    start = time.time()
    model.fit(features_train, labels_train)
    pred = model.predict(features_test)
    # Training and Test accuracy
    results1 = 'Model : ' + str(model) + '\n'
    results2 = 'Model Training Accuracy : ' + (accuracy_score(labels_train, model.predict(features_train))*100).astype(str) + '%' + '\n'
    results3 = 'Model Test Accuracy : ' + (accuracy_score(labels_test, pred)*100).astype(str) + '%' + '\n'
    # Classification report
    results4 = 'Classification report\n' + classification_report(labels_test, pred)
    print(results1, results2, results3, results4)
    end = time.time()
    duration = str(round((end - start),1))
    print(duration, '(s)')
    results5 = 'Duration : ' + duration + '(s)' + '\n' + '*****************************************************' + '\n\n'
    # Confusion matrix
    plot_c_matrix(df, labels_test, pred)
    f = open('data/results.txt', 'a')
    f.write(results1 + results2 + results3 + results4 + results5)
    f.close()
    return (model)