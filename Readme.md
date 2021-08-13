## Data Science Python Projects
![MTU Logo](/1/data/MTU_Logo.jpg)

### ETL Project 1
#### ETL and analysis using Pandas and text-based menu for user interactions on Auto Dataset.

<li><a href="https://github.com/bjmcnamee/ETL_and_ML_Python_Assignments/blob/main/1/COMP8060_ProjectSpecification-1.pdf">Project Specification</a></li>
<li><a href="https://github.com/bjmcnamee/ETL_and_ML_Python_Assignments/blob/main/1/data/importsAuto.csv">Dataset</a></li>
<li><a href="https://github.com/bjmcnamee/ETL_and_ML_Python_Assignments/blob/main/1/McNamee_R00207204_Lab8060.py">Project Code</a></li>

### ETL Project 2
#### ETL and analysis using Pandas and text-based menu for user interactions on Census and Nutrition Datasets.

<li><a href="https://github.com/bjmcnamee/ETL_and_ML_Python_Assignments/blob/main/2/COMP8060_ProjectSpecification-1.pdf">Project Specifications</a></li>
<li><a href="https://github.com/bjmcnamee/ETL_and_ML_Python_Assignments/blob/main/2/data">Datasets (input and output)</a></li>
<li><a href="https://github.com/bjmcnamee/ETL_and_ML_Python_Assignments/blob/main/2/McNamee_R00207204_Project.py">Project Code</a></li>
<li><a href="https://github.com/bjmcnamee/ETL_and_ML_Python_Assignments/blob/main/2/McNamee_R00207204_Report.pdf">Report & Conclusions</a></li>

### Machine Learning Project 3
#### Linear Regression (LR)
<li><a href="https://github.com/bjmcnamee/ETL_and_ML_Python_Assignments/blob/main/3/DATA8001%20Assignment%201%20Instructions.pdf">Project Specifications</a></li>
<li><a href="https://github.com/bjmcnamee/ETL_and_ML_Python_Assignments/blob/main/3/data">Datasets (input and output)</a></li>
<li><a href="https://github.com/bjmcnamee/ETL_and_ML_Python_Assignments/blob/main/3/R00207204_A1_Notebook.ipynb">Project Code</a></li>
<li><a href="https://github.com/bjmcnamee/ETL_and_ML_Python_Assignments/blob/main/3/R00207204_A1_Report.pdf">Report & Conclusions</a></li>

### Machine Learning Project 4
#### Natural language Processing (NLP)
<li><a href="https://github.com/bjmcnamee/ETL_and_ML_Python_Assignments/blob/main/4/DATA8001%20Assignment%202%20Instructions.pdf">Project Specifications</a></li>
<li><a href="https://github.com/bjmcnamee/ETL_and_ML_Python_Assignments/blob/main/4/data">Datasets (input and output)</a></li>
<li><a href="https://github.com/bjmcnamee/ETL_and_ML_Python_Assignments/blob/main/4/R00207204_A2_Notebook.ipynb">Project Code</a></li>
<li><a href="https://github.com/bjmcnamee/ETL_and_ML_Python_Assignments/blob/main/4/R00207204_A2_Report.pdf">Report & Conclusions</a></li>

#### This ML Multiclass project using python and sklearn modules trains and tests three classification models that predict the category of a given news article.
#### Process
  1. ETL – 2000 text file documents were imported, filtered by html tag, and content features (headline, article and category) saved to related columns in a Pandas dataframe.
  2. Explore – a statistical summary and plots shows the data is quite uniform (word count and document category count) with no missing or incomplete features.
  3. Pre-process – in preparation for running ML models, the text was ‘cleaned’ again removing unwanted punctuation, digits, uppercase and stopwords
  4. Tokenise – inflected words were reduced to smaller units called token by one of two functions, Stem and Lemmatise, eg caring → car or caring → care
  5. Vectorise – the remaining words were converted to float numbers using a frequency–inverse document frequency TF-IDF method. 
  6. Model – 3 multiclass classifcation models were selected
      1. K Nearest Neighbour (KNN)
      2. Support Vector Machine (SVM)
      3. Multinomial Naive Bayes (MNB).
  7. Tune Model - Grid Search was applied to each model to discover the best hyperparameters and values.
  8. Model – the data was split into training and test, and each of three models, trained and tested. Model accuracy was calculated including classification reports and confusion matrices.
  9. Model selection – each model was run with various parameters fixed one at a time and the results written to file for compilation in order find the best model.

