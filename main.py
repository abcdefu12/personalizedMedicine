# PERSONALIZED MEDICINE
# 1. IMPORT MODULES
# 2. DATA FILES
# 3. PRE-PROCESSING OF TEXT
####################


# 1. IMPORT MODULES
import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
# from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

from mlxtend.classifier import StackingClassifier

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

# 2. DATA FILES
# IF DIDNOT DOWNLOAD DATA (using Github kaggle data can be downloaded) #
# CHECKING DIRECTORY
print('List of files in input folder:', os.listdir('./input'))
# print(os.listdir('./input/msk-redefining-cancer-treatment'))

# READING GENE & VARIANCE DATA
data = pd.read_csv('./input/training_variants')
print('< training_variants > -> pic1.data')
print('Number of data points : ', data.shape[0])
print('Number of features : ', data.shape[1])
print('Features : ', data.columns.values)
data.head()

# ABOUT FEATURES
# ID : the id of the row used to link the mutation to the clinical evidence
# Gene : the gene where this genetic mutation is located
# Variation : the amino-acid change for this mutations
# Class : 1-9 the class this genetic mutation has been classified on

# READING TEXT DATA
data_text = pd.read_csv("./input/training_text", sep="\|\|", engine="python", names=["ID", "TEXT"],
                        skiprows=1)  # skiprows=1 used to skip heading
print('< training_text > -> pic2.data_text')
print('Number of data points : ', data_text.shape[0])
print('Number of features : ', data_text.shape[1])
print('Features : ', data_text.columns.values)
data_text.head()

# 3. PRE-PROCESSING OF TEXT
# Use nltk library to load stopwords
stop_words = set(stopwords.words('english'))


def nlp_preprocessing(total_text, index, column):
    if type(total_text) is not stop_words:
        string = ""
        # replace every special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)
        # replace multiple spaces with single space
        total_text = re.sub('\s+', ' ', total_text)
        # converting all the chars into lower-case.
        total_text = total_text.lower()

        for word in total_text.split():
            # if the word is a not a stop word then retain that word from the data
            if word not in stop_words:
                string += word + " "
        data_text[column][index] = string


# text processing stage
print('< preprocessing text data > -> pic3.total_text')
# use time.perf_counter instead of time.clock() due to python version changed.
start_time = time.perf_counter()
for index, row in data_text.iterrows():
    if type(row['TEXT']) is str:
        nlp_preprocessing(row['TEXT'], index, 'TEXT')
    else:
        print("there is no text description for id:", index)
print('Time took for preprocessing the text :', time.perf_counter() - start_time, "seconds")

# Merge both gene_variations and text data based on ID
result = pd.merge(data, data_text, on='ID', how='left')
result.head()
# pic4.Merging two datasets together based on ID

# Handle Missing data
# pic5. result.info
print('< pic5 >')
result.info()
# pic6.
print('< pic6 >')
result.isnull().any(axis=0)
# pic7.
print('< pic7 >')
result.isnull().any(axis=1)
# pic8.
print('< pic8 >')
result[result.isnull().any(axis=1)]

result.loc[result['TEXT'].isnull(), 'TEXT'] = result['Gene'] + ' ' + result['Variation']
# pic9.
print('< pic9 >')
result[result['ID'] == 1109]


