import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import model_selection

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import mean
from numpy import std
import nltk
from nltk.tokenize import word_tokenize
import csv
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,precision_score,recall_score,f1_score

dataset = []
sentences = []
y = []

# "80-20" or 10_fold
class_type = "80-20"
sample_type = "100k"

seed = 1
np.random.seed = seed


with open("./test_sheets/sample_" +sample_type+ ".csv", "r") as csvfile:
    reader = csv.reader(csvfile, skipinitialspace=True)
    for row in reader:
        dataset.append(row)
        sentences.append(row[0])
        y.append(row[1])


# create bag-of-words
all_words = []

for message in sentences:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)

# Extracting top 1500 words as features
word_features = list(all_words.keys())[:1500]

def find_features(message):
    words = word_tokenize(message)
    features = []
    for word in word_features:
        if(word in words):
            features.append(1)
        else:
            features.append(0)

    return features

test_sentences = []
y_test = []

#Path of the file generated after pre-processing urls
with open("./secondary_test/test.csv", "r") as testfile:
    reader = csv.reader(testfile, skipinitialspace=True)
    for row in reader:
        test_sentences.append(row[0])
        y_test.append(row[1])

test_featuresets = [find_features(text) for text in test_sentences]

model_locations = [
    "./pickle_10fold/100k/pkl_Random Forest.pkl",
]

def confusion_metrics (conf_matrix):
    # save confusion matrix and slice into four pieces    
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]    
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)
    
    # calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
    
    # calculate mis-classification
    conf_misclassification = 1- conf_accuracy
    
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))   

    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    
    # calculate precision
    conf_precision = (TP / float(TP + FP))    
    # calculate f1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))    

    print('-'*50)
    print('Accuracy: ',round(conf_accuracy,2)) 
    print('Mis-Classification: ',round(conf_misclassification,2)) 
    print('Recall / Sensitivity: ',round(conf_sensitivity,2)) 
    print('Specificity: ',round(conf_specificity,2)) 
    print('Precision: ',round(conf_precision,2))
    print('f_1 Score: ',round(conf_f1,2))

for model_location in model_locations:
    with open(model_location, 'rb') as model_file:
        model = pickle.load(model_file)
    preds = model.predict(test_featuresets)

    print(model_location)
    print('Accuracy:')
    print(accuracy_score(y_test, preds)*100)


    precision_average = precision_score(y_test, preds, average="binary", pos_label='1')
    recall_average = recall_score(y_test, preds, average="binary", pos_label='1')
    f1_average = f1_score(y_test, preds, average="binary", pos_label='1')

    print('Precision: ',precision_average)
    print('Recall: ',recall_average)
    print('F1: ',f1_average)

    confusion_matrix_table = confusion_matrix(y_test, preds)
    confusion_metrics(confusion_matrix_table)
    
    print('\n')