import numpy as np
import re
import nltk
from sklearn.datasets import load_files
import csv
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,precision_score,recall_score,f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import csv
import pickle
from sklearn import model_selection


final_processed = []
y = []


# "80-20" or 10_fold
class_type = "80-20"
sample_type = "100k"

with open("./test_sheets/sample_" +sample_type+ ".csv", "r") as csvfile:
    reader = csv.reader(csvfile, skipinitialspace=True)
    for row in reader:
        #print(row)
        final_processed.append(row[0])
        y.append(row[1])



# create bag-of-words
all_words = []

for message in final_processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)


# print the total number of words and the 15 most common words
print('Number of words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words.most_common(15)))

# use the 1500 most common words as features
word_features = list(all_words.keys())[:1500]
#print(word_features)


def find_features(message):
    words = word_tokenize(message)
    features = []
    for word in word_features:
        if(word in words):
            features.append(1)
        else:
            features.append(0)

    return features

#for all the messages
messages = zip(final_processed, y)

#Shuffle the zip of messages ('text', 'label')
temp = list(messages)
random.shuffle(temp)

res1, res2 = zip(*temp)
shuffle_final_processed = list(res1)
shuffled_y = list(res2)

shuffled_messages = zip(shuffle_final_processed, shuffled_y)


# define a seed for reproducibility
seed = 1
np.random.seed = seed

# call find_features function for each text
X_new = []
y_new = []

for text,label in shuffled_messages:
    X_new.append(find_features(text))
    y_new.append(label)


# split the data into training and testing datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new, y_new, test_size = 0.20, random_state=seed)


# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
        "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(n_jobs=-1),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_jobs=-1),
    LogisticRegression(n_jobs=-1),
    SGDClassifier(max_iter = 100, n_jobs=-1),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

print("Starting classification")


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



for name, model in models:
    
    classifier = model
    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)


    print(name)
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

    #SAVE MODEL
    with open("./pickle_" +class_type+ "/" +sample_type+  "/" + str(i_num) + "/pkl_"+ name +".pkl", "wb") as file:
        pickle.dump(classifier, file)
        print(name + ' saved in pickle file\n')