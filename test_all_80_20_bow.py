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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
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

with open("./path-to-file/sample_" +sample_type+ ".csv", "r") as csvfile:
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

# Now lets do it for all the messages
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
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

for name, model in models:
    
    classifier = model
    classifier.fit(X_train, y_train)

    preds = classifier.predict(X_test)
    print(accuracy_score(y_test, preds)*100)


    #SAVE MODEL
    
    with open("./pickle_" +class_type+ "/" +sample_type+ "/pkl_"+ name +".pkl", "wb") as file:
        pickle.dump(classifier, file)
        print(name + ' saved in pickle file\n')
    