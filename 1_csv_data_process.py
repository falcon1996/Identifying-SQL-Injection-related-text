import re
import nltk
from sklearn.datasets import load_files
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv

documents = []
final_processed = []

stemmer = WordNetLemmatizer()

X = []
with open('./statements_after_preprocess/IMDb movies.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        print('Fetching imdb sentence....')
        description = row[0]
        X.append(description)


for sen in range(0, len(X)):
    
    # Removing prefixed 'b'
    print('Removing prefixed b......')
    document = re.sub(r'^b\s+', '', str(X[sen]))
    
    # Converting to Lowercase
    print('Converting to Lowercase......')
    document = document.lower()

    # Removing 'injection'
    print('Removing sql injection......')
    document = re.sub(r'sql injection\s*', '', document)
    
    # Remove single characters
    print('Removing single characters............')
    document = re.sub(r'\b\w\b', ' ', document)

    # Remove all the special characters
    print('Removing special characters....')
    document = re.sub(r'\W', ' ', document)

    # remove all underscore characters
    print('Removing all underscore characters....')
    document = re.sub(r'_', ' ', document)

    # Substituting multiple spaces with single space
    print('Substituting multiple spaces with single space............')
    document = re.sub(r'\s+', ' ', document)

    # remove all numeric characters
    print('remove all numeric characters......')
    document = re.sub(r'[0-9]', ' ', document)
    
    documents.append(document)

wordlist = [w for w in nltk.corpus.words.words('en')]
wordlist_final = [i.lower() for i in wordlist]

stop_words = set(stopwords.words('english'))

for sentence in documents:
    processed = []
    word_tokens = sentence.split()
    for word in word_tokens:
        #removing all stopwords and selecting only english words
        print('removing all stopwords and selecting only english words....................')
        if((word not in stop_words) and (word in wordlist_final)):
        #if(word not in stop_words):
            processed.append(word)

    processed_sentence = ' '.join(processed)
    final_processed.append(processed_sentence)


def preprocessed_imdb():
    with open('./statements_after_preprocess/1_imdb.csv', 'a+') as csvfile:    
        for text in final_processed:
            print('Entering preprocessed imdb data!')
            csvfile.write(text + '\n')

def preprocessed_positives():
    with open('./path-to-file/2_preprocessed_positives_removed_non_english.csv', 'a+') as csvfile:    
        for text in preprocessed_positives_list:
            print('Entering positive preprocessed data!')
            csvfile.write(text + '\n')

def preprocessed_negatives():
    with open('./path-to-file/2_preprocessed_negatives_removed_special.csv', 'a+') as csvfile:
        for text in preprocessed_negatives_list:
            print('Entering negative preprocessed data!')
            csvfile.write(text + '\n')


preprocessed_positives()
preprocessed_negatives()

preprocessed_imdb()