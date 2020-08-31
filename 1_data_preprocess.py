import re
import nltk
from sklearn.datasets import load_files
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv

movie_data = load_files(r"./cyberdata/")
X, y = movie_data.data, movie_data.target


documents = []
final_processed = []

preprocessed_positives_list = []
preprocessed_negatives_list = []

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):

    # remove all numeric characters (Have to be commented out in case numbers are required)
    #print('remove all numeric characters......')
    #document = re.sub(r'\s*[0-9]\s*', ' ', document)
    
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
    
    documents.append(document)

wordlist = [w for w in nltk.corpus.words.words('en')]
wordlist_final = [i.lower() for i in wordlist]

stop_words = set(stopwords.words('english'))

for sentence in documents:
    processed = []
    word_tokens = sentence.split()
    for word in word_tokens:
        #removing all stopwords 
        print('removing all stopwords....................')
        
        if(word not in stop_words):
            processed.append(word)

    processed_sentence = ' '.join(processed)
    final_processed.append(processed_sentence)

messages = zip(final_processed, y)

for (text, label) in messages:
    print('Preparing data for csv files.............')
    content = str(text)
    content = content.replace(',', '')
    content = content.replace('.', '')
    content = content.replace('\n', ' ')

    if(str(label) == '1'):
        preprocessed_positives_list.append(str(content))
    elif(str(label) == '0'):
        preprocessed_negatives_list.append(str(content))


def preprocessed_positives():
    with open('./path-to-file/2_preprocessed_positives_removed_special.csv', 'a+') as csvfile:    
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