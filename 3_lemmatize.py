import nltk
import csv
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

def create_lemmatized_sentences():
    negative_final_processed = []
    positive_final_processed = []
    
    with open('./path-to-file/2_imdb.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            print('Fetching negative sentence....')
            try:
                print('Lemmatizing........')
                document_split = nltk.word_tokenize(row[0])
                pos_tagged = nltk.pos_tag(document_split)

                document_pos = []
                for word in pos_tagged:
                    if(word[1]=='VBD' or word[1]=='VB' or word[1] =='VBG' or word[1] =='VBN' or word[1] =='VBZ'):
                        document_pos.append(stemmer.lemmatize(word[0],'v'))
                    else:
                        document_pos.append(stemmer.lemmatize(word[0]))

                processed_sentence = ' '.join(document_pos)
                negative_final_processed.append(processed_sentence)

            except:
                pass
    with open('./path-to-file/3_imdb.csv', 'a+') as csvfile:    
        for text in negative_final_processed:
            print(text)
            print('Entering negative preprocessed data!')
            csvfile.write(text + '\n')

    
    with open('./path-to-file/4_preprocessed_positives_removed_pos.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            print('Fetching positive sentence....')
            try:
                print('Lemmatizing........')
                document_split = nltk.word_tokenize(row[0])
                pos_tagged = nltk.pos_tag(document_split)

                document_pos = []
                for word in pos_tagged:
                    if(word[1]=='VBD' or word[1]=='VB' or word[1] =='VBG' or word[1] =='VBN' or word[1] =='VBZ'):
                        document_pos.append(stemmer.lemmatize(word[0],'v'))
                    else:
                        document_pos.append(stemmer.lemmatize(word[0]))

                processed_sentence = ' '.join(document_pos)
                positive_final_processed.append(processed_sentence)

            except:
                pass
    with open('./path-to-file/5_preprocessed_positives_lemmatized.csv', 'a+') as csvfile:    
        for text in positive_final_processed:
            print(text)
            print('Entering negative preprocessed data!')
            csvfile.write(text + '\n')


create_lemmatized_sentences()