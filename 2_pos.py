import nltk
import csv


def remove_adj():

        negative_preprocessed_english = []
        positive_preprocessed_english = []
        buffer = []
        
        with open('./path-to-file/2_preprocessed_negatives_removed_special.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                print('Fetching negative sentence....')
                try:
                    text = nltk.word_tokenize(row[0])
                    neg_tagged = nltk.pos_tag(text)
                    for word in neg_tagged:
                        if(word[1] =='RB' or word[1] =='RBR' or word[1] =='RBS' or word[1] =='RP' or word[1] =='DT' or word[1] =='CC' or word[1] =='EX' or word[1] =='IN' or word[1] =='MD' or word[1] =='PDT' or word[1] =='PRP' or word[1] =='PRP$' or word[1] =='WDT' or word[1] =='WP' or word[1] =='WP$' or word[1] =='WRB'):
                            pass
                        else:
                            buffer.append(word[0])
                    cleaned_string = ' '.join(buffer)
                    negative_preprocessed_english.append(cleaned_string)
                    buffer.clear()

                except:
                    pass

        with open('./path-to-file/4_preprocessed_negatives_removed_pos.csv', 'a+') as csvfile:    
            for text in negative_preprocessed_english:
                print(text)
                print('Entering negative preprocessed data!')
                csvfile.write(text + '\n')


        
        with open('./path-to-file/2_preprocessed_positives_removed_special.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    print('Fetching positive sentence....')
                    try:
                        text = nltk.word_tokenize(row[0])
                        pos_tagged = nltk.pos_tag(text)
                        for word in pos_tagged:
                            if(word[1]=='JJ' or word[1]=='JJR' or word[1] =='JJS' or word[1] =='RB' or word[1] =='RBR' or word[1] =='RBS'):
                                pass
                            else:
                                buffer.append(word[0])
                        cleaned_string = ' '.join(buffer)
                        positive_preprocessed_english.append(cleaned_string)
                        buffer.clear()

                    except:
                        pass

        with open('./path-to-file/4_preprocessed_positives_removed_pos.csv', 'a+') as csvfile:    
            for text in positive_preprocessed_english:
                print('Entering positive preprocessed data!')
                csvfile.write(text + '\n')
        


remove_adj()