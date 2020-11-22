import re
import nltk
from sklearn.datasets import load_files
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


from rouge_score import rouge_scorer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords 


from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen


def preprocess_list(summarized_sentences):

    final_processed = []
    documents = []

    for doc in summarized_sentences:

        # Removing prefixed 'b'
        print('Removing prefixed b......')
        document = re.sub(r'^b\s+', '', doc)

        # Converting to Lowercase
        print('Converting to Lowercase......')
        document = document.lower()

        # Removing 'sql injection'
        print('Removing sql injection......')
        document = re.sub(r'sql injection\s*', '', document)

        # Remove single characters
        print('Removing single characters............')
        document = re.sub(r'\b\w\b', ' ', document)

        removelist = ""

        # Remove all the special characters
        print('Removing special characters except fullstop....')
        document = re.sub(r'[^\w'+removelist+']', ' ', document)

        # remove all underscore characters
        print('Removing all underscore characters....')
        document = re.sub(r'_', ' ', document)

        # Substituting multiple spaces with single space
        print('Substituting multiple spaces with single space............')
        document = re.sub(r'\s+', ' ', document)

        documents.append(document)

    #wordlist = [w for w in nltk.corpus.words.words('en')]
    #wordlist_final = [i.lower() for i in wordlist]

    additional_filters = ['is','there','via','earlier','after','prior','later','also','allows','an','the','or','was','were','where','here','that','it','are','obtained','these','those','become','may','might','should','would','will','shall','can','could','has','had','have','crafted','by','some','their','with','without','cannot','not','many','few','small','through','related','relate','discover','discovered','aka','other','same','hence','therefore','one','more','cause','sustain','adjacent','occur','as','sql injection','sql queries','similar','thus','thereby','although','trusted','while leading','apparantly','filtered','forwarded','vulnerability','in','to','when','before','this','for','from','a','however','directly','around','already','extract','multiple','certain','send','projects','action','get','possibly','offer','made','achieve','containing','involving','within']

    stop_words = set(stopwords.words('english'))

    for sentence in documents:
        processed = []
        word_tokens = sentence.split()
        for word in word_tokens:
            #removing all stopwords and selecting only english words
            print('removing all stopwords....................')
            #if((word not in stop_words) and (word in wordlist_final)):
            if((word not in stop_words) and (word not in additional_filters) ):
                processed.append(word)

        processed_sentence = ' '.join(processed)
        final_processed.append(processed_sentence)

    #print(final_processed)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    buffer = []
    final_processed1 = []

    for row in final_processed:
        print('Fetching sentence....')
        try:
            text = nltk.word_tokenize(row)
            tagged = nltk.pos_tag(text)

            for word in tagged:
                if(word[1] =='RB' or word[1] =='RBR' or word[1] =='RBS' or word[1] =='RP' or word[1] =='DT' or word[1] =='CC' or word[1] =='EX' or word[1] =='IN' or word[1] =='MD' or word[1] =='PDT' or word[1] =='PRP' or word[1] =='PRP$' or word[1] =='WDT' or word[1] =='WP' or word[1] =='WP$' or word[1] =='WRB'):
                    pass
                else:
                    buffer.append(word[0])
            cleaned_string = ' '.join(buffer)
            final_processed1.append(cleaned_string)
            buffer.clear()
        
        except:
            pass

    #print(final_processed1)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    final_processed2 = []

    try:
        for row in final_processed1:
            stemmer = WordNetLemmatizer()
            print('Lemmatizing........')
            document_split = nltk.word_tokenize(row)
            pos_tagged = nltk.pos_tag(document_split)

            document_pos = []
            for word in pos_tagged:
                if(word[1]=='VBD' or word[1]=='VB' or word[1] =='VBG' or word[1] =='VBN' or word[1] =='VBZ'):
                    document_pos.append(stemmer.lemmatize(word[0],'v'))
                else:
                    document_pos.append(stemmer.lemmatize(word[0]))

            processed_sentence = ' '.join(document_pos)
            final_processed2.append(processed_sentence)

    except:
        pass

    return(final_processed2)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#Contains all the url links to be preprocessed and summarized.
urls = []
num_words_list = [50]

for num_words in num_words_list:
    preprocessed_summaries = []
    all_text = []

    for url in urls:

        try:
            page = urlopen(url)
            soup = BeautifulSoup(page,"html.parser")
            document = ' '.join(map(lambda p: p.text, soup.find_all('p')))
            #print(document)

            print(url)
            print(urls.index(url))

            #Using Textrank
            textrank_summary = summarize(document, word_count = num_words)
            all_text.append(textrank_summary)
            print('Added summary!')

        except:
            continue


    preprocessed_summaries = preprocess_list(all_text)
    #print(preprocessed_summaries)

    with open('./'+str(num_words)+'_sqli.csv', 'a+') as csvfile:
        for text in preprocessed_summaries:
            #print(text)
            print('Saving Summaries!')
            text = text.replace('.','')
            text = text.replace(',','')
            csvfile.write(text + '\n')
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')