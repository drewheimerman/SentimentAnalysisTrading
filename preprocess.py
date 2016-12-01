import sys
import pandas
import nltk
import re

import math
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import numpy as np
import _pickle as pc

STOCKS = ['aapl','goog','amzn','msft']
SYMBOLS = ['@','#','$','.',',',':', '…','...','(',')']
REMOVABLES = ['rt'] #'the', 'my','i','we','me','you']
STOPWORDS = set(stopwords.words('english'))
EMOTICONS = [(':)','smile'), ('(:','smile'), ('):','frown'), (':(','frown'), (':D','biggrin'), (':\'(','crying'), (':\'‑(','crying'), (')\':','crying'), (')-\':','crying'), ('D:','sadness'), (':O','surprise'), (':o','shock') ]


def preprocess(pdata):
    print('Preprocessing...')
    
    dataframe = pandas.read_pickle(pdata) #get pickled dataset from location passed in as a parameter to the function
    preprocessed_dataframe = pandas.DataFrame(columns=['date','text','length','retweets','favorites','followers'])
    
    ##ITERATE THROUGH EVERY TWEET IN THE DATAFRAME
    for it, tweet in dataframe.iterrows():
        text = tweet[0]
        retweets = tweet[1]
        favorites = tweet[2]
        followers = tweet[3]
        date = it
        
        ##if either retweets or favorites is NaN, replace NaN with 0
        if(math.isnan(retweets)):
            retweets = 0
        if(math.isnan(favorites)):
            favorites = 0
            
        #text = text+':) :( :) :(' #TEST for emoticon replacement
        #text = text+"gooooood, jeeeeezz" #TEST for repeated letter reduction
        
        text = text.replace('#','') #remove hashes
        
        ##Iterate though listed emoticons and their corrisponding emotions, replace symbol with emotion word
        for symbol, emotion  in EMOTICONS:
            text = text.replace(symbol, emotion)
            
        text =  re.sub(r"http\S+", "", text) #remove URLs from Tweet text
        text = re.sub(r'([a-zA-Z])\1{3,}', r'\1\1\1', text) #reduce excessively long repeated letters
        tk = TweetTokenizer(preserve_case=False, strip_handles=True) #create new TweetTokenizer, take all text to lowercase and remove users handles from Tweet text
        
        tokenizedtext = tk.tokenize(text) #tokenize the Tweet text using TweetTokenizer
        tokenizedtext = [word for word in tokenizedtext if (word not in STOPWORDS and word not in SYMBOLS and word not in STOCKS and word not in REMOVABLES)] #remove stopwords, extra symbols, target stocks, and other removable phrases
        
        reducedtext = ' '.join(str(word) for word in tokenizedtext) #create string from tokenized strings
        print(str(reducedtext))
        #preprocessed_dataframe.append([date,text,length, ])
        break;
    print('Done')
    
preprocess(str(sys.argv[1]))