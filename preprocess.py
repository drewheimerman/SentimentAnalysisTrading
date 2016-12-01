import sys,pandas,nltk,re,math,time, tqdm
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import MWETokenizer
from nltk.corpus import stopwords
import numpy as np
import _pickle as pc

STOCKS = ['aapl','goog','amzn','msft']
SYMBOLS = ['@','#','$','.',',',':', '…','...','(',')','"','[',']']
REMOVABLES = ['rt'] #'the', 'my','i','we','me','you']
STOPWORDS = set(stopwords.words('english'))
EMOTICONS = [(':)','smile'), ('(:','smile'), ('):','frown'), (':(','frown'), (':D','biggrin'), (':\'(','crying'), (':\'‑(','crying'), (')\':','crying'), (')-\':','crying'), ('D:','sadness'), (':O','surprise'), (':o','shock') ]

def preprocess(pdata):
    print('Preprocessing...')  
    dataframe = pandas.read_pickle(pdata) #get pickled dataset from location passed in as a parameter to the function
    dataframe['date'] = dataframe.index
    dataframe.index = range(0, len(dataframe))
    ##ITERATE THROUGH EVERY TWEET IN THE DATAFRAME
    for it, tweet in tqdm.tqdm(dataframe.iterrows()):
        
        text = tweet[0]
        retweets = tweet[1]
        favorites = tweet[2]
        followers = tweet[3]
        date = it
        
        ##if either retweets, favorites, or followers is NaN, replace NaN with 0
        if(math.isnan(retweets)):
            retweets = 0
        if(math.isnan(favorites)):
            favorites = 0
        if(math.isnan(followers)):
            followers = 0
        text = text.replace('#','') #remove hashes
        text = text.replace('%', 'percent') #replace % symbol with 'percent'
        ##Iterate though listed emoticons and their corrisponding emotions, replace symbol with emotion word
        for symbol, emotion  in EMOTICONS:
            text = text.replace(symbol, emotion)   
        text =  re.sub(r"http\S+", "", text) #remove URLs from Tweet text
        tk = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True) #create new TweetTokenizer, take all text to lowercase and remove users handles from Tweet text
        tokenizedtext = tk.tokenize(text) #tokenize the Tweet text using TweetTokenizer
        tokenizedtext = [word for word in tokenizedtext if (word not in STOPWORDS and word not in SYMBOLS and word not in STOCKS and word not in REMOVABLES)] #remove stopwords, extra symbols, target stocks, and other removable phrases
        
        pred_text = ' '.join(word for word in tokenizedtext) #join the tokens and space separate them
        ##Modify the contents of the dataframe at the current interated row
        dataframe.set_value(it,'text', pred_text)
        dataframe.set_value(it,'retweets',retweets)
        dataframe.set_value(it,'favorites',favorites)
        dataframe.set_value(it,'followers',followers)
    dataframe.to_pickle('preprocessed_tweets_s'+str(int(len(dataframe)/100))+'.p') #create a pickled dataframe with a semi-unique identifier (based on the number of rows in the dataframe)
    return dataframe #return the dataframe
    
preprocess(str(sys.argv[1]))