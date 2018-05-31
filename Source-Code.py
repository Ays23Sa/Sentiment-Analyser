import numpy as np
import seaborn as sns
import matplotlib as plt
import pandas as pd
import re
import string	
from nltk.corpus import stopwords
%matplotlib inline

# Reducing the number of emotion classes
df = pd.read_csv('text_emotion.csv',index_col=0)
df.loc[(df['sentiment']=='enthusiasm') | (df['sentiment']=='fun') |  (df['sentiment']=='love') | (df['sentiment']=='happiness') | (df['sentiment']=='relief'),'sentiment'] = 'Happy'
df.loc[(df['sentiment']=='sadness') | (df['sentiment']=='empty') | (df['sentiment']=='hate') | (df['sentiment']=='anger')|(df['sentiment']=='worry') | (df['sentiment']=='boredom'),'sentiment'] ='Sad'
df.drop(df[df.sentiment=='surprise'].index,inplace = True)
sns.countplot(x='sentiment',data=df)
df['length'] = df['content'].str.len()
sns.barplot(x='sentiment',y='length',data=df)
sns.stripplot(x='sentiment',y='length',data=df,jitter=True)
# stop = stopwords.words('english')
# replaced_words=['AT_USER','URL']
# stop.extend(replaced_words)
stop=['AT_USER','URL']
def text_process(tweet):
    # process the tweets
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return [word for word in tweet.split() if word not in stop] 
df['content'].head(10).apply(text_process)
from sklearn.feature_extraction.text import CountVectorizer #converts bag of words into vector
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
tweet_train,tweet_test,label_train,label_test=train_test_split(df['content'],df['sentiment'],test_size=0.3)
from sklearn.pipeline import Pipeline
pipeline=Pipeline([('bow',CountVectorizer(analyzer=text_process)),('tfidf',TfidfTransformer()),('classifier',MultinomialNB()),])
pipeline.fit(tweet_train,label_train)
predictions = pipeline.predict(tweet_test)
from sklearn.metrics import classification_report
print(classification_report(label_test,predictions))
