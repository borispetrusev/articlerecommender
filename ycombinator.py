from nltk.stem import PorterStemmer
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import requests
import re

class Ycombinator:
    
    def __init__(self, user_name, PATH):
        self.url = 'https://hacker-news.firebaseio.com/v0/newstories.json'   
        self.news_url = 'https://hacker-news.firebaseio.com/v0/item/'
        self.porter = PorterStemmer()
        self.user_name = user_name
        try:
            self.article_history = pd.read_csv(PATH+'english_article_history_'+self.user_name+'.csv', index_col=None)
        except:
            self.article_history = pd.DataFrame(columns=['url', 'title','prediction'])
            self.article_history.to_csv(PATH+'english_article_history_'+self.user_name+'.csv', index=False)
     
    def retrieve_data(self):
        ids = requests.get(self.url)
        self.ycombinator_df = pd.DataFrame(columns=['url', 'title'])
        for id_ in ids.json():
            url = self.news_url + str(id_) +'.json'
            article = requests.get(url)
            try:
                self.ycombinator_df = self.ycombinator_df.append({'url':article.json()['url'], 
                                                                  'title':article.json()['title']}, 
                                                                 ignore_index=True)
            except:
                continue
        self.ycombinator_df = self.ycombinator_df.drop_duplicates(subset='url').reset_index(drop=True)
        self.ycombinator_df = self.ycombinator_df[~self.ycombinator_df.url.isin(
            self.article_history.url)].reset_index(drop=True)
        self.ycombinator_df['type'] = 'ycombinator'

    def stemSentence(self, sentence):
        token_words=word_tokenize(sentence)
        token_words
        stem_sentence=[]
        for word in token_words:
            stem_sentence.append(self.porter.stem(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)

    def clean_text(self, sentence):
        sentence = re.sub(r'[^a-zA-Z ]', ' ', sentence)
        sentence = " ".join(sentence.split())
        return sentence
    
    def lemmatize_data(self):
        if len(self.ycombinator_df) > 0:
            tqdm.pandas()
            self.ycombinator_df['lemmatize_title'] = self.ycombinator_df['title'].progress_apply(
                lambda x: self.clean_text(self.stemSentence(x)))
            self.ycombinator_df = self.ycombinator_df[~self.ycombinator_df['lemmatize_title'].isna()]