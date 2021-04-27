from nltk.stem import PorterStemmer
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import requests
import re
from pymystem3 import Mystem

class Habr:
    
    def __init__(self, user_name, PATH):
        self.urls = ['https://habr.com','https://habr.com/news/']
        self.user_name = user_name
        self.m = Mystem()
        try:
            self.article_history = pd.read_csv(PATH+'russian_article_history_'+self.user_name+'.csv', index_col=None)
        except:
            self.article_history = pd.DataFrame(columns=['url', 'title','prediction'])
            self.article_history.to_csv(PATH+'russian_article_history_'+self.user_name+'.csv', index=False)
    
    
    def retrieve_data(self):
        self.habr_df = pd.DataFrame(columns=['url', 'title'])
        for url in self.urls:
            html = requests.get(url)
            soup = BeautifulSoup(html.text)
            for link in soup.find_all('a'):
                try:
                    href = link.get('href')
                    if url == 'https://habr.com':
                        if ('/post/' in href) and ('#' not in href):
                            self.habr_df = self.habr_df.append({'url':href, 'title':link.contents[0].strip()}, 
                                                   ignore_index=True)
                    else:
                        if ('/news/' in href) and ('#' not in href):
                            self.habr_df = self.habr_df.append({'url':href, 'title':link.contents[0].strip()}, 
                                                   ignore_index=True)
                except:
                    continue
        self.habr_df = self.habr_df[self.habr_df['title']!='']
        self.habr_df = self.habr_df.drop_duplicates(subset='url').reset_index(drop=True)
        self.habr_df = self.habr_df[self.habr_df['title'].str.split().str.len() > 3].reset_index(drop=True)
        self.habr_df = self.habr_df[~self.habr_df.url.isin(
            self.article_history.url)].reset_index(drop=True)
        self.habr_df['type'] = 'habr'
        
    def lemmatize(self, text):
        lemm_list = self.m.lemmatize(text)
        lemm_text = "".join(lemm_list)
        return lemm_text

    def clear_text(self, text):
        clear_text = re.sub(r'[^а-яА-ЯёЁ ]', ' ', text)
        clear_text = clear_text.split()
        clear_text = " ".join(clear_text)
        return clear_text
    
    def lemmatize_data(self):
        if len(self.habr_df) > 0:
            tqdm.pandas()
            self.habr_df['lemmatize_title'] = self.habr_df['title'].progress_apply(
                lambda x: self.lemmatize(self.clear_text(x)))
            self.habr_df = self.habr_df[~self.habr_df['lemmatize_title'].isna()]