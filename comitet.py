from nltk.stem import PorterStemmer
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import requests
import re
from pymystem3 import Mystem

class Comitet:
    
    def __init__(self, user_name, PATH):
        self.urls = ['https://api.tjournal.ru/v1.8/timeline/mainpage/recent',
       'https://api.tjournal.ru/v1.8/timeline/mainpage/popular',
       'https://api.tjournal.ru/v1.8/timeline/mainpage/week',
       "https://api.tjournal.ru/v1.8/timeline/mainpage/month",
       "https://api.dtf.ru/v1.8/timeline/mainpage/recent",
        "https://api.dtf.ru/v1.8/timeline/mainpage/popular",
        "https://api.dtf.ru/v1.8/timeline/mainpage/week",
        "https://api.dtf.ru/v1.8/timeline/mainpage/month",
       "https://api.vc.ru/v1.8/timeline/mainpage/recent",
       "https://api.vc.ru/v1.8/timeline/mainpage/popular",
       "https://api.dtf.ru/v1.8/timeline/mainpage/week",
       "https://api.dtf.ru/v1.8/timeline/mainpage/month"]
        self.user_name = user_name
        self.m = Mystem()
        try:
            self.article_history = pd.read_csv(PATH+'russian_article_history_'+self.user_name+'.csv', index_col=None)
        except:
            self.article_history = pd.DataFrame(columns=['url', 'title','prediction'])
            self.article_history.to_csv(PATH+'russian_article_history_'+self.user_name+'.csv', index=False)    
        
    def form_dataset(self, result):
        dataset = pd.DataFrame(columns = ['url','title'])
        for i in result.json()['result']:
            dataset = dataset.append({'url':i['url'], 'title':i['title']},ignore_index=True)
        return dataset
    
    def lemmatize(self, text):
        lemm_list = self.m.lemmatize(text)
        lemm_text = "".join(lemm_list)
        return lemm_text

    def clear_text(self, text):
        clear_text = re.sub(r'[^а-яА-ЯёЁ ]', ' ', text)
        clear_text = clear_text.split()
        clear_text = " ".join(clear_text)
        return clear_text
    
    def retrieve_data(self):
        self.comitet_df = pd.DataFrame(columns = ['url','title'])
        for url in self.urls:
            result = requests.get(url)
            self.comitet_df = self.comitet_df.append(self.form_dataset(result))
        self.comitet_df = self.comitet_df[self.comitet_df['title']!=''].reset_index(drop=True)
        self.comitet_df = self.comitet_df.drop_duplicates()
        self.comitet_df = self.comitet_df[~self.comitet_df.url.isin(self.article_history.url)].reset_index(
            drop=True)
        self.comitet_df['type'] = 'comitet'
            
    def lemmatize_data(self):
        if len(self.comitet_df) > 0:
            tqdm.pandas()
            self.comitet_df['lemmatize_title'] = self.comitet_df['title'].progress_apply(
                lambda x: self.lemmatize(self.clear_text(x)))
            self.comitet_df = self.comitet_df[~self.comitet_df['lemmatize_title'].isna()]
            self.comitet_df = self.comitet_df[self.comitet_df['lemmatize_title']!='']
            self.comitet_df = self.comitet_df.reset_index(drop=True)