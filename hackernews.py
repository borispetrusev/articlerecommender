from nltk.stem import PorterStemmer
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import requests
import re

class Thehackernews:
    
    def __init__(self, user_name, PATH):
        self.url = 'https://thehackernews.com'    
        self.porter = PorterStemmer()
        self.user_name = user_name
        try:
            self.article_history = pd.read_csv(PATH+'english_article_history_'+self.user_name+'.csv', index_col=None)
        except:
            self.article_history = pd.DataFrame(columns=['url', 'title','prediction'])
            self.article_history.to_csv(PATH+'english_article_history_'+self.user_name+'.csv', index=False)
            
    def combine_titles(self, row):
        if 'href' in row['title_1']:
            return row['title_2']
        return row['title_1']
    
    
    def retrieve_data(self):
        html = requests.get(self.url)
        soup = BeautifulSoup(html.text)
        self.thehackernews_df = pd.DataFrame(columns=['url', 'title_1', 'title_2'])
        for link in soup.find_all('a'):
            href = link.get('href')
            if 'https://thehackernews.com' in href and 'search?' not in href:
                conv_link = str(link)
                location_start = conv_link.find('<div class="pop-title">')+23
                location_end = conv_link.find('</div>', location_start)
                title_1 = conv_link[location_start:location_end]
                location_start = conv_link.find('<h2 class="home-title">')+23
                location_end = conv_link.find('</h2>', location_start)
                title_2 = conv_link[location_start:location_end]
                self.thehackernews_df = self.thehackernews_df.append({'url':href, 'title_1':title_1, 
                                                                      'title_2':title_2}, ignore_index=True)
        self.thehackernews_df['title'] = self.thehackernews_df.apply(lambda x: self.combine_titles(x), axis=1)
        del self.thehackernews_df['title_1']
        del self.thehackernews_df['title_2']
        self.thehackernews_df = self.thehackernews_df.drop_duplicates(subset='url').reset_index(drop=True)  
        self.thehackernews_df = self.thehackernews_df[~self.thehackernews_df.url.isin(
            self.article_history.url)].reset_index(drop=True)
        self.thehackernews_df['type'] = 'hackernews'

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
        if len(self.thehackernews_df) > 0:
            tqdm.pandas()
            self.thehackernews_df['lemmatize_title'] = self.thehackernews_df['title'].progress_apply(
                lambda x: self.clean_text(self.stemSentence(x)))
            self.thehackernews_df = self.thehackernews_df[~self.thehackernews_df['lemmatize_title'].isna()]