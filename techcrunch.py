from nltk.stem import PorterStemmer
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import requests
import re

class Techcrunch:
    
    def __init__(self, user_name, PATH):
        self.url = 'https://techcrunch.com'    
        self.porter = PorterStemmer()
        self.user_name = user_name
        try:
            self.article_history = pd.read_csv(PATH+'english_article_history_'+self.user_name+'.csv', index_col=None)
        except:
            self.article_history = pd.DataFrame(columns=['url', 'title','prediction'])
            self.article_history.to_csv(PATH+'english_article_history_'+self.user_name+'.csv', index=False)
    
    
    def retrieve_data(self):
        html = requests.get(self.url)
        soup = BeautifulSoup(html.text)
        self.techcrunch_df = pd.DataFrame(columns=['url', 'title'])
        for link in soup.find_all('a'):
            href = link.get('href')
            if 'author' not in href and 'page' not in href:
                self.techcrunch_df = self.techcrunch_df.append({'url':href, 'title':link.contents[0].strip()}, 
                                                     ignore_index=True)
        self.techcrunch_df = self.techcrunch_df.drop_duplicates(subset='url').reset_index(drop=True)
        self.techcrunch_df = self.techcrunch_df[~self.techcrunch_df.url.isin(
            self.article_history.url)].reset_index(drop=True)
        self.techcrunch_df['type'] = 'techcrunch'
    
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
        if len(self.techcrunch_df) > 0:
            tqdm.pandas()
            self.techcrunch_df['lemmatize_title'] = self.techcrunch_df['title'].progress_apply(
                lambda x: self.clean_text(self.stemSentence(x)))
            self.techcrunch_df = self.techcrunch_df[~self.techcrunch_df['lemmatize_title'].isna()]