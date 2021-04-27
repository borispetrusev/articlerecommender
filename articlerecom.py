from habr import Habr
from comitet import Comitet
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from englishrecommender import English_recommender
from russian_recommender import Russian_recommender


class ArticleRecommender:
    
    def __init__(self, user_name, PATH):
        self.user_name = user_name
        self.path = PATH
        
    def get_all_predictions(self):
        english_pred = English_recommender(self.user_name, self.path)
        english_pred.train_model()
        english_df = english_pred.get_daily_english_predictions()
        russian_pred = Russian_recommender(self.user_name, self.path)
        russian_pred.train_model()
        russian_df = russian_pred.get_daily_russian_predictions()
        if english_df is not None:
            english_df = english_df.sort_values(by='prediction', ascending=False)[:min(15, len(english_df))]
        else:
            english_df = ''
        if russian_df is not None:
            russian_df = russian_df.sort_values(by='prediction', ascending=False)[:min(15, len(russian_df))]
        else:
            russian_df = ''
        #english_df['lang'] = 'english'
        #russian_df['lang'] = 'russian'
        if len(english_df) > 0 and len(russian_df) > 0:
            self.combined_df = english_df.append(russian_df)
            self.combined_df = self.combined_df.reset_index(drop=True)
        elif len(english_df) > 0:
            self.combined_df = english_df
        elif len(russian_df) > 0:
            self.combined_df = russian_df
        else:
            self.combined_df = ''