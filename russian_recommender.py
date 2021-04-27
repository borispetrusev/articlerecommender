from habr import Habr
from comitet import Comitet
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score


class Russian_recommender:
    
    def __init__(self, user_name, PATH):
        self.path = PATH
        self.user_name = user_name
        self.training_dataset = pd.read_csv(PATH+'russian_training_'+self.user_name+'.csv')
        self.training_dataset = self.training_dataset[~self.training_dataset['lemmatize_title'].isna()]
        try:
            self.article_history = pd.read_csv(PATH+'russian_article_history_'+self.user_name+'.csv', index_col=None)
        except:
            self.article_history = pd.DataFrame(columns=['url', 'title','prediction'])
            self.article_history.to_csv(PATH+'russian_article_history_'+self.user_name+'.csv', index=False)
        
    def get_all_data(self):
        habr = Habr(self.user_name, self.path)
        habr.retrieve_data()
        habr.lemmatize_data()
        comitet = Comitet(self.user_name, self.path)
        comitet.retrieve_data()
        comitet.lemmatize_data()
        prediction_dataset = habr.habr_df.append(comitet.comitet_df)
        prediction_dataset = prediction_dataset.reset_index(drop=True)
        return prediction_dataset
        
    def train_model(self):
        X = self.training_dataset['lemmatize_title']
        y = self.training_dataset['is_interesting']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42, 
                                                            shuffle=True)
        self.model = CatBoostClassifier(verbose=1000, iterations=2000, eval_metric='Precision', depth=8)
        self.transformer = TfidfVectorizer()
        self.transformer.fit(X_train)
        X_train_tr = self.transformer.transform(X_train)
        self.model.fit(X_train_tr.toarray(), y_train)
        y_pred = self.model.predict(self.transformer.transform(X_test).toarray())
        print('Recall score', recall_score(y_test, y_pred))
        print('Precision score', precision_score(y_test, y_pred))
        print('F1 score', f1_score(y_test, y_pred))
        
    def get_daily_russian_predictions(self):
        fresh_dataset = self.get_all_data()
        if fresh_dataset is None:
            fresh_dataset = ''
        if len(fresh_dataset) > 0:
            fresh_dataset['prediction'] = self.model.predict_proba(self.transformer.transform(
                fresh_dataset['lemmatize_title']).toarray())[:,1]
            self.article_history = self.article_history.append(fresh_dataset[['url','title','prediction']])
            self.article_history.to_csv(self.path+'russian_article_history_'+self.user_name+'.csv', index=False)
            #fresh_dataset = fresh_dataset[fresh_dataset['prediction']==1]
            return fresh_dataset