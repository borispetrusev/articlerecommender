from articlerecom import ArticleRecommender
from goose3 import Goose
from tldry import TLDRy
import pandas as pd
import telebot

#PATH = ''
PATH = ''
user = ''
token = ''
chat_id = None

bot = telebot.TeleBot(token)

def get_predictions(user, PATH):
    recommender = ArticleRecommender(user, PATH)
    recommender.get_all_predictions()
    recommender.combined_df.to_csv(PATH+'predicted_articles_'+user+'.csv', index=False)

def get_article_text(url, lang):
    if (lang=='english'):
        g = Goose({'use_meta_language': False, 'target_language':'en'})
    else:
        g = Goose({'use_meta_language': False, 'target_language':'ru'})
    return g.extract(url=url).cleaned_text

def get_summary(text, lang):
    if (lang=='english'):
        summarizer = TLDRy(language='english',
                           min_sent_len=5,
                           min_df=2)
    else:
        summarizer = TLDRy(language='russian',
                           min_sent_len=5,
                           min_df=2)
    try:
        summ = '\n$$$\n'.join(summarizer.summarize(text, topn=3))
        return summ
    except:
        return text

def get_next_article(user, PATH):
    try:
        df = pd.read_csv(PATH+'predicted_articles_'+user+'.csv', index_col=None)
        if len(df) > 0:
            current_article = df.iloc[0]
            if current_article['type'] in ['techcrunch','hackernews','ycombinator']:
                lang = 'english'
            else:
                lang = 'russian'
            url = current_article['url']
            title = current_article['title']
            text = get_article_text(url, lang)
            summary = get_summary(text, lang)
            return url, title, get_summary(get_article_text(
                current_article['url'],lang),lang)
        return None,None,'No new articles'
    except:
        return None,None,'No new articles'

def update_training_set(user, PATH, vote):
    df = pd.read_csv(PATH+'predicted_articles_'+user+'.csv', index_col=None)
    row = df.iloc[0]
    if row['type'] in ['techcrunch','hackernews','ycombinator']:
        lang = 'english'
    else:
        lang = 'russian'
    try:
        training_dataset = pd.read_csv(PATH+lang+'_training_'+user+'.csv')
    except:
        training_dataset = pd.DataFrame(columns=['url','title','is_interesting',
                                                 'lemmatize_title'])
    training_dataset = training_dataset.append({'url':row['url'],
                                               'title':row['title'],
                                               'is_interesting':vote,
                                               'lemmatize_title':row['lemmatize_title']}, 
                                               ignore_index=True).reset_index(drop=True)
    training_dataset.to_csv(PATH+lang+'_training_'+user+'.csv', index=False)
    df = df.drop([0]).reset_index(drop=True)
    df.to_csv(PATH+'predicted_articles_'+user+'.csv', index=False)

@bot.message_handler(content_types=["text"])
def main(message):
    if message.chat.id == chat_id:
        action = message.text
        if action == '/get_new_articles':
            get_predictions(user, PATH)
            bot.send_message(message.chat.id,"Articles are refreshed")
        elif action == '/get_next':
            url, title, text = get_next_article(user, PATH)
            if text == 'No new articles':
                bot.send_message(message.chat.id,"No new articles. Refresh?")
                bot.send_message(message.chat.id,"/get_new_articles")
            else:
                if title != '':
                    bot.send_message(message.chat.id,title)
                if text != '':
                    bot.send_message(message.chat.id,text)
                if url != '':
                    bot.send_message(message.chat.id,url)
                bot.send_message(message.chat.id,'Vote - up or down?')
                bot.send_message(message.chat.id,'/up')
                bot.send_message(message.chat.id,'/down')
        elif action == '/up':
            update_training_set(user, PATH, 1)
        elif action == '/down':
            update_training_set(user, PATH, 0)
        else:
            bot.send_message(message.chat.id,"Incorrect command. Correct commands are:")
            bot.send_message(message.chat.id,"/get_new_articles")
            bot.send_message(message.chat.id,"/get_next")


bot.polling(none_stop=True)
