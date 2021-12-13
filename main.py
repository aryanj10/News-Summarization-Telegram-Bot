import telebot
import nltk
from nltk import sent_tokenize,word_tokenize
#nltk.download('punkt')
import numpy as np
from sentence_transformers import SentenceTransformer
import math
from newspaper import Article
from gnews import GNews
import time

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def convert_to_embeddings(hypothesis):
  text = hypothesis
  hypothesis_embeddings = {}
  hypothesis = sent_tokenize(hypothesis)
  sum = 0
  for sent in hypothesis:
    var = sbert_model.encode([sent])[0]
    sum+=var
    hypothesis_embeddings[sent] = var

  centroid = sum/len(hypothesis)
  #print("done")
  #print(hypothesis_embeddings)
  content_relevance_score = content_relevance(centroid,hypothesis_embeddings)
  sentence_novelty_score = sentence_novelty(content_relevance_score,hypothesis_embeddings)
  sentence_position_score = sentence_position(hypothesis)
  return total_score(content_relevance_score,sentence_novelty_score,sentence_position_score,text)

def content_relevance(centroid,hypothesis_embeddings):
  d = {}
  for sentence,embed in hypothesis_embeddings.items():
    d[sentence] = cosine(centroid,embed)
  return d
def sentence_novelty(content_relevance_score,hypothesis_embeddings):
  novel_sentences = {}
  TAU = 0.95
  for sent1,embed1 in hypothesis_embeddings.items():
    max_similarity = 0
    for sent2,embed2 in hypothesis_embeddings.items():
      if sent1!=sent2 and cosine(embed1,embed2)>max_similarity:
          max_similarity = cosine(embed1,embed2)

    if max_similarity<TAU:
      novel_sentences[sent1] = 1

    if max_similarity>TAU:
      if content_relevance_score[sent1]>content_relevance_score[sent2]:
        novel_sentences[sent1] = 1
      else:
        novel_sentences[sent2] = 1

    else:
      novel_sentences[sent1] = 1-max_similarity
  return novel_sentences
def sentence_position(hypothesis):
  score_sent = {}
  for i,sent in enumerate(hypothesis):
    score_sent[sent] = max(0.5,math.exp(-(i+1)/(len(hypothesis)**(1/3))))
  return score_sent
def total_score(content_relevance_score,sentence_novelty_score,sentence_position_score,text):
  ALPHA = 0.6
  BETA = 0.2
  GAMMA = 0.2
  final_score = {}
  for sent in content_relevance_score:
    final_score[sent] = ALPHA*content_relevance_score[sent]+BETA*sentence_novelty_score[sent]+GAMMA*sentence_position_score[sent]
  final_score = {k: v for k, v in sorted(final_score.items(), key=lambda item: item[1],reverse=True)}
  summary = ""
  for sent in final_score:
    if len(summary)<len(text)//4: #making summary of approx half length.
      summary+=sent
  return summary



API_KEY='1'

bot = telebot.TeleBot(API_KEY,parse_mode=None)
print("**Bot initiated**")

@bot.message_handler(commands=['start'])
def greet(message):
  bot.send_message(message.chat.id, "Hi, Welcome to News On the Go \n1. Enter /news to get summarized news \n2. Enter /latest to get top 10 summarized news")

@bot.message_handler(commands=['news'])
def cowin(message):
  bot.send_message(message.chat.id,"Enter command as *news Topic* for example *news Latest News*",parse_mode="Markdown")

@bot.message_handler(commands=['latest'])
def latest(message):
  try:
    request = message.text[5::]
    google_news = GNews(language='en', country='IN', period='7d', max_results=10)
    json = google_news.get_news('latest news')
    # print(json[0]['url'])
    for i in json:
      article = Article(i['url'], 'en')
      article.download()
      article.parse()
      text = article.text
      actual_summary = article.summary
      predicted_summary = convert_to_embeddings(text)
      print(str(i['title'])+'\n' +str(predicted_summary)+ '\n Read the full article here \n '+i['url'])
      bot.send_message(message.chat.id,str(i['title'])+'\n\n' +str(predicted_summary)+ '\n\n Read the full article here \n '+i['url'] )
      time.sleep(5)
  except:
    bot.send_message(message.chat.id,'No Data Found')

def news_request(message):
  request = message.text.split()
  if len(request) < 2 or request[0].lower() not in "news":
    return False
  else:
    return True

@bot.message_handler(func=news_request)
def news_name(message):
  try:
    request = message.text[5::]
    google_news = GNews(language='en', country='IN', period='7d', max_results=10)
    json = google_news.get_news(request)
    # print(json[0]['url'])
    for i in json:
      article = Article(i['url'], 'en')
      article.download()
      article.parse()

      text = article.text
      actual_summary = article.summary
      predicted_summary = convert_to_embeddings(text)
      print(str(i['title'])+'\n' +str(predicted_summary)+ '\n Read the full article here \n '+i['url'])
      bot.send_message(message.chat.id,str(i['title'])+'\n\n' +str(predicted_summary)+ '\n\n Read the full article here \n '+i['url'] )
      time.sleep(5)


  except:
    bot.send_message(message.chat.id,'No Data Found')

bot.polling()