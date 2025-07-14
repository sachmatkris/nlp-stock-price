from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import torch
import requests
from bs4 import BeautifulSoup
device = "cuda" if torch.cuda.is_available() else "cpu"
sentiment_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

def extract_date_title(html_object):
    title = html_object.find('a', class_='news-link').get_text(strip=True)
    date = pd.to_datetime(html_object.find('time', class_='latest-news__date')['datetime']).normalize()
    return date, title


def market_news_extractor(ticker, num_of_pages):
    dates, headlines = [], []
    url = "https://markets.businessinsider.com/news/{ticker}-stock?p={page_number}"
    for idx in range(1, num_of_pages):
        response = requests.get(url.format(ticker = ticker.lower(), page_number=idx))
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        latest_news_for_page = soup.find_all('div', class_ = 'latest-news__story')
        for news_article in latest_news_for_page:
            date, title = extract_date_title(news_article)
            if 'disney' in title.lower():
                dates.append(date)
                headlines.append(title)
    return headlines, dates


def add_market_news(df_orig, ticker, num_pages):
    headlines, dates = market_news_extractor(ticker, num_pages)
    df_news_raw = pd.DataFrame({"news": headlines}, index=dates)
    df_news = df_news_raw.groupby(df_news_raw.index).agg(list)
    df_orig['news'] = df_news['news']
    return df_orig



def sentiment_evaluator(headline: str) -> str:
    inputs = sentiment_tokenizer(headline, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
    if probs.tolist()[0][2] > 0.7:
        predicted_class = 'neutral'
    else:
        if probs.tolist()[0][0] > probs.tolist()[0][1]:
            predicted_class = 'positive'
        if probs.tolist()[0][0] <= probs.tolist()[0][1]:
            predicted_class = 'negative'
    if predicted_class == 'positive':
        score = probs.tolist()[0][0]
    elif predicted_class == 'negative':
        score = - probs.tolist()[0][1]
    elif predicted_class == 'neutral':
        score = 0
    return score



def add_sentiment_score(df_orig):
    for idx, headlines_list in df_orig['news'].dropna().items():
        scores_list = []
        for headline in headlines_list:
            score = sentiment_evaluator(headline)
            scores_list.append(score)
        df_orig.at[idx, "sentiment_score"] = np.mean(scores_list) 
    df_orig['sentiment_score'] = df_orig['sentiment_score'].fillna(0) 
    return df_orig