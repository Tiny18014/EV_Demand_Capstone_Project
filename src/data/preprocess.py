import pandas as pd
from huggingface_hub import InferenceClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.data.scrape import get_monthly_data
def preprocess(s,e):
    r1 = get_monthly_data("Ashok Leyland", "ASHOKLEY.NS", s, e)
    r2 = get_monthly_data("Hero MotoCorp", "HEROMOTOCO.NS", s, e)
    r3 = get_monthly_data("Mahindra", "M&M.NS", s, e)
    r4 = get_monthly_data("Tata Motors", "TATAMOTORS.NS", s, e)
    r5 = get_monthly_data("TVS Motors", "TVSMOTOR.NS", s, e)
    final = pd.concat([r1, r2, r3, r4, r5], ignore_index=True)
    final['date'] = pd.to_datetime(final['date'], format="%d-%m-%Y")
    final = final.sort_values(by='date', ascending=True)
    client = InferenceClient(
    provider="hf-inference",
    api_key="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",
    )
    def extract_sentiment(text):
        if not text or not isinstance(text, str) or text.strip() == "":
            return None

        try:
            result = client.text_classification(
                text,
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            )

            if isinstance(result, list) and all('label' in x and 'score' in x for x in result):
                # Select the label with max score
                top = max(result, key=lambda x: x['score'])
                polarity = 1 if top['label'].upper() == 'POSITIVE' else -1
                return polarity, polarity * top['score']
            else:
                return None, None

        except Exception as e:
            print(f"Error during inference: {e}")
            return None, None
    final[['sentiment_label', 'distilbertscore']] = final['description'].fillna("").apply(extract_sentiment).apply(pd.Series)
    analyzer = SentimentIntensityAnalyzer()
    def get_vader_score(text):
        try:
            score = analyzer.polarity_scores(str(text))  # in case text is NaN
            return score['compound']
        except:
            return 0.0
    final['vader_compound'] = final['description'].apply(get_vader_score)
    final['vader_momentum'] = final['vader_compound'] - final['vader_compound'].shift(1)
    final['vader_roll_avg'] = final['vader_compound'].rolling(3).mean().fillna(method='bfill')
    std_df = final.groupby('date')['vader_compound'].std().reset_index(name='sentiment_stddev')
    final = final.merge(std_df, on='date', how='left')
    volume_df = (
    final.groupby('date')
            .size()
            .reset_index(name='news_volume')
    )
    volume_df['news_volume'] = volume_df['news_volume'].astype('float64')
    volume_df = volume_df.fillna(method='bfill') 
    final = final.merge(volume_df, on='date', how='left')
    final['trend_delta'] = final['trend_score'] - final['trend_score'].shift(1).fillna(method='bfill')
    final['bert_roll_avg'] = final['distilbertscore'].rolling(3).mean().fillna(method='bfill')
    brand_categories = ['Ashok Leyland', 'Hero MotoCorp', 'Mahindra', 'Tata Motors', 'TVS Motors']
    final['brand'] = pd.Categorical(final['brand'], categories=brand_categories)
    final['brand_encoded'] = final['brand'].cat.codes
    final['sentiment_label_shifted'] = final['sentiment_label'].shift(-1).fillna(method="bfill")
    final['sentiment_label_lag1'] = final['sentiment_label'].shift(1).fillna(method="bfill")
    final['bert_score_lag1'] = final['bert_roll_avg'].shift(1).fillna(method="bfill")
    final['vader_lag1'] = final['vader_compound'].shift(1).fillna(method="bfill")
    final["transition"] = final['sentiment_label_shifted'] * final["sentiment_label"]
    final['vader_trend_interaction'] = final['vader_compound'] * final['trend_delta']
    final['volume_volatility_interaction'] = final['news_volume'] * final['sentiment_stddev']
    final['transition_switch'] = (final['sentiment_label'] != final['sentiment_label'].shift(1)).astype(int)
    final['transition_count_last_3_days'] = final['transition_switch'].rolling(window=3, min_periods=1).sum()
    final = final.dropna()
    x = final[[
        'vader_compound',
        'vader_momentum',
        'vader_roll_avg',
        'sentiment_stddev',
        'news_volume',
        'brand_encoded',
        'trend_delta',
        'bert_roll_avg',
        'sentiment_label_lag1',
        'bert_score_lag1',
        'vader_lag1',
        'transition',
        'vader_trend_interaction',
        'volume_volatility_interaction',
        'transition_count_last_3_days'
    ]]

    y = final["sentiment_label"]
    x.head()
    return x, y


   