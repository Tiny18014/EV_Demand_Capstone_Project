from datetime import datetime, timedelta
import pandas as pd
from gnews import GNews
from pytrends.request import TrendReq
import yfinance as yf

def get_monthly_data(keyword, ticker, start_date, end_date):
    # 1. Get Google Trends
    pytrends = TrendReq(hl='en-US', tz=330, timeout=(10, 25))
    timeframe = start_date + " " + end_date
    pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')
    trends_df = pytrends.interest_over_time()
    if 'isPartial' in trends_df.columns:
        trends_df.drop(columns=['isPartial'], inplace=True)
    if keyword in trends_df.columns:
        trends_df.rename(columns={keyword: 'trend_score'}, inplace=True)
    trends_df.reset_index(inplace=True)
    trends_df.rename(columns={'date': 'date'}, inplace=True)
    # 2. Get GNews Articles
    start_dated = datetime.strptime(start_date, "%Y-%m-%d")
    end_dated = datetime.strptime(end_date, "%Y-%m-%d")
    gnews = GNews(language='en', country='IN', max_results=100)
    gnews.start_date = (start_dated.year, start_dated.month, start_dated.day)
    gnews.end_date = (end_dated.year, end_dated.month, end_dated.day)
    articles = gnews.get_news(keyword)
    news_df = pd.DataFrame(articles)
    if not news_df.empty:
        news_df = news_df[['description', 'published date']]
        news_df.rename(columns={'published date': 'date'}, inplace=True)
        news_df['date'] = pd.to_datetime(news_df['date'], format="mixed").dt.date
        news_df['date'] = pd.to_datetime(news_df['date'])
        news_df['news_count'] = 1
        news_daily = news_df.groupby('date').agg({
            'description': lambda texts: " ".join(texts),
            'news_count': 'sum'
        }).reset_index()
    else:
        news_daily = pd.DataFrame(columns=['date', 'description', 'news_count'])
    # 3. Get Stock Data
    stock_data = yf.download(ticker, start=start_dated, end=end_dated, progress=False)
    stock_data = stock_data.reset_index()[['Date', 'Close']]
    stock_data.columns = ['date', 'close_price']
    # 4. Merge all three
    df = pd.merge(trends_df, news_daily, on='date', how='inner')
    df = pd.merge(df, stock_data, on='date', how='inner')
    df['brand'] = keyword
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df.info())
    return df
