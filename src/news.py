from tavily import TavilyClient
from huggingface_hub import InferenceClient
from datetime import datetime, timedelta
import os
#importing from env and not githubsecrets
def fetch_news_data():
    today = datetime.today()
    end_date = today
    start_date = end_date - timedelta(days=7)
    apii_key = os.environ.get("TAVILY_API_KEY_1")
    news_api = os.environ.get("NEWS_API_KEY_1")
    client = TavilyClient(apii_key)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    response = client.search(
        query="give me the recent news of electric vehicles from india",
        include_answer="basic",
        max_results=10,
        start_date=start_str,
        end_date=end_str,
        country="india",
        include_domains=["https://timesofindia.indiatimes.com/","https://www.deccanherald.com/","https://indianexpress.com/","https://www.thehindu.com/"]
    )
    data = " ".join(item['content'] for item in response['results'] if 'content' in item)
    client = InferenceClient(
    provider="novita",
    api_key=news_api,
    )

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": f"You want users to stay up to date with news about eletric vehicles in India. "
                           f"While your primary context for news articles is {data}, understand it well enough to highlight both the positive and negative aspects in the news if any"
                           f"Your context is time bound between {start_str} and {end_str}. So give it as a weekly news update in clear, easy to understand language. Don't make it too short, but be brief, and engaging in your delivery."
            }
        ],
    )

    answer = completion.choices[0].message["content"]
    return answer