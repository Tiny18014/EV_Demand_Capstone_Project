import os
from huggingface_hub import InferenceClient
import requests
import streamlit as st

SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
HF_KEY = st.secrets["PESTEL_AGENT"]
SERPER_URL = "https://google.serper.dev/search"

# Define the PESTEL factors
PESTEL_QUERIES = {
    "Political": ["EV subsidies India", "EV regulations India", "EV government policy India"],
    "Economic": ["EV prices India", "EV battery costs", "EV market incentives India"],
    "Social": ["EV adoption India", "EV public sentiment", "EV consumer behavior"],
    "Technological": ["EV battery technology", "EV charging tech", "EV fast charging infrastructure"],
    "Environmental": ["EV climate targets India", "EV CO2 reduction", "EV emissions policy India"],
    "Legal": ["EV compliance regulations India", "EV safety standards India", "EV legal mandates India"]
}

def pestel_classifier():
      
  def scrape_serper(factor_queries, num_results=5):
      """
      Scrape top search results for each PESTEL factor using Serper API.
      Returns a dictionary {Factor: concatenated text of results}.
      """
      factor_texts = {}

      for factor, queries in factor_queries.items():
          all_text = []
          for q in queries:
              payload = {"q": q, "num": num_results}
              headers = {"X-API-KEY": SERPER_API_KEY}
              resp = requests.post(SERPER_URL, headers=headers, json=payload)

              if resp.status_code == 200:
                  data = resp.json()
                  # Serper returns 'organic' results
                  for r in data.get("organic", []):
                      snippet = r.get("snippet", "")
                      if snippet:
                          all_text.append(snippet)
              else:
                  print(f"Error fetching {q}: {resp.status_code}")

          # Join all snippets for this factor
          factor_texts[factor] = " ".join(all_text)

      return factor_texts

  ev_pestel_texts = scrape_serper(PESTEL_QUERIES, num_results=5)
  client = InferenceClient(api_key=HF_KEY)
  completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": f"""
        You are an EV PESTEL classifier.

        Your task:
        - Read the EV-related text.
        - Update the count of occurence of each PESTEL sub-factor.
        - Return ONLY valid JSON.
        - No commentary, no explanation, no natural language.

        TEXT:
        {ev_pestel_texts}

        Output format (same keys, always present):

        {{
          "Political": {{
            "Subsidies": 0.0,
            "Charging Policy": 0.0,
            "Import Duties": 0.0
          }},
          "Economic": {{
            "EV Prices": 0.0,
            "Battery Costs": 0.0,
            "Interest Rates": 0.0
          }},
          "Social": {{
            "Public Sentiment": 0.0,
            "Adoption Narratives": 0.0
          }},
          "Technological": {{
            "Charging Tech": 0.0,
            "Battery Innovation": 0.0,
            "Manufacturing Tech": 0.0
          }},
          "Environmental": {{
            "Emission Norms": 0.0,
            "Climate Targets": 0.0,
            "Sustainability Pressure": 0.0
          }},
          "Legal": {{
            "Compliance Mandates": 0.0,
            "Safety Regulations": 0.0,
            "Fleet Rules": 0.0
          }}
        }}

        Rules:
        - Numbers represent weighted relevance.
        - If a sub-factor is implied but not explicit, assign a small non-zero weight (0.01–0.05).
        - Output ONLY JSON.
        """
            }
        ],
    )
  x = completion.choices[0].message["content"]
  print(x)
  return x


