import os
from textwrap import dedent
from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from agno.tools.serper import SerperTools
import streamlit as st
openaii = st.secrets["CHARGING_AGENT"]


charging_intelligence_agent = Agent(
    model=HuggingFace(id="openai/gpt-oss-20b", api_key=openaii, temperature=0.25),
    description="You analyze EV forecast data and summarize trends.",
    instructions=dedent("""
        You receive EV forecast data as JSON.

        Provide:
        - Key growth trends
        - Energy demand patterns
        - Vehicle-type insights

        RULES:
        - Use ONLY the dataset provided.
        - Output MUST be bullet points.
        - Do NOT exceed 80 words.
        - Do NOT show calculations.
        - Do NOT include paragraphs.
        - Do NOT add references.
        - Keep the summary business-focused and concise.
        
        Expected Output Format:
        ##### Expected Trends (2026-2027)
    """),
    markdown=True,
)

