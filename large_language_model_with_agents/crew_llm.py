import os
from crewai import Agent, Task, Crew, LLM

GEMINI=os.getenv("GEMINI")

crew_llm = LLM(
    # model="gemini/gemini-1.5-flash",
    # model="gemini/gemini-2.0-flash",
    model="gemini/gemini-2.5-flash",
    api_key=GEMINI,
    max_tokens=500,
    # max_tokens=200,
    temperature=0.7,
    # temperature=0.5
)