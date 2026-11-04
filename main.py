
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load the API key from .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY missing! Check your .env file."
    )

# Connect to the Groq language model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=api_key
)

# Ask the AI a question
response = llm.invoke(
    "Explain SQL JOIN in simple English with one example."
)

print("\nAI Response:\n")
print(response.content) 
