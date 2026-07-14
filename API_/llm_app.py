from dotenv import load_dotenv
import os
from google import genai
load_dotenv()
API_KEY = os.getenv("gemini_api_key")
client = genai.Client(api_key=API_KEY)
result = client.models.generate_content(model='gemini-3.5-flash', contents='explain llm in 20 words')
print(result.text)