
import os
from dotenv import load_dotenv
load_dotenv()

import os
from groq import Groq

client = Groq(
    api_key=os.getenv("groq_api_key")
)
print(os.getenv("groq_api_key"))

response = client.chat.completions.create(
    model="qwen/qwen3-32b",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France? answer in one word"
        }
    ],
    temperature=0.3,
)

print(response.choices[0].message.content)
