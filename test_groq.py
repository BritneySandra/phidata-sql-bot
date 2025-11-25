from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

resp = client.chat.completions.create(
    model="qwen/qwen3-32b",
    messages=[
        {"role": "system", "content": "Reply with only JSON"},
        {"role": "user", "content": "Give me JSON: {\"x\":1}"}
    ],
    temperature=0
)

print(resp.choices[0].message.content)
