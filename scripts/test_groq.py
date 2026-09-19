import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

print("API key loaded:", bool(api_key))

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

with open(
    "data/pilot/judge_validation_100.jsonl",
    "r",
    encoding="utf-8"
) as file:
    example = json.loads(file.readline())


print("\nSending first example to Groq...")
print("ID:", example["id"])


response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=1000,
    messages=[
        {
            "role": "system",
            "content": (
                "Evaluate this Bengali instruction-response pair. "
                "Return JSON with complexity and quality, each an integer from 1 to 5."
            )
        },
        {
            "role": "user",
            "content": (
                "Instruction:\n"
                + example["instruction"]
                + "\n\nResponse:\n"
                + example["response"]
            )
        }
    ]
)


print("\n===== RAW RESPONSE =====")
print(response)

print("\n===== MESSAGE =====")
print(response.choices[0].message)

print("\n===== CONTENT =====")
print(repr(response.choices[0].message.content))

print("\n===== MODEL =====")
print(response.model)

print("\nTest completed.")
