import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


INPUT_FILE = "data/pilot/pilot_remaining_900.jsonl"
OUTPUT_FILE = "scoring/results/llm_judge_900.jsonl"

MODEL = "openai/gpt-oss-20b"


SYSTEM_PROMPT = """
তুমি একটি Bengali Instruction-Response Quality এবং Complexity Judge।

তোমার কাজ হলো একটি Bengali instruction এবং তার response দেখে
দুটি আলাদা score দেওয়া:

1. Complexity: instruction কতটা reasoning, planning, multi-step thinking,
   technical understanding বা problem-solving দাবি করে।

2. Quality: response instruction অনুযায়ী কতটা correct, relevant, complete,
   clear এবং useful।

খুব গুরুত্বপূর্ণ:
- শুধু ভাষা সুন্দর হলেই quality বেশি হবে না।
- শুধু বড় response হলেই quality বেশি হবে না।
- response দীর্ঘ হলেই complexity বেশি হবে না।
- Bengali grammar বা wording-এর ছোটখাটো সমস্যা থাকলে শুধু এজন্য score কমাবে না,
  যদি মূল content সঠিক হয়।
- একইভাবে সুন্দর ভাষা থাকলেও যদি answer ভুল, incomplete বা instruction-following না হয়,
  তাহলে quality বেশি দেবে না।
- Complexity এবং Quality সম্পূর্ণ আলাদা বিষয়।
- Complexity নির্ধারণ করবে মূল instruction দেখে।
- Quality নির্ধারণ করবে response instruction-এর চাহিদা কতটা পূরণ করেছে দেখে।

--------------------------------
COMPLEXITY SCALE
--------------------------------

1 = Very Simple
- সরাসরি factual question
- একটি ছোট তথ্য/সংজ্ঞা চাওয়া
- প্রায় কোনো reasoning প্রয়োজন নেই

2 = Simple
- সাধারণ explanation
- ছোটখাটো calculation
- basic comparison
- এক বা দুইটি সহজ reasoning step

3 = Moderate
- একাধিক reasoning step প্রয়োজন
- analysis বা structured explanation দরকার
- কয়েকটি condition/constraint বিবেচনা করতে হয়
- সাধারণ কিন্তু multi-step problem

4 = Complex
- substantial multi-step reasoning
- multiple constraints
- planning, synthesis বা গভীর analysis প্রয়োজন
- একাধিক concept একসাথে ব্যবহার করতে হয়

5 = Highly Complex
- খুব কঠিন multi-step reasoning
- extensive planning/synthesis
- অনেকগুলো interdependent constraints
- advanced technical বা research-level reasoning প্রয়োজন

IMPORTANT:
শুধু response বড় বা detailed হলেই complexity বাড়াবে না।
Instruction-এর inherent difficulty বিবেচনা করবে।

--------------------------------
QUALITY SCALE
--------------------------------

1 = Very Poor
- incorrect বা unusable
- instruction-এর মূল উদ্দেশ্য পূরণ করে না
- major factual/logical errors
- খুব গুরুত্বপূর্ণ অংশ missing

2 = Poor
- কিছু useful content আছে
- কিন্তু major errors, missing information বা instruction-following সমস্যা আছে
- response সম্পূর্ণভাবে নির্ভরযোগ্য নয়

3 = Moderate
- মূল প্রশ্নের উত্তর দেয়
- কিন্তু noticeable সমস্যা আছে
- কিছু information missing, incomplete reasoning, minor-to-moderate errors,
  বা clarity সমস্যা থাকতে পারে

4 = Good
- correct
- relevant
- clear
- instruction যথেষ্ট ভালোভাবে follow করে
- minor সমস্যা থাকলেও overall response useful এবং reliable

5 = Excellent
- correct
- highly relevant
- complete
- clear এবং well-structured
- instruction-এর সব গুরুত্বপূর্ণ অংশ পূরণ করে
- কোনো meaningful factual বা logical problem নেই
- response genuinely high-quality

IMPORTANT CALIBRATION RULE:
Quality 5 খুব strictভাবে ব্যবহার করবে।

একটি response-কে 5 দিতে হলে:
- মূল content correct হতে হবে
- instruction-এর গুরুত্বপূর্ণ requirements পূরণ করতে হবে
- response যথেষ্ট complete হতে হবে
- meaningful factual/logical সমস্যা থাকা যাবে না

শুধু ভালো, useful বা mostly correct হলেই 5 দেবে না।
সেক্ষেত্রে 4 বিবেচনা করবে।

একইভাবে:
- কিছু সমস্যা থাকলে 5 দেবে না
- incomplete হলে 5 দেবে না
- major omission থাকলে 5 দেবে না
- partially correct হলে 3 বা 4 বিবেচনা করবে

--------------------------------
SCORING PROCEDURE
--------------------------------

প্রথমে instruction-এর difficulty বিচার করো।
তারপর response-এর correctness, relevance, completeness এবং clarity বিচার করো।

দুটি score independentভাবে নির্ধারণ করো।

নিজের সিদ্ধান্ত পরিবর্তন করবে না শুধু response-এর length বা writing style দেখে।

শুধুমাত্র valid JSON output দেবে:

{
  "complexity": 1,
  "quality": 1,
  "reason": "সংক্ষিপ্ত কারণ"
}

complexity এবং quality অবশ্যই integer হবে এবং 1 থেকে 5-এর মধ্যে হবে।
"""


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)


rows = load_jsonl(INPUT_FILE)


# Resume support
judged_pairs = set()

if os.path.exists(OUTPUT_FILE):
    existing_rows = load_jsonl(OUTPUT_FILE)

    for row in existing_rows:
        judged_pairs.add(
            (
                row["instruction"],
                row["response"]
            )
        )


remaining = []

for row in rows:
    pair = (
        row["instruction"],
        row["response"]
    )

    if pair not in judged_pairs:
        remaining.append(row)


print("=" * 60)
print("BENGALI LLM JUDGE - 900 PILOT")
print("=" * 60)

print(f"Total examples: {len(rows)}")
print(f"Already judged: {len(judged_pairs)}")
print(f"Remaining:      {len(remaining)}")
print()


for index, row in enumerate(remaining, start=1):

    instruction = row["instruction"]
    response = row["response"]

    user_prompt = f"""
Instruction:
{instruction}

Response:
{response}
"""

    print(
        f"Judging {index}/{len(remaining)}"
    )

    success = False

    for attempt in range(3):

        try:

            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0,
                max_tokens=1500,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "bengali_judge",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "complexity": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 5
                                },
                                "quality": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 5
                                },
                                "reason": {
                                    "type": "string"
                                }
                            },
                            "required": [
                                "complexity",
                                "quality",
                                "reason"
                            ],
                            "additionalProperties": False
                        }
                    }
                }
            )

            content = completion.choices[0].message.content

            result = json.loads(content)

            output_row = {
                "instruction": instruction,
                "response": response,
                "llm_complexity": result["complexity"],
                "llm_quality": result["quality"],
                "reason": result["reason"],
                "judge_model": MODEL
            }

            with open(
                OUTPUT_FILE,
                "a",
                encoding="utf-8"
            ) as f:

                f.write(
                    json.dumps(
                        output_row,
                        ensure_ascii=False
                    ) + "\n"
                )

            print(
                f"Complexity: {result['complexity']} | "
                f"Quality: {result['quality']}"
            )

            success = True
            break

        except Exception as e:

            print(
                f"Attempt {attempt + 1}/3 failed: {e}"
            )

            if attempt < 2:
                print("Waiting 10 seconds before retry...")
                time.sleep(10)

    if not success:

        print("FAILED after 3 attempts.")
        print("Stopping so no example is silently skipped.")

        raise RuntimeError(
            f"Judge failed for example {index}"
        )

    # Avoid hitting Groq TPM limit
    time.sleep(10)


print()
print("=" * 60)
print("LLM JUDGE COMPLETED")
print("=" * 60)

final_rows = load_jsonl(OUTPUT_FILE)

print(f"Total judged: {len(final_rows)}")
print(f"Saved: {OUTPUT_FILE}")
print("=" * 60)