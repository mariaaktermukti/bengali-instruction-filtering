import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


INPUT_PATH = "data/pilot/judge_validation_100.jsonl"
OUTPUT_PATH = "scoring/results/llm_judge_100.jsonl"

MODEL = "openai/gpt-oss-20b"


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)


def load_jsonl(path):
    records = []

    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def save_result(record):
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


examples = load_jsonl(INPUT_PATH)
previous_results = load_jsonl(OUTPUT_PATH)

completed_ids = set()

for record in previous_results:
    completed_ids.add(record["id"])


print("Total examples:", len(examples))
print("Already judged:", len(completed_ids))
print(
    "Remaining:",
    len(examples) - len(completed_ids)
)


SYSTEM_PROMPT = """
You are a strict evaluator of Bengali instruction-response pairs.

Evaluate the instruction and response directly in Bengali.
Do not translate them into another language before judging.

You must independently assign two scores:

1. COMPLEXITY
Measure how difficult the INSTRUCTION is to satisfy.

1 = Very simple
- Direct factual question
- Simple transformation
- Very short/simple task

2 = Simple
- Straightforward task
- Minor reasoning or formatting required

3 = Moderate
- Multiple steps
- Several requirements
- Some reasoning or interpretation required

4 = Complex
- Multi-step reasoning
- Multiple interacting constraints
- Requires substantial analysis

5 = Highly complex
- Deep reasoning
- Multiple interacting requirements
- Significant planning or sophisticated analysis

Judge complexity from the instruction itself, NOT from the quality of the response.

2. QUALITY
Evaluate how well the RESPONSE satisfies the INSTRUCTION.

1 = Incorrect or unusable
- Wrong, irrelevant, nonsensical, or fails the task

2 = Major problems
- Significant factual, relevance, completeness, or instruction-following problems

3 = Partially correct
- Some useful/correct content but noticeable problems or omissions

4 = Good
- Correct, relevant, clear, and generally complete

5 = Excellent
- Correct, relevant, complete, clear, well-written, and strongly satisfies the instruction

Do not give a high quality score merely because the response is long.
Do not give a low complexity score merely because the response is short.
Evaluate each dimension independently.

Return only the requested structured scores.
"""


def judge_example(example):
    user_prompt = (
        "Evaluate the following Bengali instruction-response pair.\n\n"
        "INSTRUCTION:\n"
        + example["instruction"]
        + "\n\n"
        "RESPONSE:\n"
        + example["response"]
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=1500,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "bengali_instruction_evaluation",
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
        },
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    content = response.choices[0].message.content

    return json.loads(content)


print("\nLLM Judge Started")
print("=" * 60)


for index, example in enumerate(examples):

    example_id = example["id"]

    if example_id in completed_ids:
        continue

    print(
        f"\nJudging {index + 1}/{len(examples)} "
        f"(ID: {example_id})"
    )

    try:

        result = judge_example(example)

        record = {
            "id": example_id,
            "instruction": example["instruction"],
            "response": example["response"],
            "llm_complexity": result["complexity"],
            "llm_quality": result["quality"],
            "llm_reason": result["reason"],
            "judge_model": MODEL
        }

        save_result(record)

        completed_ids.add(example_id)

        print(
            "Complexity:",
            result["complexity"],
            "| Quality:",
            result["quality"]
        )

    except Exception as error:

        print("ERROR:", error)
        print("Waiting 5 seconds before continuing...")

        time.sleep(5)

        continue


print("\n" + "=" * 60)
print("LLM Judge completed.")
print("Results:", OUTPUT_PATH)
print("Total judged:", len(completed_ids))
print("=" * 60)