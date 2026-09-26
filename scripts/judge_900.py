import json
import os
import time
import re

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")


# ============================================================
# GROQ CLIENT
# ============================================================

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)


# ============================================================
# CONFIG
# ============================================================

MODEL = "openai/gpt-oss-20b"

INPUT_FILE = "data/pilot/pilot_remaining_900.jsonl"

OUTPUT_FILE = "scoring/results/llm_judge_900.jsonl"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a strict Bengali instruction-data quality evaluator.

Your task is to evaluate one Bengali instruction-response pair.

You must return ONLY ONE valid JSON object.

Do NOT use Markdown.
Do NOT use ```json.
Do NOT add any explanation outside the JSON object.

The JSON must have exactly these fields:

{
  "complexity": integer,
  "quality": integer,
  "reason": string
}

============================================================
COMPLEXITY
============================================================

Score the inherent difficulty of the INSTRUCTION.

Do NOT judge complexity based on response length.

1 = Very simple
2 = Simple
3 = Moderate
4 = Difficult
5 = Very difficult

Consider:
- reasoning difficulty
- number of steps
- technical knowledge required
- ambiguity
- mathematical or logical difficulty
- amount of planning required

============================================================
QUALITY
============================================================

Score the quality of the RESPONSE.

Consider:
- correctness
- relevance
- completeness
- clarity
- usefulness
- whether the response actually satisfies the instruction

1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

Quality 5 should be strict.

============================================================
LANGUAGE
============================================================

Evaluate the Bengali instruction-response pair carefully.

Machine-translated, unnatural, repetitive, incomplete, irrelevant,
factually incorrect, or poorly written responses should receive lower
quality scores when appropriate.

============================================================
IMPORTANT
============================================================

Complexity and quality must be judged independently.

Return ONLY valid JSON.

Example:

{
  "complexity": 3,
  "quality": 4,
  "reason": "The instruction requires moderate reasoning and the response is mostly correct and relevant."
}
"""


# ============================================================
# ERROR DETECTION
# ============================================================

def is_tpd_limit_error(error):

    error_text = str(error).lower()

    return (
        "tokens per day" in error_text
        or "tpd" in error_text
        or "token per day" in error_text
    )


# ============================================================
# LOAD INPUT DATA
# ============================================================

rows = []

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        rows.append(json.loads(line))


print(f"Total input rows: {len(rows)}")


# ============================================================
# LOAD EXISTING JUDGED RESULTS
# ============================================================

judged_pairs = set()

if os.path.exists(OUTPUT_FILE):

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                row = json.loads(line)

                pair = (
                    row.get("instruction"),
                    row.get("response")
                )

                judged_pairs.add(pair)

            except json.JSONDecodeError:

                print(
                    "Warning: Invalid JSON line found in output file."
                )

                continue


print(f"Already judged: {len(judged_pairs)}")


# ============================================================
# FIND REMAINING ROWS
# ============================================================

remaining = []

for row in rows:

    pair = (
        row["instruction"],
        row["response"]
    )

    if pair not in judged_pairs:

        remaining.append(row)


print(f"Remaining: {len(remaining)}")


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):

    if not text:

        raise ValueError(
            "Empty model response."
        )

    text = text.strip()

    # --------------------------------------------------------
    # Remove Markdown code fences
    # --------------------------------------------------------

    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"\s*```$",
            "",
            text
        )

        text = text.strip()

    # --------------------------------------------------------
    # Direct JSON parse
    # --------------------------------------------------------

    try:

        return json.loads(text)

    except json.JSONDecodeError:

        pass

    # --------------------------------------------------------
    # Try extracting first JSON object
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:

        raise ValueError(
            f"No JSON object found in model response:\n{text}"
        )

    json_text = text[start:end + 1]

    try:

        return json.loads(json_text)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Could not parse model JSON: {e}\n"
            f"Raw response:\n{text}"
        )


# ============================================================
# VALIDATE RESULT
# ============================================================

def validate_result(result):

    if not isinstance(result, dict):

        raise ValueError(
            "Model output is not a JSON object."
        )

    required_keys = {
        "complexity",
        "quality",
        "reason"
    }

    missing_keys = required_keys - set(result.keys())

    if missing_keys:

        raise ValueError(
            f"Missing required fields: {missing_keys}"
        )

    complexity = result["complexity"]
    quality = result["quality"]
    reason = result["reason"]

    # --------------------------------------------------------
    # Validate complexity
    # --------------------------------------------------------

    if (
        isinstance(complexity, bool)
        or not isinstance(complexity, int)
    ):

        raise ValueError(
            "Complexity must be an integer."
        )

    if complexity < 1 or complexity > 5:

        raise ValueError(
            "Complexity must be between 1 and 5."
        )

    # --------------------------------------------------------
    # Validate quality
    # --------------------------------------------------------

    if (
        isinstance(quality, bool)
        or not isinstance(quality, int)
    ):

        raise ValueError(
            "Quality must be an integer."
        )

    if quality < 1 or quality > 5:

        raise ValueError(
            "Quality must be between 1 and 5."
        )

    # --------------------------------------------------------
    # Validate reason
    # --------------------------------------------------------

    if not isinstance(reason, str):

        raise ValueError(
            "Reason must be a string."
        )

    if not reason.strip():

        raise ValueError(
            "Reason cannot be empty."
        )

    return True


# ============================================================
# JUDGE LOOP
# ============================================================

for index, row in enumerate(
    remaining,
    start=1
):

    instruction = row["instruction"]

    response = row["response"]

    user_prompt = f"""
Instruction:
{instruction}

Response:
{response}
"""

    current_number = len(judged_pairs) + index

    print()

    print(
        f"Judging {current_number}/900 "
        f"(remaining: {len(remaining) - index + 1})"
    )

    success = False

    for attempt in range(3):

        # Reset content for every attempt
        content = None

        try:

            # ------------------------------------------------
            # API CALL
            # ------------------------------------------------

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

                temperature=0.2,

                max_completion_tokens=2000,

                reasoning_effort="low"
            )

            # ------------------------------------------------
            # GET RAW RESPONSE
            # ------------------------------------------------

            message = completion.choices[0].message

            content = message.content

            # ------------------------------------------------
            # EMPTY RESPONSE CHECK
            # ------------------------------------------------

            if not content:

                print(
                    "Empty model response."
                )

                print(
                    "Finish reason:",
                    completion.choices[0].finish_reason
                )

                print(
                    "Reasoning:",
                    getattr(message, "reasoning", None)
                )

                raise ValueError(
                    "Empty model response."
                )

            # ------------------------------------------------
            # PARSE JSON
            # ------------------------------------------------

            result = extract_json(content)

            # ------------------------------------------------
            # VALIDATE JSON
            # ------------------------------------------------

            validate_result(result)

            # ------------------------------------------------
            # PREPARE OUTPUT
            # ------------------------------------------------

            output_row = {

                "instruction": instruction,

                "response": response,

                "llm_complexity": result["complexity"],

                "llm_quality": result["quality"],

                "reason": result["reason"],

                "judge_model": MODEL
            }

            # ------------------------------------------------
            # SAVE IMMEDIATELY
            # ------------------------------------------------

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

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            print(
                f"Complexity: {result['complexity']} | "
                f"Quality: {result['quality']}"
            )

            success = True

            break

        except Exception as e:

            # ------------------------------------------------
            # TPD LIMIT
            # ------------------------------------------------

            if is_tpd_limit_error(e):

                print()

                print(
                    "Groq Tokens-Per-Day limit reached."
                )

                print(
                    "Stopping script immediately."
                )

                print(
                    f"Completed results remain saved in: "
                    f"{OUTPUT_FILE}"
                )

                raise RuntimeError(
                    "Groq Tokens-Per-Day limit reached."
                ) from e

            # ------------------------------------------------
            # OTHER ERROR
            # ------------------------------------------------

            print(
                f"Attempt {attempt + 1}/3 failed:"
            )

            print(e)

            # ------------------------------------------------
            # SHOW RAW RESPONSE
            # ------------------------------------------------

            if content:

                print()
                print("Raw model response:")
                print(content)

            # ------------------------------------------------
            # RETRY
            # ------------------------------------------------

            if attempt < 2:

                print(
                    "Waiting 5 seconds before retry..."
                )

                time.sleep(5)

    # ========================================================
    # FAILURE AFTER 3 ATTEMPTS
    # ========================================================

    if not success:

        print()

        print(
            "FAILED after 3 attempts."
        )

        print(
            f"Example number: {current_number}"
        )

        print(
            "No result was saved for this example."
        )

        raise RuntimeError(
            f"Judge failed for example {current_number}"
        )

    # --------------------------------------------------------
    # NORMAL DELAY
    # --------------------------------------------------------

    time.sleep(10)


# ============================================================
# COMPLETION
# ============================================================

print()

print("======================================")

print(
    "LLM JUDGE COMPLETED"
)

print("======================================")

print(
    f"Total input: {len(rows)}"
)

print(
    f"Previously judged: {len(judged_pairs)}"
)

print(
    f"Processed this run: {len(remaining)}"
)

print(
    f"Output file: {OUTPUT_FILE}"
)

print("======================================")