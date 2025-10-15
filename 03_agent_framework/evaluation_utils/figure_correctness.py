import os
import mlflow
import json
from openai import OpenAI
from mlflow.entities import Document

from typing import List, Dict, Any, cast
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback
from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback

judge_system_prompt = """
System Prompt: LLM Judge for Numerical Accuracy (Two Decimal Place Precision)

You are an impartial judge tasked with verifying the numerical accuracy of a generated response compared to a ground truth value. Your goal is to determine whether the generated number is numerically correct within two decimal places of the ground truth.

Evaluation Rules:
* Round both the generated value and the ground truth to two decimal places.
* If the two rounded values are exactly equal, the answer is Correct.
* If the rounded values differ, the answer is Incorrect.

Special Notes:
* Use standard rounding (i.e., round half up: 1.005 rounds to 1.01).
* Ignore any other content in the response — only evaluate numeric correctness.
* If either value is missing, non-numeric, or malformed, return Incorrect.

Your output MUST be a single valid JSON object with two keys: "score" (an integer, 0 or 1) and "rationale" (a string).
Example:
{"score": 1, "rationale": "The ground truth was 1.02 and the generated responsee was 1.01999999, which is more precise"}
"""

judge_user_prompt = """
Please evaluate the AI's Response below based on the ground truth provided.

Ground truth:
```{ground_truth}```

AI's Response:
```{llm_response_from_app}```

Provide your evaluation strictly as a JSON object with "score" and "rationale" keys.
"""

# Define a custom scorer that wraps the guidelines LLM judge to check if the response follows the policies
@scorer
def figure_correctness(expectations: dict[str, Any], outputs: str) -> Feedback:
    ground_truth = expectations["expected_response"]

    client = OpenAI(
            api_key=os.environ["DATABRICKS_TOKEN"],
            base_url=f"{os.environ['DATABRICKS_URL']}/serving-endpoints"
        )

    # Call the Judge LLM using the OpenAI SDK client.
    judge_llm_response_obj = client.chat.completions.create(
        model="databricks-claude-sonnet-4",  # This example uses Databricks hosted Claude. If you provide your own OpenAI credentials, replace with a valid OpenAI model e.g., gpt-4o-mini, etc.
        messages=[
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": judge_user_prompt.format(ground_truth=ground_truth, llm_response_from_app=outputs)},
        ],
        max_tokens=200,  # Max tokens for the judge's rationale
        temperature=0.0, # For more deterministic judging
    )
    judge_llm_output_text = judge_llm_response_obj.choices[0].message.content

    # Parse the Judge LLM's JSON output.
    judge_eval_json = json.loads(judge_llm_output_text)
    parsed_score = int(judge_eval_json["score"])
    parsed_rationale = judge_eval_json["rationale"]

    return Feedback(
        value=parsed_score,
        rationale=parsed_rationale,
        # Set the source of the assessment to indicate the LLM judge used to generate the feedback
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id="claude-3-7-sonnet",
        )
    )