import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Run `source init.sh` to use local prometheus_eval
from prometheus_eval import PrometheusEval
from prometheus_eval.litellm import AsyncLiteLLM, LiteLLM
from prometheus_eval.mock import AsyncMockLLM, MockLLM
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
from prometheus_eval.vllm import VLLM
from transformers import AutoTokenizer

flask_eval_prompt = {
    "logical_correctness": """Score 1: The model’s final answer is completely incorrect and lacks sound reasoning.
Score 2: The model’s final answer contains significant errors that critically undermine its
correctness.
Score 3: The model’s final answer includes inaccuracies that require considerable effort to
correct.
Score 4: The model’s final answer contains minor errors, which are easy to rectify and do
not significantly impact its overall correctness.
Score 5: The model’s final answer is completely accurate and sound.""",
    "factuality": """Score 1: The model did not extract pertinent background knowledge and provided inaccurate
or misleading information. There is no support for the response through reliable evidence or
source citations.
Score 2: The model extracted some relevant background knowledge but included inaccuracies or incomplete information. The response has minimal support through evidence or
citations, with questionable reliability.
Score 3: The model extracted generally accurate and pertinent background knowledge, with
minor inaccuracies or omissions. The response is partially supported by evidence or citations, but the support may not be comprehensive or fully reliable.
Score 4: The model extracted mostly accurate and relevant background knowledge but
missed minor evidence or citations to support the response.
Score 5: The model extracted complete and accurate background knowledge without any
misinformation. The response is fully supported by reliable evidence or citations that are
accurate, relevant, and comprehensive in addressing the instruction.""",
    "comprehension": """Score 1: The response is completely unrelated to the instruction, or the model entirely misunderstands the instruction.
Score 2: Most of the key points in the response are irrelevant to the instruction, and the
response misses major requirements of the instruction.
Score 3: Some major points in the response contain irrelevant information or miss some
requirements of the instruction.
Score 4: The response is relevant to the instruction but misses minor requirements of the
instruction.
Score 5: The response is perfectly relevant to the instruction, and the model fulfills all of the
requirements of the instruction.""",
    "completeness": """Score 1: The response doesn’t include any specifics or examples to support the statements
made.
Score 2: The response does not provide sufficient details or supportive examples, requiring
a major effort to make the response more complete.
Score 3: It is a decent response, but the breadth and depth of the response are rather limited.
The details and examples used to substantiate the response may be insufficient.
Score 4: The response provides detailed explanations, but there is room for enhancement.
The response could be further improved by including more details and supportive examples.
Score 5: The response fully provides comprehensive explanations. It delves deep into the
topic, providing as much detail as possible, and it offers several examples to back up its
points.""",
    "insightfulness": """Score 1: The response is overly simplistic, lacking any originality or novelty.
Score 2: The ideas or perspectives within the response are commonplace, demonstrating a
lack of originality or novelty.
Score 3: Some may perceive the response as original and novel, but others may find it
ordinary or uninspiring.
Score 4: The response includes some innovative perspectives or ideas that require thoughtful
consideration, yet they aren’t particularly surprising.
Score 5: The response is infused with surprisingly creative perspectives or ideas that are
challenging to conceive, showcasing significant originality and novelty."""
}

ABSOLLUTE_REFINE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.
5. Don't get confused. You are conducting an absolute grading of another model's grading! For convenience, I will seperate the input and output of the other model's relative grading with "@@@"s.

@@@
###The instruction to evaluate:
{orig_instruction}
@@@

###Response to evaluate:
{orig_response}

###Reference Answer (Score 5):
{orig_reference_answer}

###Score Rubrics:
{score_rubric}

###Feedback: """


def read_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def init_eval():
    # records: Full data that has all the information of BiGGen-Bench
    records = []
    # necessary for PrometheusEval
    instructions = []
    responses = []
    reference_answers = []
    rubric = []
    return records, instructions, responses, reference_answers, rubric


def main(args):
    load_dotenv()

    input_file_path = args.input_file_path
    output_file_path = args.output_file_path
    eval_model_name = args.model_name
    flask_eval_type = args.flask_eval_type

    is_prometheus = False

    if eval_model_name in [
        "prometheus-eval/prometheus-7b-v2.0",
        "prometheus-eval/prometheus-8x7b-v2.0",
        "prometheus-eval/prometheus-bgb-8x7b-v2.0",
    ]:
        is_prometheus = True

    input_data = read_json(input_file_path)
    input_data_list = list(input_data.values())

    assert isinstance(input_data, dict), "Wrong input file data format"
    assert isinstance(input_data_list, list), "Wrong input file data format"
    assert isinstance(input_data_list[0], dict), "Wrong input file data format"
    assert len(input_data_list[0].keys()) == 10, "Wrong input file data format"

    records, instructions, responses, reference_answers, rubric = init_eval()
    records_2, instructions_2, responses_2, reference_answers_2, rubric_2 = init_eval()

    for instance_id, instance in input_data.items():
        record = instance

        if "llm_judge" in record["id"]:
            if is_prometheus:
                continue
            records_2.append(record)
            instructions_2.append(record["input"])
            responses_2.append(record["response"])
            reference_answers_2.append(record["reference_answer"])
            score_rubric = flask_eval_prompt[flask_eval_type]
            rubric_2.append(score_rubric)
        else:
            records.append(record)
            instructions.append(record["input"])
            responses.append(record["response"])
            reference_answers.append(record["reference_answer"])
            score_rubric = flask_eval_prompt[flask_eval_type]
            rubric.append(score_rubric)

    assert (
        len(records)
        == len(instructions)
        == len(responses)
        == len(reference_answers)
        == len(rubric)
    ), "Data mismatch"
    assert (
        len(records_2)
        == len(instructions_2)
        == len(responses_2)
        == len(reference_answers_2)
        == len(rubric_2)
    ), "Data mismatch"

    if is_prometheus:
        model = VLLM(eval_model_name, gpu_memory_utilization=0.9, max_model_len=8192)
        judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)
    else:
        model = MockLLM(mode="absolute")
        # model = AsyncLiteLLM(eval_model_name, batch_size=100, requests_per_minute=100)
        judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    feedbacks, scores = judge.absolute_grade(
        instructions=instructions,
        responses=responses,
        rubric=rubric,
        reference_answers=reference_answers,
    )

    if not is_prometheus:
        judge.absolute_grade_prompt = ABSOLLUTE_REFINE_PROMPT
        feedbacks_2, scores_2 = judge.absolute_grade(
            instructions=instructions_2,
            responses=responses_2,
            rubric=rubric_2,
            reference_answers=reference_answers_2,
        )

        # Extend the feedbacks and scores
        records.extend(records_2)
        feedbacks.extend(feedbacks_2)
        scores.extend(scores_2)

    result = {}

    for record, output in zip(records, zip(feedbacks, scores)):
        instance_id = record["id"]

        result[instance_id] = record.copy()
        result[instance_id]["feedback"] = output[0]
        result[instance_id]["score"] = output[1]
        result[instance_id]["eval_model_name"] = eval_model_name

    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    with output_file_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="prometheus-eval/prometheus-7b-v2.0",
        help="Model name to use as evaluation model",
    )
    parser.add_argument(
        "--flask_eval_type",
        type=str,
        default="logical_correctness"
    )
    parser.add_argument(
        "--input_file_path",
        type=str,
        required=True,
        help="Path to the input response file",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        required=True,
        help="Path to save the output feedback file",
    )

    args = parser.parse_args()

    main(args)
