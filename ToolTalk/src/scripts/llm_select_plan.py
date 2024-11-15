import argparse
import json
import os
import copy
import time
import anthropic
from openai import OpenAI


from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from tooltalk.apis import ALL_APIS
from tooltalk.utils.file_utils import get_names_and_paths
from tooltalk.utils.eval_utils import get_all_conversations, build_conversation_str, build_tool_options_str, extract_plan
from tooltalk.evaluation.tool_executor import ToolExecutor
from tooltalk.utils.cohere_utils import co
from tooltalk.utils.anthropic_utils import _messages_complete
from tooltalk.utils.mistral_utils import chat_with_backoff


TOOLTALK_EVAL_ROOT = Path(__file__).resolve().parents[2]


def get_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-r", dest="results_dir", type=str, nargs="+", help="Path to results directory")
    parser.add_argument("-m", dest="model", type=str, nargs="+", help="Model in directory to use")
    parser.add_argument("-c", dest="class_type", choices=["command-r-plus", "command-r-plus-refresh", "mistral", "claude", "gpt-4o", "mbr"], default="command-r-plus", help="Type of classification to use")
    parser.add_argument("-e", dest="extract_plans", action="store_true", help="Extract plans from logs")
    return parser

user_proxy_prompt = """You are tasked with selecting the best course of action that an AI assistant should take in order to meet a user's request.
Below, you are provided with a list of the tools that the AI assistant has at its disposal, as well as the conversation metadata and interaction history between a user and an AI assistant.

## Tools
{tools}

## Metadata
{metadata}

## Conversation history
{conversation}

## Instructions
Please select the best course of action that the AI assistant should take in order to meet the user's request from the options below. Respond only with the number of the option you choose.

## Plans
{plans}

## Selected plan:"""



def mistral_generate(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    response = chat_with_backoff("mistral-large-2407", messages)
    response_numbers = [x for x in response.choices[0].message.content if x.isdigit()]
    if len(response_numbers) == 0:
        raise ValueError("No number in response")
        # print("No number in response")
        # return 1
        
    first_num = response_numbers[0]
    return first_num

def gpt_generate(prompt: str, model: str, temperature: float) -> str:
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    # get first integer from response
    response_numbers = [x for x in response.choices[0].message.content if x.isdigit()]
    if len(response_numbers) == 0:
        raise ValueError("No number in response")
        # print("No number in response")
        # return 1
        
    first_num = response_numbers[0]
    print(f"Selected plan: {first_num}")
    
    return first_num


def claude_generate(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    response = _messages_complete(
        messages,
        10,
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        0
    )
    # get first integer from response
    response_numbers = [x for x in response.content[0].text if x.isdigit()]
    if len(response_numbers) == 0:
        raise ValueError("No number in response")
        
    first_num = response_numbers[0]
    
    return first_num

def cohere_generate(prompt: str, model: str, temperature: float) -> str:
    kwargs = {}
    if temperature == 0:
        # anecdote suggests this is useful for greedy decoding
        kwargs["k"] = 1
    response = co.chat(
        message=prompt, model=model, temperature=temperature, **kwargs #raw_prompting=True, 
    )

    # get first integer from response
    response_numbers = [x for x in response.text if x.isdigit()]

    if len(response_numbers) == 0:
        print(response.text)
        raise ValueError("No number in response")
        
    first_num = response_numbers[0]
    
    return first_num


def get_most_similar_plan(
        plan_options: Dict[str, Dict[str, str]],
):
    time.sleep(1)
    plan_options_list = list(plan_options.keys())
    response = co.embed(texts=plan_options_list, model="embed-english-v3.0", input_type="classification")
    plan_embeddings = response.embeddings
    plan_similarities = np.zeros((len(plan_options_list), len(plan_options_list)))
    for i, emb1 in enumerate(plan_embeddings):
        for j, emb2 in enumerate(plan_embeddings):
            plan_similarities[i, j] = cosine_similarity(np.array(emb1).reshape(1, -1), np.array(emb2).reshape(1, -1))[0][0]

    plan_similarities = np.mean(plan_similarities, axis=1)
    selected_idx = np.argmax(plan_similarities)
    selected_plan = plan_options_list[selected_idx]
    return plan_options[selected_plan][0]


def classify_plan_options(
        plan_options: Dict[str, Dict[str, str]],
        classification_type: str,
        ground_truth_history: List[Dict[str, str]],
        metadata: Dict[str, str],
) -> Dict[str, str]:

    if classification_type == "mbr":
        # Get the similarity between each pair of plans and select the plan with the highest (average) similarity
        return get_most_similar_plan(plan_options)
        
    tool_options_str = build_tool_options_str(ALL_APIS, False)
    conv_history_str = build_conversation_str(ground_truth_history)
    metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
    enumerated_plans = list(enumerate(plan_options.keys()))
    plans_str = "\n".join([f"{i+1}. {plan}" for i, plan in enumerated_plans])
    
    prompt = user_proxy_prompt.format(
        tools=tool_options_str,
        metadata=metadata_str,
        conversation=conv_history_str,
        plans=plans_str,
    )
        
    if classification_type == "mistral":
        # Use Mistral to select best option based on plan
        selected_idx = mistral_generate(prompt)
    elif classification_type == "claude":
        # Use Claude to select best option based on plan
        selected_idx = claude_generate(prompt)
    elif classification_type == "gpt-4o":
        # Use GPT-4 to select best option based on plan
        selected_idx = gpt_generate(prompt, "gpt-4o", 0)
    elif classification_type == "command-r-plus-refresh":
        # Use single LLM to select best option based on plan
        selected_idx = cohere_generate(prompt, "command-r-plus-08-2024", 0)
    else:
        # Use single LLM to select best option based on plan
        selected_idx = cohere_generate(prompt, "command-r-plus", 0)
    
    selected_plan = [x for x in enumerated_plans if x[0] == int(selected_idx)-1][0][1]
    
    if len(plan_options[selected_plan]) > 1:
        print("Multiple options for selected plan, taking first")
    
    return plan_options[selected_plan][0]
    

def get_classified_conversation(
        conversations: List[Dict[str, str]],
        classification_type: str,
        extract_plans: bool = False,
        metadata_param: str = "reflection",
) -> Tuple[str, str]:
    conversation = conversations[0]
    turns = conversation["conversation"]
    ground_truth_history = [] 
       
    output_turns = []
    for turn_index, turn in enumerate(turns):
        if turn["role"] == "user":
            ground_truth_history.append({"role": "user", "text": turn["text"]})

        if turn["role"] == "assistant":
            turn_options = [c["conversation"][turn_index] for c in conversations]
            plan_options = {}

            # Get options "plans"
            for o in turn_options:
                if "predictions" in o:
                    if "metadata" in o["predictions"][0] and \
                        metadata_param in o["predictions"][0]["metadata"]:

                        if not o["predictions"][0]["metadata"][metadata_param]:
                            if "request" in o["predictions"][0]:
                                # API call made but with no reflection, take the action as the plan
                                plan = f'I will call {o["predictions"][0]["request"]["api_name"]} with the following parameters: {json.dumps(o["predictions"][0]["request"]["parameters"])}'

                            else:
                                # No API call made, take direct LLM response as plan
                                plan = f'I will respond to the user with the following: "{o["predictions"][0]["text"]}"'
                            
                        else:
                            # API call is made, take first reflection as plan
                            plan = o["predictions"][0]["metadata"][metadata_param]

                            if extract_plans:
                                plan = extract_plan(plan)

                    else:
                        # No API call made, take direct LLM response and convert to a "plan"
                        plan = f'I will respond to the user with the following: "{o["predictions"][0]["text"]}"'
                    
                    if plan in plan_options:
                        plan_options[plan].append(o)
                    else:
                        plan_options[plan] = [o]

                else:
                    raise ValueError("No predictions in turn options") # shouldn't happen, put this in to confirm
            
            if len(plan_options) == 1:

                # All plans the same, pick first option
                output_turns.append(plan_options[list(plan_options.keys())[0]][0])

                # Use LLM to select best option based on plan
                best_option = classify_plan_options(
                    plan_options, 
                    classification_type, 
                    ground_truth_history.copy(),
                    conversation["metadata"]
                )
                output_turns.append(best_option)

            if "apis" in turn:
                for api in turn["apis"]:
                    ground_truth_history.append(
                        {
                            "role": "api",
                            "request": api["request"],
                            "response": api["response"],
                            "exception": api["exception"],
                        }
                    )
            ground_truth_history.append({"role": "assistant", "text": turn["text"]})
            
        else:
            output_turns.append(turn)

    output_conversation = conversation
    output_conversation["conversation"] = output_turns

    return output_conversation


def get_llm_classifications(
        results_dirs: List[str],
        tool_executor: ToolExecutor,
        output_dir: str,
        classification_type: str,
        extract_plans: bool,
) -> Tuple[Dict[str, float], Dict[str, str], List[Tuple[str, str]]]:

    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists, checking for unprocessed files")
    else:
        os.makedirs(output_dir)
    
    print("results_dirs", results_dirs)

    if "claude" in results_dirs[0]:
        metadata_param = "content"
    else:
        metadata_param = "reflection"

    for file_name, file_path in get_names_and_paths(results_dirs[0]):
        if not file_name.endswith(".json"):
            continue

        outfile = output_dir + "/" + file_name

        if os.path.exists(outfile):
            print(f"File {outfile} already exists, skipping")
            continue
        
        print(f"Processing {file_name}")

        conversations = get_all_conversations(results_dirs, file_name)

        classified_conversation = get_classified_conversation(
            conversations, 
            classification_type, 
            extract_plans,
            metadata_param
        )

        classified_conversation = tool_executor.evaluate_predictions(classified_conversation)

        with open(outfile, "w", encoding="utf-8") as writer:
            json.dump(classified_conversation, writer, indent=4)
            

def main(flags: Optional[List[str]] = None):
    parser = get_arg_parser()
    args = parser.parse_args(flags)

    print("args", args)

    if args.results_dir is None:
        raise ValueError("Results directory is required")
        
    if args.model is None:
        raise ValueError("Model subdir is required")

    database_dir = str(TOOLTALK_EVAL_ROOT / "data" / "databases")

    tool_executor = ToolExecutor(init_database_dir=database_dir)

    results_dir = args.results_dir[0]
    model = args.model[0]        

    if args.class_type == "command-r-plus":
        output_suffix = "llm"
    elif args.class_type == "command-r-plus-refresh":
        output_suffix = "llm-refresh"
    elif args.class_type == "llama" or args.class_type == "claude" or \
          args.class_type == "gpt4" or args.class_type == "mistral":
        output_suffix = f"llm-{args.class_type}"
    else:
        output_suffix = args.class_type

    model_options = os.listdir(results_dir)


    model_dirs = [results_dir + "/" + x for x in model_options if (model in x and len(x.replace(model, "")) <= 1)]
    output_dir = results_dir + f"/{model}-{output_suffix}_selected_top_{len(model_dirs)}"
    
    get_llm_classifications(model_dirs, tool_executor, output_dir, args.class_type, args.extract_plans)


if __name__ == "__main__":
    main()