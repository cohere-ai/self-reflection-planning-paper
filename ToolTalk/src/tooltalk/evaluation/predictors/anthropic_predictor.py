import json
import logging
from typing import Any, Dict, List, Tuple
from deep_planner.deep_planner import DeepPlanner
from anthropic import Anthropic, AnthropicBedrock

from tooltalk.apis.api import API
from tooltalk.evaluation.predictors.base import BaseAPIPredictor
from tooltalk.evaluation.predictors.utils import strip_session_token
from tooltalk.utils.anthropic_tools.tool_user import ToolUser
from tooltalk.utils.anthropic_tools.tools.base_tool import BaseTool
from tooltalk.utils.anthropic_tools.messages_api_converters import (
    convert_completion_to_messages,
    convert_messages_completion_object_to_completions_completion_object,
)
from tooltalk.utils.anthropic_tools.prompt_constructors import construct_use_tools_prompt 
logger = logging.getLogger(__name__)


class DummyTool(BaseTool):
    """Mockup tool only for rendering purposes, as the execution logic is offloaded to ToolTalk."""

    def use_tool(self, *args, **kwargs):
        pass


TYPES_MAP = {
    "string": "str",
    "number": "float",
    "boolean": "bool",
    "integer": "int",
    "int": "int",
    "array": "list",
    "object": "dict",
}


def to_anthropic_tool_parameter(
    tt_param: Tuple[str, dict], disable_docs: bool = False
) -> Dict[str, str]:
    name, attributes = tt_param
    description = "" if disable_docs else attributes["description"]
    if attributes["type"] in ("object", "dict"):
        type = "dict"
    elif attributes["type"] == "array":
        type = "List[" + TYPES_MAP[attributes["items"]["type"]] + "]"
    else:
        type = TYPES_MAP[attributes["type"]]
    return {
        "name": name,
        "description": description,
        "type": type,
        "required": attributes["required"],
    }


def to_anthropic_tool(api: API, disable_docs: bool) -> DummyTool:
    """Converts a ToolTalk api into an Anthropic Tool."""
    anthropic_tool = DummyTool(
        name=api.to_dict()["name"],
        description="" if disable_docs else api.description,
        parameters=[
            to_anthropic_tool_parameter(param, disable_docs)
            for param in api.parameters.items()
        ],
    )
    return anthropic_tool


ENDPOINTS = {
    "anthropic": {
        "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-2.1": "claude-2.1",
        "claude-2.0": "claude-2.0",
        "claude-instant-1.2": "claude-instant-1.2",
    },
    "bedrock": {
        "claude-2.0": "anthropic.claude-v2",
        "claude-2.1": "anthropic.claude-v2:1",
        "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3.5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "claude-instant-1": "anthropic.claude-instant-v1",
        "titan-express": "amazon.titan-text-express-v1",
        "titan-lite": "amazon.titan-text-lite-v1",
    },
}


class AnthropicPredictor(BaseAPIPredictor):
    """
    See https://github.com/anthropics/anthropic-tools
    """

    system_prompt = (
        "\nYou are a helpful assistant. Here is some user data:"
        "\nlocation: {location}"
        "\ntimestamp: {timestamp}"
        "\nusername (if logged in): {username}"
    )

    planning_prompt = (
        "\n\nBefore making use of any tools, make a plan that can solve the problem step by step.", 
        "For each step, indicate which of the available tools will be used together with desired parameter values.",
        "You should store the response from each step in a variable #V that can be used to by tools in subsequent steps.",
        "Each step should have exactly one variable that stores the results of one tool and has a name that increments with the step number (e.g: Step 1, #V1, Step 2, #V2, Step 3, ...)."
    )

    planning_example = (
        "\nBelow is an example of the required plan format:"
        "\nStep 1: Translate the problem into algebraic expressions.",
        "\n#V1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]",
        "\nStep 2: Find out the number of hours Thomas worked.",
        "\n#V2 = Search[What is x, given #V1]",
        "\nStep 3: Calculate the number of hours Rebecca worked.",
        "\n#V3 = Calculator[(2 ∗ #V2 − 10) − 8]"
    )

    def __init__(
        self,
        model,
        apis_used: List[API],
        disable_docs=False,
        route="bedrock",
        force_single_hop=False,
        temperature=None,
        debug=False,
        is_detailed_plans=True,
        is_sea=False,
    ):
        self.tools = [to_anthropic_tool(api, disable_docs) for api in apis_used]
        if debug:
            print(f"Anthropic tools: {json.dumps(self.tools, indent=2, default=vars)}")
        self.force_single_hop = force_single_hop
        self.temperature = temperature or 0
        self.debug = debug
        self.tool_calls_backlog = []
        self.tool_user = ToolUser(
            self.tools,
            temperature=self.temperature,
            first_party=route == "anthropic",
            model=ENDPOINTS[route][model],
        )
        self.is_detailed_plans = is_detailed_plans
        self.is_sea = is_sea

        if is_sea:
            self.tools_string = "\n\n".join([api.to_docstring() for api in apis_used])
            self.deep_planner = DeepPlanner(
                n_plans=3, 
                temperature=1, 
                plan_generator_func=self.deep_planner_generate_plan, 
                eval_generate_func=self.deep_planner_generate,
                compile_generate_func=self.deep_planner_generate,
                extract_plan_func=self.extract_plan_from_completion,
            )


    def make_tool_call_turn(self, tool_call: dict, metadata: dict = {}) -> dict:
        return {
            "role": "api",
            "request": {
                "api_name": tool_call["tool_name"],
                "parameters": tool_call["tool_arguments"],
            },
            "metadata": metadata,
        }


    def extract_plan_from_completion(self, completion: str) -> str:

        if "#V" in completion:
            lines = completion.split("\n")
            while lines and not lines[-1].startswith("#V"):
                lines.pop()
            completion = "\n".join(lines)

        if "Step" in completion:
            lines = completion.split("\n")
            while lines and not lines[0].startswith("Step"):
                lines.pop(0)
            completion = "\n".join(lines)
            return completion
        
        return f'Respond to user with: "{completion}"'

    def deep_planner_generate_plan(self, messages, temperature):
        prompt = ToolUser._construct_prompt_from_messages(messages)
        prompt = construct_use_tools_prompt(prompt, self.tools, messages[-1]["role"])
        messages = convert_completion_to_messages(prompt)
        if "system" not in messages:
            completion = self.tool_user.client.messages.create(
                model=self.tool_user.model,
                max_tokens=2000,
                temperature=temperature,
                stop_sequences=["</function_calls>", "\n\nHuman:"],
                messages=messages["messages"],
            )
        else:
            assert isinstance(messages["system"], str)
            completion = self.tool_user.client.messages.create(
                model=self.tool_user.model,
                max_tokens=2000,
                temperature=temperature,
                stop_sequences=["</function_calls>", "\n\nHuman:"],
                messages=messages["messages"],
                system=messages["system"],
            )
        return convert_messages_completion_object_to_completions_completion_object(completion).completion
    
    def deep_planner_generate(self, prompt, temperature):
        completion = self.tool_user.client.messages.create(
            model=self.tool_user.model,
            max_tokens=2000,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return completion.content[0].text

    def create_messages(self, metadata: dict, conversation_history: List[dict]) -> List[Dict[str, Any]]:
        system_prompt = self.system_prompt.format(
            location=metadata["location"],
            timestamp=metadata["timestamp"],
            username=metadata.get("username"),
        )
        if metadata.get("task_and_context"):
            system_prompt += f"\nTASK AND CONTEXT\n{metadata['task_and_context']}"

        if self.is_detailed_plans:
            system_prompt = system_prompt + " ".join(self.planning_prompt) + "".join(self.planning_example)
            system_prompt = system_prompt + "\n\nNote that you only need to provide a plan if you are using tools (i.e., not if you asking a question of the user)."

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for turn in conversation_history:
            if turn["role"] == "user" or turn["role"] == "assistant":
                messages.append({"role": turn["role"], "content": turn["text"]})
            elif turn["role"] == "api":
                turn = strip_session_token(turn)
                tool_name = turn["request"]["api_name"]
                messages.append(
                    {
                        "role": "tool_inputs",
                        "content": "",
                        "tool_inputs": [
                            {
                                "tool_name": tool_name,
                                "tool_arguments": turn["request"]["parameters"],
                            }
                        ],
                    }
                )
                tool_outputs_msg = {
                    "role": "tool_outputs",
                    "tool_outputs": None,
                    "tool_error": None,
                }
                if turn["response"] is None:
                    assert isinstance(turn["exception"], str)
                    tool_outputs_msg["tool_error"] = turn["exception"]
                else:
                    tool_outputs_msg["tool_outputs"] = [
                        {"tool_name": tool_name, "tool_result": turn["response"]}
                    ]
                messages.append(tool_outputs_msg)
        return messages


    def predict(self, metadata: dict, conversation_history: dict) -> dict:
        turn = self.maybe_make_quick_turn(conversation_history[-1])
        if turn is not None:
            return turn

        messages = self.create_messages(metadata, conversation_history)
        
        if self.debug:
            print(
                "Anthropic chat messages:\n"
                + "\n".join(json.dumps(m, indent=2) for m in messages)
            )

        tool_plan = None

        if self.is_detailed_plans and conversation_history[-1]["role"] == "user":
            
            if self.is_sea:
                planner_conversation_history = [{"role": "metadata", "text": json.dumps(metadata)}] + conversation_history
                conversation_history_str = "\n".join([turn["role"] + ": " + turn["text"] if turn["role"] != "api" else turn["role"] + ": " + \
                                                    json.dumps(turn["request"]) + "\nResponse: " + json.dumps(turn["response"]) for turn in planner_conversation_history])

                tool_plan = self.deep_planner.plan(
                    planning_messages=messages,
                    previous_conversation_str=conversation_history_str,
                    tools_string = self.tools_string,
                )

        response = self.tool_user.use_tools(
            messages, 
            verbose=0.5 if self.debug else 0, 
            temperature=self.temperature, 
            tool_plan=tool_plan,
        )  # type: ignore

        
        metadata_content = tool_plan if tool_plan else response["content"]

        assert isinstance(response, dict)
        if self.debug:
            print(f"Anthropic full response: {response}")

        if response.get("status") == "ERROR":
            return {
                "role": "assistant",
                "text": response["error_message"],
                "metadata": {"source": "error"},
            }
        elif response["role"] == "tool_inputs":
            tool_calls = response["tool_inputs"]
            if len(tool_calls) == 0:
                return {
                    "role": "assistant",
                    "text": metadata_content,
                    "metadata": {"source": "no_tool_calls"},
                }
            self.tool_calls_backlog = tool_calls[1:]
            metadata = {
                "content": metadata_content,
                "tool_calls_backlog": self.tool_calls_backlog,
                "source": "hop_on",
            }
            return self.make_tool_call_turn(tool_calls[0], metadata)
        else:
            assert response["role"] == "assistant"
            return {
                "role": "assistant",
                "text": metadata_content,
                "metadata": {"source": "hop_done"},
            }
