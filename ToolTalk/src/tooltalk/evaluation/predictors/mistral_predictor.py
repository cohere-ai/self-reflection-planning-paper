import json
import logging
from typing import List

from deep_planner.deep_planner import DeepPlanner

from mistralai.models.chat_completion import (
    ChatMessage,
    Function,
    FunctionCall,
    ToolCall,
)
from tooltalk.apis.api import API
from tooltalk.evaluation.predictors.base import BaseAPIPredictor
from tooltalk.evaluation.predictors.utils import strip_session_token
from tooltalk.utils.mistral_utils import chat_with_backoff

logger = logging.getLogger(__name__)


class MistralPredictor(BaseAPIPredictor):
    """
    See https://docs.mistral.ai/api/
    """

    system_prompt = (
        "You are a helpful assistant. Here is some user data:"
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
        force_single_hop=False,
        temperature=None,
        debug=False,
        is_detailed_plans=True,
        is_sea=False,
    ):
        self.model = model
        self.tools = []
        for api in apis_used:
            api_doc = api.to_openai_doc(disable_docs)
            # relocate "required"
            api_doc["parameters"]["required"] = api_doc.pop("required")
            self.tools.append({"type": "function", "function": Function(**api_doc)})
        if debug:
            print(f"Mistral tools: {json.dumps(self.tools, indent=2)}")
        self.force_single_hop = force_single_hop
        self.temperature = temperature
        self.debug = debug
        self.tool_calls_backlog = []
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
                is_printing=True
            )

    def extract_plan_from_completion(self, completion: str) -> str:
        
        if "#V" in completion:
            lines = completion.split("\n")
            while lines and not lines[-1].startswith("#V"):
                lines.pop()
            
            if "".join(lines).strip() == "":
                return completion

            completion = "\n".join(lines)

        if "Step" in completion:
            lines = completion.split("\n")
            while lines and not "Step" in lines[0]:
                lines.pop(0)
            completion = "\n".join(lines)
            return completion
        
        return f'Respond to user with: "{completion}"'

    def make_tool_call_turn(self, tool_call: ToolCall, metadata: dict = {}) -> dict:
        api_name = tool_call.function.name
        try:
            parameters = json.loads(tool_call.function.arguments)
        except json.decoder.JSONDecodeError:
            # check termination reason
            logger.info(
                f"Failed to decode arguments for {api_name}: {tool_call.function.arguments}"
            )
            parameters = None
        return {
            "role": "api",
            "request": {"api_name": api_name, "parameters": parameters},
            "metadata": metadata,
        }

    def deep_planner_generate(self, prompt: str, temperature: float) -> str:
        return chat_with_backoff(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        ).choices[0].message.content

    
    def deep_planner_generate_plan(self, messages: List[dict], temperature: float) -> str:
        return chat_with_backoff(
            model=self.model,
            messages=messages,
            temperature=temperature,
            tools=self.tools,
        ).choices[0].message.content


    def create_messages(self, metadata: dict, conversation_history: List[dict]) -> List[ChatMessage]:
        system_prompt = self.system_prompt.format(
            location=metadata["location"],
            timestamp=metadata["timestamp"],
            username=metadata.get("username"),
        )

        if self.is_detailed_plans and conversation_history[-1]["role"] == "user":
            system_prompt = system_prompt + " ".join(self.planning_prompt) + "".join(self.planning_example)
            system_prompt = system_prompt + "\n\nNote that you only need to provide a plan if you are using tools (i.e., not if you asking a question of the user). Once the user confirms the plan, you can proceed with the execution."

        if metadata.get("task_and_context"):
            system_prompt += f"\nTASK AND CONTEXT\n{metadata['task_and_context']}"

        messages = [ChatMessage(role="system", content=system_prompt)]
        for i, turn in enumerate(conversation_history):
            if turn["role"] == "user" or turn["role"] == "assistant":
                messages.append(ChatMessage(role=turn["role"], content=turn["text"]))
            elif turn["role"] == "api":
                turn = strip_session_token(turn)
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content="",
                        tool_calls=[
                            ToolCall(
                                id=(str(i) * 10)[
                                    :9
                                ],  # "Tool call id was null but must be a-z, A-Z, 0-9, with a length of 9."
                                function=FunctionCall(
                                    name=turn["request"]["api_name"],
                                    arguments=json.dumps(turn["request"]["parameters"]),
                                ),
                            )
                        ],
                    )
                )
                response_content = {
                    "response": turn["response"],
                    "exception": turn["exception"],
                }
                messages.append(
                    ChatMessage(
                        role="tool",
                        name=turn["request"]["api_name"],
                        content=json.dumps(response_content),
                    )
                )
        return messages

    def predict(self, metadata: dict, conversation_history: dict) -> dict:

        turn = self.maybe_make_quick_turn(conversation_history[-1])
        if turn is not None:
            return turn

        messages = self.create_messages(metadata, conversation_history)

        # TODO: replace print with logger.debug (which isn't working for some reason ...)
        if self.debug:
            print(
                "Mistral chat messages:\n"
                + "\n".join(m.model_dump_json(indent=2) for m in messages)
            )


        if self.is_detailed_plans and conversation_history[-1]["role"] == "user":
                    

            if self.is_sea:
                planner_conversation_history = [{"role": "metadata", "text": json.dumps(metadata)}] + conversation_history
                conversation_history_str = "\n".join([turn["role"] + ": " + turn["text"] if turn["role"] != "api" else turn["role"] + ": " + \
                                                    json.dumps(turn["request"]) + "\nResponse: " + json.dumps(turn["response"]) for turn in planner_conversation_history])

                plan = self.deep_planner.plan(
                    planning_messages=messages, 
                    previous_conversation_str=conversation_history_str,
                    tools_string=self.tools_string,
                )
                reflection_text = plan

            else:

                response = chat_with_backoff(
                    model=self.model,
                    messages=messages,
                    # tools=self.tools,
                    temperature=self.temperature,
                    # tool_choice="none",
                )
                reflection_text = response.choices[0].message.content
            
            messages.append({ "role": "assistant", "content": "Here's a plan on how to solve your request:\n" + reflection_text })
            messages.append({"role": "user", "content": "Looks good, let's execute the plan."})
            response = chat_with_backoff(
                model=self.model,
                messages=messages,
                tools=self.tools,
                temperature=self.temperature,
            )

        else:
            response = chat_with_backoff(
                model=self.model,
                messages=messages,
                tools=self.tools,
                temperature=self.temperature,
            )
            reflection_text = response.choices[0].message.content

        if self.debug:
            print(f"Mistral full response: {response}")

        message = response.choices[0].message
        metadata = {
            "reflection": reflection_text,
            "response": response.model_dump_json(indent=2)
        }
        if message.tool_calls is not None:
            tool_call = message.tool_calls[0]
            self.tool_calls_backlog = message.tool_calls[1:]
            return self.make_tool_call_turn(tool_call, metadata)
        else:
            return {
                "role": "assistant",
                "text": message.content,
                "metadata": metadata,
            }
