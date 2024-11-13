import copy
import json
import logging
import re
from typing import Any, List, Optional, Tuple, Dict, Callable

from cohere import Client
from cohere import InternalServerError as CohereISError
from cohere.core import ApiError
from tooltalk.apis.api import API
from tooltalk.evaluation.predictors.base import BaseAPIPredictor
from langchain_cohere.react_multi_hop.prompt import (
    render_structured_preamble,
    create_directly_answer_tool,
    render_tool
)
from langchain_cohere.react_multi_hop.default_prompt_constants import (
    _SpecialToken,
    default_multi_hop_instruction,
)
from langchain_cohere.react_multi_hop.parsing import (
    # parse_actions,
    parse_answer_with_prefixes,
    parse_jsonified_tool_use_generation
)

from deep_planner.deep_planner import DeepPlanner

logger = logging.getLogger(__name__)


DIRECTLY_ANSWER_TOOL = create_directly_answer_tool()
DIRECTLY_ANSWER_TOOL = {
    "name": DIRECTLY_ANSWER_TOOL.name,
    "description": DIRECTLY_ANSWER_TOOL.description,
}


def parse_actions(completion: str) -> Tuple[str, Optional[str], List[dict]]:

        completion = completion.strip()
        actions = ""
        try:
            if ("Plan: " in completion or "Reflection: " in completion) and "Action: ```json\n" not in completion:
                # Model is trained to output a Plan or Reflection followed by an action.
                # Use regex to extract the plan or reflection only.
                regex = r"^(Plan|Reflection)\s*\d*\s*:(.*)"
                action_match = re.search(regex, completion, re.DOTALL)
                assert action_match is not None
                plan = action_match.group(2).strip()
                actions = ""
            elif "Plan: " in completion or "Reflection: " in completion:
                # Model is trained to output a Plan or Reflection followed by an action.
                # Use regex to extract the plan and action.
                regex = r"^(Plan|Reflection)\s*\d*\s*:(.*?)(Action\s*\d*\s*:\s*\d*\s*```json\n.*)"
                action_match = re.search(regex, completion, re.DOTALL)
                assert action_match is not None
                plan = action_match.group(2).strip()
                actions = action_match.group(3).strip()
            else:
                # Catch the case where model outputs only an action.
                regex = r"^(Action\s*\d*\s*:\s*\d*\s*```json\n.*)"
                action_match = re.search(regex, completion, re.DOTALL)
                plan = ""
                assert action_match is not None
                actions = action_match.group(1).strip()
        except Exception as e:
            logging.error(f"Failed to parse multihop completion for input: {completion}")
            logging.error(f"Error: {e}")

        parsed_actions = []
        if actions != "":
            # try to parse incomplete json
            potential_endings = ["", "```", "]```", "}]```", "}}]```", '"}}]```']
            for ending in potential_endings:
                try:
                    parsed_actions = parse_jsonified_tool_use_generation(
                        actions + ending, "Action:"
                    )
                    break
                except:
                    pass

        return completion, plan, parsed_actions


def render_role(message: dict) -> str:
    """Renders the role of a message into prompt content."""
    if "is_system" in message and message["is_system"]:
        return _SpecialToken.role_system.value
    elif "is_bot" in message and message["is_bot"]:
        return _SpecialToken.role_chatbot.value
    else:
        return _SpecialToken.role_user.value

def render_messages(messages: List[dict]) -> str:
    """Renders one or more BaseMessage implementations into prompt content."""
    return "".join(
        [
            f"{_SpecialToken.start_turn.value}{render_role(message)}{message['message']}{_SpecialToken.end_turn.value}"
            for message in messages
        ]
    )

def discard_unmatched_tool_use_calls(tool_use_calls: List[dict], tool_specs: List[dict]) -> List[dict]:
    result = []
    # tool_names = [tool_spec["info"]["title"] for tool_spec in tool_specs]
    tool_names = [tool_spec["name"] for tool_spec in tool_specs]

    for action in tool_use_calls:
        if action["tool_name"] in tool_names:
            result.append(action)
        else:
            logger.warning(f"Discarding unmatched action: {action}")
    return result


def push_directly_answer_to_the_end(tool_use_calls: List[dict]) -> List[dict]:
    directly_answer_action = None
    result = []
    for action in tool_use_calls:
        if action["tool_name"] == DIRECTLY_ANSWER_TOOL["name"]:
            if directly_answer_action is not None:
                logger.warning("Multiple directly-answer actions found, keeping only the first one")
            else:
                directly_answer_action = action
        else:
            result.append(action)
    if directly_answer_action is not None:
        result.append(directly_answer_action)
    return result


def make_failed_turn(error_message: str, source: str, reflection: Optional[str] = None) -> dict:
    turn = {"role": "assistant", "text": f"Sorry! There was an error: {error_message}", "metadata": {"source": source}}
    if reflection:
        turn["metadata"]["reflection"] = reflection
    return turn


def extract_plan_from_completion(completion: str) -> str:
    if "Action:" in completion:
        if "Step 1:" in completion:
            return "Step 1:" + completion.split("Step 1:")[-1].split("Action:")[0].strip()

        if "Plan:" in completion:
            return completion.split("Plan:")[1].split("Action:")[0].strip()
        else:
            return completion.split("Action:")[0].strip()

    else:
        if "Step 1:" in completion:
            return "Step 1:" + completion.split("Step 1:")[-1]
        return f'Respond to user with: "{completion}"'

class CoherePredictor(BaseAPIPredictor):
    mh_system_prompt = """{structured_preamble}

## Available Tools
Here is a list of tools that you have available to you:

{tools}{end_turn}{history}{user_prompt}{start_turn}{system_role}{multi_hop_instruction}{end_turn}"""

    tooltalk_system_prompt = (
        "Here is some user data (timestamp is in the local time zone):"
        "\nlocation: {location}"
        "\ntimestamp: {timestamp}"
        "\nusername (if logged in): {username}"
    )

    orig_planning_string = "Write 'Plan:' followed by an initial high level plan of how you will solve the problem including the tools and steps required."

    new_planning_string = (
        "Write 'Plan:' followed a plan that can solve the problem step by step. For each step, indicate which of the available tools will be used together with desired parameter values. You should store the response from each step in a variable #V that can be used to by tools in subsequent steps. Each step should have exactly one variable that stores the results of one tool and has a name that increments with the step number (e.g: Step 1, #V1, Step 2, #V2, Step 3, ...). Below is an example of the required plan format:",
        "<plan_format>",
        "Step 1: Translate the problem into algebraic expressions.",
        "#V1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]",
        "Step 2: Find out the number of hours Thomas worked.",
        "#V2 = Search[What is x, given #V1]",
        "Step 3: Calculate the number of hours Rebecca worked.",
        "#V3 = Calculator[(2 ∗ #V2 − 10) − 8]",
        "</plan_format>",
        "It is of paramount importance that you generate a plan in this format for every tool use, even if it is a single tool, and that you pay particular attention to the parameter values specified in the plan (ensuring they match the format required by the tool).",
    )


    def __init__(
        self,
        model: str,
        apis_used: List[API],
        disable_docs: bool = False,
        force_single_hop=False,
        skip_augmented_gen: bool = False,
        temperature: Optional[float] = None,
        debug: bool = False,
        is_detailed_plans=True,
        is_sea=False,
    ):
        self.tool_specs = {api.__name__: api.to_openai_doc(disable_doc=disable_docs) for api in apis_used}
        self.tool_specs[DIRECTLY_ANSWER_TOOL["name"]] = { 
            "name": DIRECTLY_ANSWER_TOOL["name"], 
            "description": DIRECTLY_ANSWER_TOOL["description"],
        }

        self.multihop_call_preamble = render_structured_preamble()
        self.model = model

        # other helper fields
        self.action_prefix = "Action:"
        self.action_results_prefix = "<results>\n"
        self.action_results_suffix = "\n</results>"
        self.reflection_prefix = "Reflection:"
        self.plan_prefix = "Plan:"
        self.answer_prefix = "Answer:"
        self.response_prefix = "Response:"
        self.exception_prefix = "Exception:"
        self.exception_errmsg_key = "error_message"

        self.action_backlog: List[dict] = []
        self.temperature = temperature
        self.debug = debug

        self.client = Client()
        self.no_markup_answer_only = True
        self.prompt = None
        self.completion = None
        self.is_detailed_plans = is_detailed_plans
        self.is_sea = is_sea

        if is_sea:
            self.tools_string = "\n\n".join([api.to_docstring() for api in apis_used])
            self.deep_planner = DeepPlanner(
                n_plans=3, 
                temperature=1, 
                plan_generator_func=self.generate, 
                eval_generate_func=self.generate_standard,
                compile_generate_func=self.generate_standard,
                extract_plan_func=extract_plan_from_completion,
                is_printing=True
            )

    def _remake_action_message(self, request: dict) -> str:
        if "session_token" in request["parameters"]:
            request = copy.deepcopy(request)
            del request["parameters"]["session_token"]
        action_call = {
            "tool_name": request["api_name"],
            "parameters": request["parameters"],
        }
        actions_text = "```json\n" + json.dumps([action_call], indent=4) + "\n```"
        return self.action_prefix + " " + actions_text

    def _remake_result_message(self, turn: dict, doc_id: Optional[int] = None) -> str:
        if turn["exception"] is None:
            result = self.response_prefix + " " + json.dumps(turn["response"])
        else:
            # tooltalk exception is plain text, but we want to make it more generic
            result = self.exception_prefix + " " + json.dumps({self.exception_errmsg_key: turn["exception"]})
        if doc_id is not None:
            result = f"Document: {doc_id}\n" + result
        return self.action_results_prefix + result + self.action_results_suffix

    def remake_turn(self, tt_turn: dict) -> List[dict]:
        if tt_turn["role"] == "user":
            return [{"is_bot": False, "message": tt_turn["text"]}]
        elif tt_turn["role"] == "assistant":
            return [{"is_bot": True, "message": tt_turn["text"]}]
        else:
            if tt_turn["role"] != "api":
                raise ValueError(f"Unexpected role: {tt_turn['role']}")
            return [
                {
                    "is_bot": True,
                    "message": self._remake_action_message(tt_turn["request"]),
                },
                {
                    "is_system": True,
                    "message": self._remake_result_message(tt_turn),
                },
            ]

    def remake_history(self, tt_history: List[dict], metadata: dict) -> Tuple[str, List[dict], List[dict]]:
        """
        tt_history: if from GT, must ends with user turn; else must end with mid multi-hop predictions

        return type:
        question: str (the very last user input)
        history: List[dict]
        steps: List[dict]
        """
        tt_system_prompt = self.tooltalk_system_prompt.format(
            location=metadata["location"], timestamp=metadata["timestamp"], username=metadata.get("username")
        )
        history = [{"is_system": True, "message": tt_system_prompt}]
        steps = []
        for tt_turn in tt_history:
            history.extend(self.remake_turn(tt_turn))
            if tt_turn["role"] == "user":
                steps = []
            elif tt_turn["role"] == "api":
                steps.append(tt_turn)
        if steps:
            # remove multi-hop steps from history
            history = history[: -len(steps) * 2]
        assert not (history[-1].get("is_bot", False) or history[-1].get("is_system", False))
        return history[-1]["message"], history[:-1], steps

    def generate(self, prompt: str, temperature: float, max_retries: float =3) -> str:
        kwargs = {}
        if temperature == 0:
            # anecdote suggests this is useful for greedy decoding
            kwargs["k"] = 1
        n_retries = 0
        while n_retries < max_retries:
            try:
                response = self.client.chat(
                    message=prompt,
                    model=self.model,
                    raw_prompting=True,
                    temperature=temperature,
                    **kwargs,
                )
            except ApiError as ex:
                if ex.status_code == 504:  # TODO: replace with GatewayTimeoutError when available
                    logger.warning(f"Gateway timeout: {ex}")
                    n_retries += 1
                else:
                    raise
            else:
                break
        else:
            raise ApiError(status_code=504, body="Gateway timeout")
        return response.text

    def generate_standard(self, prompt: str, temperature: float, max_retries: int = 5) -> str:
        kwargs = {}
        if temperature == 0:
            # anecdote suggests this is useful for greedy decoding
            kwargs["k"] = 1
        n_retries = 0
        while n_retries < max_retries:
            try:
                response = self.client.chat(
                    message=prompt,
                    model=self.model,
                    temperature=temperature,
                    **kwargs,
                )
            except ApiError as ex:
                if ex.status_code == 504:  # TODO: replace with GatewayTimeoutError when available
                    logger.warning(f"Gateway timeout: {ex}")
                    n_retries += 1
                else:
                    raise
            except Exception as ex:
                print("Error: ", ex)
                n_retries += 1
            else:
                break
        else:
            raise ApiError(status_code=504, body="Gateway timeout")
        return response.text

    def _append_step_to_prompt(self, prompt: str, step: dict, step_idx: int) -> str:
        assert prompt.endswith(_SpecialToken.start_turn.value + _SpecialToken.role_chatbot.value)
        assert step["role"] == "api"
        prompt += "\n"
        reflection: Optional[str] = step.get("metadata", {}).get("reflection")
        if reflection:
            prefix = self.plan_prefix if step_idx == 0 else self.reflection_prefix
            prompt += prefix + " " + reflection + "\n"
        prompt += self._remake_action_message(step["request"]) + _SpecialToken.end_turn.value
        prompt += (
            "\n"
            + _SpecialToken.start_turn.value 
            + _SpecialToken.role_system.value
            + self._remake_result_message(step, step_idx)
            + _SpecialToken.end_turn.value 
            + _SpecialToken.start_turn.value 
            + _SpecialToken.role_chatbot.value
        )
        return prompt

    def make_prompt(self, question: str, history: List[dict], steps: List[dict]) -> str:
        
        prompt = self.mh_system_prompt.format(
            structured_preamble=self.multihop_call_preamble,
            tools="\n\n".join([render_tool(json_schema = t) for t in list(self.tool_specs.values())]),
            end_turn=_SpecialToken.end_turn.value,
            start_turn=_SpecialToken.start_turn.value,
            system_role=_SpecialToken.role_system.value,
            multi_hop_instruction=default_multi_hop_instruction,
            # steps=None,
            history=render_messages(history),
            user_prompt=question,
        )

        prompt += (
            f"{_SpecialToken.start_turn.value}{_SpecialToken.role_chatbot.value}"
        )

        prompt = re.sub(r" The current date is .*\n", "\n", prompt, flags=re.IGNORECASE)
        assert (
            "current date" not in prompt.lower() and "current time" not in prompt.lower()
        ), "Found date/time in the prompt, which will likely confuse the model if it conflicts with system prompt"
        logging.debug(f"Prompt (pre-steps): {prompt}")
        for i, step in enumerate(steps):
            prompt = self._append_step_to_prompt(prompt, step, i)
        self.prompt = prompt
        return prompt

    def predict(self, metadata: dict, conversation_history: List[dict]) -> dict:
        if self.action_backlog:
            if conversation_history[-1]["role"] != "api":
                self.action_backlog = []
            elif conversation_history[-1]["exception"] is None:
                action = self.action_backlog.pop(0)
                next_turn = self.make_next_turn(action)
                logger.debug(f"Next turn (from action_backlog): {next_turn}")
                return next_turn
            else:
                logger.info(
                    f"Skipping {len(self.action_backlog)} action(s) in backlog "
                    f"because previous action failed: {conversation_history[-1]['exception']}"
                )
                self.action_backlog = []

        question, history, steps = self.remake_history(conversation_history, metadata)
        logger.debug(f"History: {history}")
        logger.debug(f"Question: {question}")
        logger.debug(f"Steps: {steps}")
        if self.debug:
            breakpoint()

        prompt = self.make_prompt(question, history, steps)

        if self.is_detailed_plans:
            assert self.orig_planning_string in prompt
            prompt = prompt.replace(self.orig_planning_string, "\n".join(self.new_planning_string))

        logger.debug(f"Prompt: {prompt}")
        if self.debug:
            breakpoint()

        try:
            
            if self.is_sea and conversation_history[-1]["role"] == "user":

                planner_conversation_history = [{"role": "metadata", "text": json.dumps(metadata)}] + conversation_history
                conversation_history_str = "\n".join([turn["role"] + ": " + turn["text"] if turn["role"] != "api" else turn["role"] + ": " + \
                                            json.dumps(turn["request"]) + "\nResponse: " + json.dumps(turn["response"]) for turn in planner_conversation_history])
                
                plan = self.deep_planner.plan(
                    planning_prompt=prompt, 
                    previous_conversation_str=conversation_history_str, 
                    tools_string=self.tools_string
                )

                plan_completion = self.plan_prefix + " \n" + plan + "\n" + self.action_prefix
                completion = self.generate(prompt + plan_completion, self.temperature)
                completion = plan_completion + completion

            else:
                completion = self.generate(prompt, self.temperature)
        except CohereISError as ex:
            logger.error("Cohere internal server error: %s", ex)
            return make_failed_turn(str(ex), "multihop_call")


        self.completion = completion
        logger.info(f"Completion: {completion}")
        if self.debug:
            breakpoint()

        reflection: Optional[str] = None
        error_msg = None
        tool_use_calls: List[dict] = []
        answer = None
        if self.action_prefix in completion:
            try:
                _, reflection, tool_use_calls = parse_actions(completion) #self.multihop_call.parse_completion(completion)  # type: ignore
                logger.debug(f"Tool use calls: {tool_use_calls}")
                if len(tool_use_calls) == 0:
                    return self.make_next_turn(
                        {"tool_name": DIRECTLY_ANSWER_TOOL["name"], "parameters": "Error: model makes no tool calls"},
                        reflection,
                    )
            except ValueError:
                error_msg = "Failed to parse multihop_call completion"
                logger.exception(error_msg)
                tool_use_calls = []
        if self.answer_prefix in completion:
            try:
                prefix_map = {
                    "answer": "Answer:",
                    "grounded_answer": "Grounded answer:",
                    "relevant_docs": "Relevant Documents:",
                    "cited_docs": "Cited Documents:",
                }
                answer = parse_answer_with_prefixes(completion, prefix_map) #self.augmented_generation_call.parse_completion(completion)
                logger.debug(f"Answer: {answer}")
                if self.no_markup_answer_only:
                    answer = answer["answer"] #answer["no_markup_answer"]
            except:
                answer = "Error: failed to parse augmented_generation_call completion"
        if tool_use_calls:
            tool_use_calls = discard_unmatched_tool_use_calls(tool_use_calls, list(self.tool_specs.values()))
            logger.debug(f"Tool use calls (post-discard): {tool_use_calls}")
            if len(tool_use_calls) == 0:
                error_msg = "No valid tool-use call found"
        if len(tool_use_calls) > 1:
            tool_use_calls = push_directly_answer_to_the_end(tool_use_calls)
            self.action_backlog = tool_use_calls[1:]
            logger.info(f"{len(tool_use_calls)} actions generated in one go, executing one by one ...")
        if tool_use_calls:
            action = tool_use_calls[0]

            if action["tool_name"] == DIRECTLY_ANSWER_TOOL["name"] and answer is not None:
                action["parameters"] = answer
            next_turn = self.make_next_turn(action, reflection)
        elif answer is not None:
            next_turn = self.make_next_turn({"tool_name": DIRECTLY_ANSWER_TOOL["name"], "parameters": answer}, reflection)
        else:
            if error_msg is None:
                assert self.answer_prefix not in completion and self.action_prefix not in completion
                error_msg = "Neither tool-use calls nor answer was generated"
            next_turn = make_failed_turn(error_msg, "multihop_call", reflection)
        logger.debug(f"Next turn: {next_turn}")
        if self.debug:
            breakpoint()
        return next_turn

    def make_next_turn(
        self,
        action: dict,
        reflection: Optional[str] = None,
    ) -> dict:
        logging.debug(f"Making next action: {action}")
        metadata = {"reflection": reflection, "source": "multihop_call"}
        if action["tool_name"] == DIRECTLY_ANSWER_TOOL["name"]:
            assert len(self.action_backlog) == 0
            params = action.get("parameters")
            if isinstance(params, dict) and params != {}:
                text = json.dumps(params)
            elif isinstance(params, str) and params != "":
                text = params
            else:
                text = f"Reflection: {reflection}\nModel calls directly_answer (separate model-call ommitted)"
            return {"role": "assistant", "text": text, "metadata": metadata}
        else:
            parameters = action.get("parameters", {}) or {}
            if not isinstance(parameters, dict):
                return make_failed_turn("parameters is not a dict", "multihop_call", reflection)
            return {
                "role": "api",
                "request": {"api_name": action["tool_name"], "parameters": parameters},
                "metadata": metadata,
            }
