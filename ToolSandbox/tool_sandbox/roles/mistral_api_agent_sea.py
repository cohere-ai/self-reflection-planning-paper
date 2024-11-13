# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Agent role for any model that conforms to OpenAI tool use API"""

from typing import Any, Iterable, List, Literal, Optional, Union, cast
from deep_planner.deep_planner import DeepPlanner

from openai import NOT_GIVEN, NotGiven, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from mistralai.models.chat_completion import (
    ChatMessage,
    Function,
    FunctionCall,
    ToolCall,
)
from mistralai.client import MistralClient


from requests.exceptions import HTTPError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
    to_openai_messages,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.common.utils import all_logging_disabled, to_docstring
from tool_sandbox.roles.base_role import BaseRole


def mistral_create_messages(messages: List[Message]) -> List[ChatMessage]:
    """Converts messages to Mistral ChatMessage format

    Args:
        messages:   List of messages to be converted

    Returns:
        List of Mistral ChatMessage
    """
    mistral_messages = []
    for message in messages:
        if message.role == RoleType.SYSTEM:
            mistral_messages.append(
                ChatMessage(
                    role="system",
                    content=message.content,
                )
            )
        elif message.role == RoleType.USER:
            mistral_messages.append(
                ChatMessage(
                    role="user",
                    content=message.content,
                )
            )
        elif message.role == RoleType.AGENT:
            mistral_messages.append(
                ChatMessage(
                    role="assistant",
                    content=message.content,
                )
            )
        elif message.role == RoleType.EXECUTION_ENVIRONMENT:
            tool_call = ToolCall(
                id=message.openai_tool_call_id,
                function=Function(
                    name=message.openai_function_name,
                ),
            )
            mistral_messages[-1].tool_calls.append(tool_call)
    return mistral_messages

class MistralAPIAgentSea(BaseRole):
    """Agent role for any model that conforms to OpenAI tool use API"""

    role_type: RoleType = RoleType.AGENT

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

    def __init__(self, model_name="mistral-large-2407") -> None:
        # We set the `base_url` explicitly here to avoid picking up the
        # `OPENAI_BASE_URL` environment variable that may be set for serving models as
        # OpenAI API compatible servers.
        # self.openai_client: OpenAI = OpenAI(base_url="https://api.openai.com/v1")
        self.mistral_client = MistralClient(max_retries=0)
        self.model_name = model_name
        self.current_tools = None

        self.deep_planner = DeepPlanner(
            n_plans=3, 
            temperature=1, 
            plan_generator_func=self.model_inference_sea, 
            eval_generate_func=self.model_inference_sea_prompt,
            compile_generate_func=self.model_inference_sea_prompt,
            extract_plan_func=self.extract_plan_from_completion,
            with_final_feedback_round=False,
            # with_multiple_compilations=True
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

    def respond(self, ending_index: Optional[int] = None) -> None:
        """Reads a List of messages and attempt to respond with a Message

        Specifically, interprets system, user, execution environment messages and sends out NL response to user, or
        code snippet to execution environment.

        Message comes from current context, the last k messages should be directed to this role type
        Response are written to current context as well. n new messages, addressed to appropriate recipient
        k != n when dealing with parallel function call and responses. Parallel function call are expanded into
        individual messages, parallel function call responses are combined as 1 OpenAI API request

        Args:
            ending_index:   Optional index. Will respond to message located at ending_index instead of most recent one
                            if provided. Utility for processing system message, which could contain multiple entries
                            before each was responded to

        Raises:
            KeyError:   When the last message is not directed to this role
        """
        messages: List[Message] = self.get_messages(ending_index=ending_index)
        response_messages: List[Message] = []
        self.messages_validation(messages=messages)
        # Keeps only relevant messages
        messages = self.filter_messages(messages=messages)
        plan_message_indexes = [i for i, message in enumerate(messages) if message.content.startswith("Try executing the following plan:")]
        if len(plan_message_indexes) > 1:
            # remove all plan messages except the most recent one
            plan_message_indexes.pop()
            messages = [message for i, message in enumerate(messages) if i not in plan_message_indexes]

        # Does not respond to System

        if messages[-1].sender == RoleType.SYSTEM:
            return
        # Get OpenAI tools if most recent message is from user
        available_tools = self.get_available_tools()
        available_tool_names = set(available_tools.keys())
        openai_tools = (
            convert_to_openai_tools(available_tools)
            if messages[-1].sender == RoleType.USER
            or messages[-1].sender == RoleType.EXECUTION_ENVIRONMENT
            else NOT_GIVEN
        )
        self.current_tools = openai_tools
        # We need a cast here since `convert_to_openai_tool` returns a plain dict, but
        # `ChatCompletionToolParam` is a `TypedDict`.

        # openai_tools = cast(
        #     Union[Iterable[ChatCompletionToolParam], NotGiven],
        #     openai_tools,
        # )

        # Convert to OpenAI messages.
        current_context = get_current_context()

        # Call model
        if messages[-1].sender == RoleType.USER:
            assert messages[0].sender == RoleType.SYSTEM
            openai_messages, _ = to_openai_messages(messages)
            system_prompt = openai_messages[0]["content"]
            system_prompt = system_prompt + " ".join(self.planning_prompt) + "".join(self.planning_example)
            system_prompt = system_prompt + "\n\nNote that you only need to provide a plan if you are using tools (i.e., not if you asking a question of the user). Once the user confirms the plan, you can proceed with the execution."
            openai_messages[0]["content"] = system_prompt

            tools_string = "\n\n".join([to_docstring(tool["function"]) for tool in openai_tools])                            
            conversation_history_str = "\n".join([turn.sender + " -> " + turn.recipient + ": " + turn.content if (not turn.openai_function_name) else \
                                                  (turn.sender + " -> " + turn.recipient + ": " + f"[{turn.openai_function_name}] parameters: " + turn.content.split("_parameters =")[-1].split("\n")[0].strip() if turn.sender == RoleType.AGENT else \
                                                   turn.sender + " -> " + turn.recipient + ": " + f"[{turn.openai_function_name}] " + turn.content )for turn in messages])
            
            plan = self.deep_planner.plan(planning_messages=openai_messages, previous_conversation_str=conversation_history_str, tools_string=tools_string)
            messages.append(Message(sender=RoleType.SYSTEM, recipient=RoleType.AGENT, content="Try executing the following plan:\n" + plan))
            openai_messages, _ = to_openai_messages(messages)
            response_messages.append(   
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content="I will try executing the following plan:\n" + plan,
                )
            )
            response = self.model_inference(
                messages=openai_messages, tools=openai_tools
            )

        else:
            openai_messages, _ = to_openai_messages(messages)
            response = self.model_inference(
                messages=openai_messages, tools=openai_tools
            )
        # Parse response
        openai_response_message = response.choices[0].message
        # Message contains no tool call, aka addressed to user
        if openai_response_message.tool_calls is None:
            assert openai_response_message.content is not None
            response_messages.append(
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=openai_response_message.content,
                )
            )
        else:
            assert openai_tools is not NOT_GIVEN
            for tool_call in openai_response_message.tool_calls:
                # The response contains the agent facing tool name so we need to get
                # the execution facing tool name when creating the Python code.
                execution_facing_tool_name = (
                    current_context.get_execution_facing_tool_name(
                        tool_call.function.name
                    )
                )
                response_messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=openai_tool_call_to_python_code(
                            tool_call,
                            available_tool_names,
                            execution_facing_tool_name=execution_facing_tool_name,
                        ),
                        openai_tool_call_id=tool_call.id,
                        openai_function_name=tool_call.function.name,
                    )
                )
        self.add_messages(response_messages)

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(HTTPError),
    )
    def model_inference(
        self,
        messages: list[
            dict[
                Literal["role", "content", "tool_call_id", "name", "tool_calls"],
                Any,
            ]
        ],
        tools: Union[Iterable[ChatCompletionToolParam], NotGiven],
    ) -> ChatCompletion:
        """Run OpenAI model inference

        Args:
            openai_messages:    List of OpenAI API format messages
            openai_tools:       List of OpenAI API format tools definition

        Returns:
            OpenAI API chat completion object
        """
        with all_logging_disabled():
            return self.mistral_client.chat(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                tools=tools,
            )


    def model_inference_sea(
        self,
        messages: list[
            dict[
                Literal["role", "content", "tool_call_id", "name", "tool_calls"],
                Any,
            ]
        ],
        temperature: float = 0.0,
    ) -> str:
        """Run OpenAI model inference

        Args:
            openai_messages:    List of OpenAI API format messages
            openai_tools:       List of OpenAI API format tools definition

        Returns:
            OpenAI API chat completion object
        """
        assert self.current_tools is not None
        with all_logging_disabled():
            return self.mistral_client.chat(
                model=self.model_name,
                messages=cast(list[ChatCompletionMessageParam], messages),
                temperature=temperature, # added
                tools=self.current_tools
            ).choices[0].message.content

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(HTTPError),
    )
    def model_inference_sea_prompt(
        self,
        prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """Run OpenAI model inference

        Args:
            openai_messages:    List of OpenAI API format messages
            openai_tools:       List of OpenAI API format tools definition

        Returns:
            OpenAI API chat completion object
        """
        with all_logging_disabled():
            return self.mistral_client.chat(
                model=self.model_name,
                messages=[{ "role": "user", "content": prompt }],
                temperature=temperature, # added
            ).choices[0].message.content