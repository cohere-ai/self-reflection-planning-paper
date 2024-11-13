
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Agent role for the Cohere models hosted as OpenAI compatible servers using vLLM."""

import os
import cohere
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from cohere.types import (
    ToolCall,
)
from langchain_cohere.react_multi_hop.default_prompt_constants import (
    _SpecialToken,
)
from langchain_cohere.react_multi_hop.parsing import CohereToolsReactAgentOutputParser
from langchain_cohere.react_multi_hop.prompt import (
    create_directly_answer_tool,
    multi_hop_prompt_partial,
    render_messages,
    render_observations,
    render_structured_preamble,
    render_tool,
)
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openai import OpenAI
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
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tool
from tool_sandbox.roles.base_role import BaseRole


def tool_call_to_python_code(
    execution_facing_tool_name: str,
    tool_call: ToolCall,
    available_tool_names: set[str],
) -> str:
    """Converts a Cohere tool call into Python code for calling the function.
    Args:
        execution_facing_tool_name:  The execution facing name of the function. In the
                                     case of tool name scrambling the Cohere API in- and
                                     outputs are filled with scrambled tool names. When
                                     executing the code we need to use the actual tool
                                     name.
        tool_call:                   The Cohere tool call describing the function name and arguments.
        available_tool_names:        Set of available tools.

    Returns:
        The Python code for making the tool call.

    Raises:
        KeyError: If the selected tool is not a known tool.
    """
    agent_facing_tool_name = tool_call.name
    if agent_facing_tool_name not in available_tool_names:
        raise KeyError(
            f"Agent tool call {agent_facing_tool_name=} is not a known allowed tool. Options "
            f"are {available_tool_names=}."
        )

    function_call_code = (
        f"{agent_facing_tool_name}_parameters = {tool_call.parameters}\n"
        f"{agent_facing_tool_name}_response = {execution_facing_tool_name}(**{agent_facing_tool_name}_parameters)\n"
        f"print(repr({agent_facing_tool_name}_response))"
    )
    return function_call_code

class CohereAgent(BaseRole):
    """Cohere agent using Cohere models hosted as an OpenAI compatible server using vLLM."""

    role_type: RoleType = RoleType.AGENT
    model_name: str
    previous_tool_call_turns: Dict[str, Sequence[BaseMessage]] = {}

    def __init__(self, model_name: str) -> None:
        super().__init__()

        self.model_name = model_name
        # assert (
        #     "OPENAI_BASE_URL" in os.environ
        # ), "The `OPENAI_BASE_URL` environment variable must be set."
        self.client = OpenAI(api_key="EMPTY")

    def respond(self, ending_index: Optional[int] = None) -> None:
        """Reads a List of messages and attempt to respond with a Message
        Specifically, interprets system, user, execution environment messages and sends out NL response to user, or
        code snippet to execution environment.
        Message comes from current context, the last k messages should be directed to this role type
        Response are written to current context as well. n new messages, addressed to appropriate recipient
        k != n when dealing with parallel function call and responses. Parallel function call are expanded into
        individual messages, parallel function call responses are combined as 1 API request
        Args:
            ending_index:   Optional index. Will respond to message located at ending_index instead of most recent one
                            if provided. Utility for processing system message, which could contain multiple entries
                            before each was responded to
        Raises:
            KeyError:   When the last message is not directed to this role
        """
        messages: list[Message] = self.get_messages(ending_index=ending_index)
        response_messages: list[Message] = []
        self.messages_validation(messages=messages)
        # Keeps only relevant messages
        messages = self.filter_messages(messages=messages)
        # Does not respond to System
        if messages[-1].sender == RoleType.SYSTEM:
            return

        # Get tools if most recent message is from user
        available_tools: Dict[str, Callable[..., Any]] = {}
        if (
            messages[-1].sender == RoleType.USER
            or messages[-1].sender == RoleType.EXECUTION_ENVIRONMENT
        ):
            available_tools = self.get_available_tools()

        # Call model
        response_messages = self.model_inference(
            messages=messages,
            available_tools=available_tools,
            agent_to_execution_facing_tool_name=get_current_context().get_agent_to_execution_facing_tool_name(),
        )

        self.add_messages(response_messages)

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(HTTPError),
    )
    def model_inference(
        self,
        messages: List[Message],
        available_tools: Dict[str, Callable[..., Any]],
        agent_to_execution_facing_tool_name: Dict[str, str],
    ) -> List[Message]:
        prompt = self.create_prompt(messages=messages, available_tools=available_tools)

        client = cohere.Client()
        response = client.chat(raw_prompting=True, message=prompt, model=self.model_name, temperature=0.0)

        return self.completion_to_messages(
            # completion=response.choices[0].text,
            completion=response.text,
            available_tool_names=set(available_tools.keys()),
            agent_to_execution_facing_tool_name=agent_to_execution_facing_tool_name,
        )

    def create_prompt(
        self,
        messages: List[Message],
        available_tools: Dict[str, Callable[..., Any]],
    ) -> str:
        langchain_messages = to_langchain_messages(
            messages, self.previous_tool_call_turns
        )
        history, user_message, steps = _split_langchain_messages(
            langchain_messages=langchain_messages
        )

        prompt = multi_hop_prompt_partial.format_prompt(
            structured_preamble=render_structured_preamble(),
            tools="\n\n".join([
                render_tool(
                    json_schema=convert_to_openai_tool(tool, name=name)["function"]
                )
                for name, tool in available_tools.items()
            ]),
            history=render_messages(history),
            user_prompt=render_messages([user_message]) if user_message else "",
            steps=render_messages(steps)
            + f"{_SpecialToken.start_turn.value}{_SpecialToken.role_chatbot.value}",
        ).to_string()

        return prompt

    def completion_to_messages(
        self,
        completion: str,
        available_tool_names: Set[str],
        agent_to_execution_facing_tool_name: Dict[str, str],
    ) -> List[Message]:
        parser = CohereToolsReactAgentOutputParser()
        try:
            parsed_response = parser.parse(completion)
        except ValueError:
            return [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=completion,
                )
            ]

        if isinstance(parsed_response, AgentFinish):
            return [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=parsed_response.return_values.get("output", ""),
                )
            ]

        messages: List[Message] = []
        for message in parsed_response:
            assert isinstance(message, AgentActionMessageLog)
            if message.tool == create_directly_answer_tool().name:
                # The directly_answer tool is added to every Cohere inference call, it represents NOT_GIVEN.
                messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.USER,
                        content="Bot response omitted",
                    )
                )
                return messages

            tool_call_id = str(uuid.uuid4())
            self.previous_tool_call_turns[tool_call_id] = message.message_log
            messages.append(
                Message(
                    sender=self.role_type,
                    recipient=RoleType.EXECUTION_ENVIRONMENT,
                    content=tool_call_to_python_code(
                        execution_facing_tool_name=agent_to_execution_facing_tool_name[
                            message.tool
                        ],
                        tool_call=ToolCall(
                            name=message.tool,
                            parameters=message.tool_input,  # type: ignore[arg-type]
                        ),
                        available_tool_names=available_tool_names,
                    ),
                    openai_tool_call_id=tool_call_id,
                    openai_function_name=message.tool,
                )
            )

        return messages


def to_langchain_messages(
    messages: List[Message], previous_tool_calls: Dict[str, Sequence[BaseMessage]]
) -> List[BaseMessage]:
    """Converts ToolSandbox messages into Langchain messages so that they can be rendered into the prompt.
    Args:
        messages: A list of ToolSandbox messages to convert.
        previous_tool_calls: A map of tool call id to rendered prompt contents, to faithfully recreate the prompt.

    Returns:
        A list of LangChain BaseMessage instances.
    """
    langchain_messages: List[BaseMessage] = []
    observation_idx = 0
    tool_results: List[Mapping[str, str]] = []

    def is_tool_result(m: Message) -> bool:
        return (
            m.sender == RoleType.EXECUTION_ENVIRONMENT and m.recipient == RoleType.AGENT
        )

    def is_tool_call(m: Message) -> bool:
        return (
            m.sender == RoleType.AGENT and m.recipient == RoleType.EXECUTION_ENVIRONMENT
        )

    for i, message in enumerate(messages):
        if message.sender == RoleType.SYSTEM and message.recipient == RoleType.AGENT:
            langchain_messages.append(SystemMessage(content=message.content))
        elif message.sender == RoleType.USER and message.recipient == RoleType.AGENT:
            langchain_messages.append(HumanMessage(content=message.content))
        elif is_tool_result(message):
            # Group multiple tool results into a single rendered turn.
            tool_results.append({"output": message.content})
            next_message_is_tool_result = False
            try:
                next_message_is_tool_result = is_tool_result(messages[i + 1])
            except IndexError:
                next_message_is_tool_result = False
            finally:
                if not next_message_is_tool_result:
                    langchain_message, observation_idx = render_observations(
                        tool_results, observation_idx
                    )
                    tool_results = []
                    langchain_messages.append(langchain_message)
        elif is_tool_call(message):
            # Parallel tool calls are executed by the sandbox in parallel, which means multiple tool sandbox messages.
            # However, these are contained in a single rendered turn in the prompt. To faithfully recreate the prompt
            # we need to combine parallel tool calls and use the prompt contents of just one.
            next_message_is_tool_call = False
            try:
                next_message_is_tool_call = is_tool_call(messages[i + 1])
            except IndexError:
                next_message_is_tool_call = False
            finally:
                if not next_message_is_tool_call:
                    assert message.openai_tool_call_id is not None
                    langchain_messages.extend(
                        previous_tool_calls[message.openai_tool_call_id]
                    )

        elif message.sender == RoleType.AGENT and message.recipient == RoleType.USER:
            langchain_messages.append(AIMessage(content=message.content))
        else:
            raise ValueError(
                "Unrecognized sender recipient pair "
                f"{(message.sender, message.recipient)}"
            )

    return langchain_messages


def _split_langchain_messages(
    langchain_messages: List[BaseMessage],
) -> Tuple[List[BaseMessage], Optional[HumanMessage], List[BaseMessage]]:
    """The Cohere prompt has separate inputs for history, the most recent user message, and the turns after the user
    message. Split a list of messages into these three inputs.

    Args:
        langchain_messages: A list of LangChain messages to split.

    Returns:
        A tuple containing the input variables for the Cohere prompt.
    """
    user_idx: Optional[int] = None
    for i, message in enumerate(langchain_messages):
        if isinstance(message, HumanMessage):
            user_idx = i

    if user_idx is None:
        return langchain_messages, None, []

    history = langchain_messages[:user_idx]
    steps = langchain_messages[user_idx:]
    try:
        user_message: Optional[HumanMessage] = steps.pop(0)  # type:ignore[assignment]
    except IndexError:
        user_message = None
    return history, user_message, steps


def _to_rendered_tools(tools: Dict[str, Callable[..., Any]]) -> str:
    """Renders the tools section of a Cohere prompt.
    Args:
        tools: The ToolSandbox tools to render.

    Returns:
        A string to use in the prompt.
    """
    rendered = [
        render_tool(json_schema=convert_to_openai_tool(tool, name=name)["function"])
        for name, tool in tools.items()
    ]
    # The directly answer tool is added to every Cohere inference call, it represents NOT_GIVEN.
    rendered.append(render_tool(tool=create_directly_answer_tool()))

    return "\n\n".join(rendered)