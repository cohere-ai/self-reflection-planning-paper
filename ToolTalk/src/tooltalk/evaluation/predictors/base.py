from abc import ABC, abstractmethod
from typing import List, Optional


class BaseAPIPredictor(ABC):
    @abstractmethod
    def __init__(self, force_single_hop: bool = False, *args, **kwargs):
        self.force_single_hop = force_single_hop
        self.tool_calls_backlog = []  # backlog for parallel tool-calls from the current hop

    @abstractmethod
    def predict(self, metadata: dict, conversation_history: List[dict]) -> dict:
        raise NotImplementedError

    def make_tool_call_turn(self, tool_call: dict, metadata: dict) -> dict:
        raise NotImplementedError

    def maybe_make_quick_turn(self, last_turn: dict) -> Optional[dict]:
        if self.tool_calls_backlog:
            # if the last turn was not a tool-call, we clear the backlog
            if last_turn["role"] != "api":
                self.tool_calls_backlog = []
            # otherwise, execute the next tool-call in the backlog if
            # a) it's multi-hop and the previous tool-call succeeded (i.e. no exception)
            # OR
            # b) it's single-hop (coz it'll have no chance to self-recover anyway)
            elif self.force_single_hop or last_turn["exception"] is None:
                tool_call = self.tool_calls_backlog.pop(0)
                return self.make_tool_call_turn(tool_call, {"source": "backlog"})
            # otherwise, we clear the backlog too (i.e. early stop in case of multi-hop)
            else:
                self.tool_calls_backlog = []

        if self.force_single_hop and last_turn["role"] == "api":
            return {
                "role": "assistant",
                "text": "[forced single-hop bot response ommited ...]",
                "metadata": {"source": "hop_done_forced"},
            }

        return None

    def __call__(self, metadata: dict, conversation_history: List[dict]) -> dict:
        """Simple wrapper for convenience."""
        return self.predict(metadata, conversation_history)
