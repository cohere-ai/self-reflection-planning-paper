"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""

import asyncio
from asyncio.locks import Semaphore
import json
import logging
import os
from collections import deque
from datetime import datetime
from typing import AsyncIterator, List, Union, Optional

from tooltalk.apis import ALL_APIS
from tooltalk.apis.account import (
    ACCOUNT_DB_NAME,
    DeleteAccount,
    LogoutUser,
    RegisterUser,
    UserLogin,
)
from tooltalk.evaluation import safely_divide
from tooltalk.utils.file_utils import get_names_and_paths

logger = logging.getLogger(__name__)

async def _anext(ait):
    '''
    Helper function, because python <3.10 doesn't have it built in
    '''
    return await ait.__anext__()


class ToolExecutor:
    """
    Handles execution of tools and maintains state of databases when simulating conversations.
    """

    def __init__(
        self,
        init_database_dir: str = None,
        ignore_list: List[str] = None,
        account_database: str = ACCOUNT_DB_NAME,
        max_chain_of_actions_per_turn: int = 10,
    ) -> None:
        self.databases = dict()
        self.database_files = dict()
        self.account_database = account_database
        self.ignore_list = ignore_list if ignore_list is not None else list()
        self.session_token = None
        self.execution_semaphore = Semaphore(1)

        for file_name, file_path in get_names_and_paths(init_database_dir):
            database_name, ext = os.path.splitext(file_name)
            if ext == ".json":
                self.database_files[database_name] = file_path
                with open(file_path, "r", encoding="utf-8") as reader:
                    self.databases[database_name] = json.load(reader)
        if self.account_database not in self.databases:
            raise ValueError(f"Account database {self.account_database} not found")

        self.apis = {api.__name__: api for api in ALL_APIS if api.__name__ not in self.ignore_list}
        self.inited_tools = dict()
        self.now_timestamp = None
        self.max_chain_of_actions_per_turn = max_chain_of_actions_per_turn

    def reset_executor(self):
        """
        Reset all tools and databases to their initial state.
        """
        self.databases = dict()
        for database_name, file_path in self.database_files.items():
            with open(file_path, "r", encoding="utf-8") as reader:
                self.databases[database_name] = json.load(reader)
        self.inited_tools = dict()
        self.now_timestamp = None
        self.session_token = None

    def get_init_tool(self, tool_name: str):
        if tool_name in self.inited_tools:
            return self.inited_tools[tool_name]
        cls = self.apis[tool_name]
        account_db = self.databases.get(self.account_database)
        if cls.database_name is not None:
            database = self.databases.get(cls.database_name)
            tool = cls(
                account_database=account_db,
                now_timestamp=self.now_timestamp,
                api_database=database,
            )
        else:
            tool = cls(
                account_database=account_db,
                now_timestamp=self.now_timestamp,
            )

        self.inited_tools[tool_name] = tool
        return tool

    def execute_tool(self, api_name: str, parameters: dict):
        request = {"api_name": api_name, "parameters": parameters}
        if api_name not in self.apis:
            response = {"response": None, "exception": f"API {api_name} not found"}
            return request, response

        tool = self.get_init_tool(api_name)
        if tool.requires_auth:
            if self.session_token is None:
                response = {"response": None, "exception": "User is not logged in"}
                return request, response
            parameters["session_token"] = self.session_token
        if api_name in [UserLogin.__name__, RegisterUser.__name__] and self.session_token is not None:
            username = tool.check_session_token(self.session_token)["username"]
            response = {
                "response": None,
                "exception": f"Only one user can be logged in at a time. Current user is {username}.",
            }
            return request, response

        # execute tool
        response = tool(**parameters)

        # capture session_token and simulate login and logout
        if api_name in [UserLogin.__name__, RegisterUser.__name__] and response["exception"] is None:
            self.session_token = response["response"]["session_token"]
        elif api_name in [LogoutUser.__name__, DeleteAccount.__name__] and response["exception"] is None:
            self.session_token = None
        return request, response

    def compare_api_calls(self, prediction: dict, ground_truth: dict) -> bool:
        api_name = prediction["request"]["api_name"]
        if api_name != ground_truth["request"]["api_name"]:
            return False

        # TODO add session_token if ground truth needs it
        return self.apis[api_name].check_api_call_correctness(prediction, ground_truth)

    def is_action(self, api_name: str) -> bool:
        if api_name not in self.apis:
            return False
        return self.apis[api_name].is_action

    def evaluate_predictions(self, conversation_with_predictions: dict) -> dict:
        """
        Compare predictions in a conversation with complete ground truth in conversation returning metrics.
        Calculates recall over ground truth, where predictions can only match to function in ground truth once.
        Additionally, calculates action precision, number of actions that match ground truth.
        Finally, calculates success, which is recall == 1.0 and action precision == 1.0.

        Metrics:
            predictions: number of predictions
            ground_truths: number of ground truths
            matches: number of predictions that match ground truth
            actions: number of predictions that are actions
            valid_actions: number of actions that match ground truth
            bad_actions: number of actions that don't match ground truth
            precision: matches / predictions
            recall: matches / ground_truths
            action_precision: valid_actions / actions
            bad_action_rate: bad_actions / actions
            success: recall == 1.0 and bad_action_rate == 0.0
        """
        predictions = list()
        ground_truths = list()
        for turn in conversation_with_predictions["conversation"]:
            if turn["role"] == "user":
                continue
            if "predictions" in turn:
                # last prediction will be assistant response
                for prediction in turn["predictions"]:
                    if prediction["role"] == "api":
                        predictions.append(prediction)
            if "apis" in turn:
                ground_truths.extend(turn["apis"])

        # simpler checks that only compare the api_names
        pred_api_names = set(pred["request"]["api_name"] for pred in predictions)
        gt_api_names = set(gt["request"]["api_name"] for gt in ground_truths)
        logger.info(f"Predicted tool calls: {pred_api_names}, \nGround truth tool calls: {gt_api_names}")
        api_name_hit = len(pred_api_names.intersection(gt_api_names))
        api_name_jaccard = (
            api_name_hit / len(pred_api_names.union(gt_api_names)) if len(pred_api_names.union(gt_api_names)) > 0 else 0
        )
        api_name_recall = (
            api_name_hit / len(gt_api_names) if len(gt_api_names) > 0 else (1 if len(pred_api_names) == 0 else 0)
        )
        api_name_precision = api_name_hit / len(pred_api_names) if len(pred_api_names) > 0 else 1

        # remove ground truth as they get matched to predictions
        match_count = 0
        action_count = 0
        valid_action_count = 0
        bad_action_count = 0
        current_ground_truths = deque(ground_truths)
        for prediction in predictions:
            is_match = False
            new_ground_truths = deque()
            while current_ground_truths:
                ground_truth = current_ground_truths.popleft()
                if self.compare_api_calls(prediction, ground_truth):
                    # don't add back in ground truth that matches
                    is_match = True
                    ground_truth["match"] = True
                    break
                else:
                    new_ground_truths.append(ground_truth)
            else:
                logger.debug(f"Failed {json.dumps(prediction, indent=4)}")

            # alter prediction data
            is_action = self.is_action(
                prediction["request"]["api_name"]
            )  # is_action: one that "writes" or has side effects
            is_successful = prediction["exception"] is None  # is_successful: no exception
            is_bad_action = (
                not is_match and is_action and is_successful
            )  # is_bad_action: successful "action" but not matching ground truth
            prediction["match"] = is_match
            prediction["bad_action"] = is_bad_action

            # add back in ground truths that don't match
            while current_ground_truths:
                new_ground_truths.append(current_ground_truths.popleft())
            current_ground_truths = new_ground_truths

            # update counters
            match_count += is_match
            action_count += is_action
            valid_action_count += is_action and is_match
            bad_action_count += is_bad_action

        for ground_truth in current_ground_truths:
            ground_truth["match"] = False

        precision = match_count / len(predictions) if len(predictions) > 0 else 0
        recall = match_count / len(ground_truths) if len(ground_truths) > 0 else 0
        action_precision = valid_action_count / action_count if action_count > 0 else 1
        bad_action_rate = bad_action_count / action_count if action_count > 0 else 0
        success = recall == 1.0 and bad_action_rate == 0.0
        soft_success = recall * (1.0 - bad_action_rate)

        metrics = {
            "predictions": len(predictions),
            "ground_truths": len(ground_truths),
            "matches": match_count,
            "actions": action_count,
            "valid_actions": valid_action_count,
            "bad_actions": bad_action_count,
            "precision": precision,
            "recall": recall,
            "action_precision": action_precision,
            # number of actions matching ground truth
            "bad_action_rate": bad_action_rate,
            # how often an action is bad aka successful but not matching ground truth
            "success": success,
            "soft_success": soft_success,
            "api_name_jaccard": api_name_jaccard,
            "api_name_recall": api_name_recall,
            "api_name_precision": api_name_precision,
        }
        conversation_with_predictions["metrics"] = metrics
        return conversation_with_predictions

    def evaluate_assistance_seeking(self, conversation_with_predictions: dict) -> dict:
        """
        Calculates the accuracy of assistance seeking in terms of accuracy of asking, accuracy of not asking, and overall F1 score of both.
        Additionally Call the evaluate_predictions function to calculate standard ToolTalk metrics.

        Metrics:
            predictions: number of predictions
            ground_truths: number of ground truths
            matches: number of predictions that match ground truth
            actions: number of predictions that are actions
            valid_actions: number of actions that match ground truth
            bad_actions: number of actions that don't match ground truth
            precision: matches / predictions
            recall: matches / ground_truths
            action_precision: valid_actions / actions
            bad_action_rate: bad_actions / actions
            success: recall == 1.0 and bad_action_rate == 0.0
            acc_asking_for_help_when_should: number of times asking for help when should / number of assistance seeking turns
            acc_asking_for_help_when_should_not: number of times asking not for help when model should not / number of non assistance seeking turns
            f1_assistance: 2 * (acc_asking_for_help_when_should * acc_asking_for_help_when_should_not) / (acc_asking_for_help_when_should + acc_asking_for_help_when_should_not)
        """
        # Compute ToolTalk metrics
        conversation_with_predictions = self.evaluate_predictions(conversation_with_predictions)

        is_assistance_seeking_list = list()
        turn_predictions = list()
        for turn in conversation_with_predictions["conversation"]:
            if turn["role"] == "assistant":
                is_assistance_seeking_list.append(turn.get("is_assistance_seeking", False))
            if "predictions" in turn:
                turn_predictions.append(turn["predictions"])

        # Assistance seeking metrics
        asking_for_help_when_should_count = 0
        not_asking_for_help_when_should_not_count = 0
        count_assistance_seeking_turns = 0
        count_non_assistance_seeking_turns = 0

        assert len(turn_predictions) == len(
            is_assistance_seeking_list
        ), "length of prediction turns and is assistance seeking list must match"

        for turn_predictions, is_assistance_seeking in zip(turn_predictions, is_assistance_seeking_list):
            if is_assistance_seeking:
                count_assistance_seeking_turns += 1
                if len(turn_predictions) == 1 and turn_predictions[0].get("role") == "assistant":
                    asking_for_help_when_should_count += 1
            else:
                count_non_assistance_seeking_turns += 1
                for turn in turn_predictions:
                    if turn.get("request") is not None:
                        not_asking_for_help_when_should_not_count += 1
                        break

        acc_asking_for_help_when_should = safely_divide(
            asking_for_help_when_should_count, count_assistance_seeking_turns
        )
        acc_asking_for_help_when_should_not = safely_divide(
            not_asking_for_help_when_should_not_count, count_non_assistance_seeking_turns
        )
        metrics = conversation_with_predictions["metrics"]

        # Update metrics with assistance seeking metrics
        metrics["acc_asking_for_help_when_should"] = acc_asking_for_help_when_should
        metrics["acc_asking_for_help_when_should_not"] = acc_asking_for_help_when_should_not
        metrics["f1_assistance"] = safely_divide(
            2 * (acc_asking_for_help_when_should * acc_asking_for_help_when_should_not),
            (acc_asking_for_help_when_should + acc_asking_for_help_when_should_not),
        )
        conversation_with_predictions["metrics"] = metrics
        return conversation_with_predictions

    def init_conversation_state(self, metadata: dict, api_history: list, user_data: dict = None) -> None:
        self.reset_executor()
        self.now_timestamp = datetime.strptime(metadata["timestamp"], "%Y-%m-%d %H:%M:%S")

        # setting these should never fail, if it does it's a bug in the dataset
        if "session_token" in user_data:
            username = user_data["username"]
            self.session_token = user_data["session_token"]
            self.databases[self.account_database][username]["session_token"] = user_data["session_token"]
        if "verification_code" in user_data:
            username = user_data["username"]
            self.databases[self.account_database][username]["verification_code"] = user_data["verification_code"]

        for api in api_history:
            # this should also never fail, if it does it's a bug in dataset
            self.execute_tool(**api["request"])

    async def gen_predictions(
        self,
        predict_func: callable,
        metadata: dict,
        current_history: List[dict],
        stream: bool = False,
    ) -> AsyncIterator[Union[dict, List[dict]]]:
        '''
        Note, this method is not thread-safe. If it is called independently from
        run_conversation, it should be wrapped in a semaphore.
        '''
        predictions = []
        while True:
            if asyncio.iscoroutinefunction(predict_func):
                prediction = await predict_func(metadata, current_history)
            else:
                prediction = await asyncio.to_thread(predict_func, metadata, current_history)
            if prediction["role"] == "assistant":
                # done with predicting apis
                predictions.append(prediction)
                if stream:
                    yield prediction
                break
            elif prediction["role"] == "api":
                # execute api call
                if len(predictions) == self.max_chain_of_actions_per_turn:
                    logger.warning(
                        "Maximum number of actions per turn reached at "
                        f"{self.max_chain_of_actions_per_turn}. Breaking ..."
                    )
                    break
                if prediction["request"]["parameters"] is None:
                    request = prediction["request"]
                    response = {
                        "response": None,
                        "exception": "Failed to parse API call",
                    }
                else:
                    request, response = self.execute_tool(**prediction["request"])
                prediction_and_response = {
                    "request": request,
                    "response": response["response"],
                    "exception": response["exception"],
                    "metadata": prediction.get("metadata"),
                    "role": "api",
                }
                current_history.append(prediction_and_response)
                predictions.append(prediction_and_response)
                if stream:
                    yield prediction_and_response
            else:
                raise ValueError(f"prediction role should be api or assistant, instead got {prediction['role']}")
        if not stream:
            yield predictions

    async def run_conversation(self, conversation: dict, predict_func: callable, task_and_context: Optional[str] = None):
        """
        Simulates a conversation, calling prediction function
        """
        async with self.execution_semaphore:
            metadata = conversation["metadata"]
            if task_and_context is not None and task_and_context != "":
                metadata["task_and_context"] = task_and_context
            user_data = conversation.get("user")
            ground_truth_history = list()
            api_history = list()

            for turn in conversation["conversation"]:
                if turn["role"] == "user":
                    ground_truth_history.append({"role": "user", "text": turn["text"]})
                    continue

                if turn["role"] != "assistant":
                    raise ValueError(
                        f"turn role must be user or assistant, instead got {turn['role']}"
                    )

                # other turns should be the assistant and could contain API calls
                self.init_conversation_state(metadata, api_history, user_data)

                # generate predictions
                predictions = await _anext(
                    self.gen_predictions(
                        predict_func, metadata, ground_truth_history.copy()
                    )
                )
                # add predictions to original conversation object
                turn["predictions"] = predictions

                if "apis" in turn:
                    for api in turn["apis"]:
                        api_history.append(api)
                        ground_truth_history.append(
                            {
                                "role": "api",
                                "request": api["request"],
                                "response": api["response"],
                                "exception": api["exception"],
                            }
                        )
                ground_truth_history.append({"role": "assistant", "text": turn["text"]})

            return conversation
