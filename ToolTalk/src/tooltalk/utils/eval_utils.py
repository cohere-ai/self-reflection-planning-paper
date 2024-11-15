import json
from typing import List


def get_all_conversations(
    results_dirs: List[str],
    file_name: str,
):
    conversations = []
    for res_dir in results_dirs:
        with open(res_dir + "/" + file_name, "r", encoding="utf-8") as reader:
            c = json.load(reader)
            conversations.append(c)

    return conversations


def build_best_conversation(
    results_dirs: List[str],
    file_name: str,
):
    conversations = get_all_conversations(results_dirs, file_name)

    # build most successful conversation
    conversation = conversations[0]

    turns = conversation["conversation"]
    output_turns = []
    for turn_index, turn in enumerate(turns):

        if turn["role"] == "assistant":
            ground_truths = []
            turn_options = [c["conversation"][turn_index] for c in conversations]
            if "apis" in turn:
                ground_truths.extend(turn["apis"])
            else:
                no_call_options = [t for t in turn_options if len([p for p in t["predictions"] if "request" in p]) == 0]
                if len(no_call_options) > 0:
                    turn_options = no_call_options

            option_results = []
            for t in turn_options:
                pred_matches = 0
                pred_bad_actions = 0
                api_matches = 0


                if "apis" in t:
                    api_matches = len([a for a in t["apis"] if "match" in a and a["match"]])


                if "predictions" in t:
                    pred_matches = len([p for p in t["predictions"] if "match" in p and p["match"]])
                    pred_non_matches = len([p for p in t["predictions"] if "match" in p and not p["match"]])
                    pred_bad_actions = len([p for p in t["predictions"] if "bad_action" in p and p["bad_action"]])
                
                option_results.append((t, {
                    "pred_matches": pred_matches,
                    "pred_non_matches": pred_non_matches,
                    "pred_bad_actions": pred_bad_actions,
                    "api_matches": api_matches,
                }))

                              
            # get most successful turn (with the most prediction matches)
            best_turn = max(option_results, key=lambda x: x[1]["pred_matches"])

            # if there are multiple turns with the same number of matches, get the one with the least bad actions
            best_turn = min([t for t in option_results if t[1]["pred_matches"] == best_turn[1]["pred_matches"]], key=lambda x: x[1]["pred_bad_actions"])
            
            output_turns.append(best_turn[0])

        else:
            output_turns.append(turn)

    conversation["conversation"] = output_turns
    return conversation
