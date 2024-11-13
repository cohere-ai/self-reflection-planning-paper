import copy


def strip_session_token(turn: dict) -> dict:
    if turn["role"] != "api" or "session_token" not in turn["request"]["parameters"]:
        return turn

    turn = copy.deepcopy(turn)
    del turn["request"]["parameters"]["session_token"]
    return turn
