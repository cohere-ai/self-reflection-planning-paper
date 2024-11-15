import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tabulate import tabulate
from tooltalk.evaluation.tool_executor import ToolExecutor
from tooltalk.utils.file_utils import get_names_and_paths
from tooltalk.utils.eval_utils import build_best_conversation

TOOLTALK_EVAL_ROOT = Path(__file__).resolve().parents[2]


def get_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-r", dest="results_dir", type=str, nargs="+", help="Path to results directory")
    parser.add_argument("-e", dest="rerun_eval", action="store_true", help="Rerun evaluation for testcases")
    parser.add_argument("-y", dest="easy_only", action="store_true", help="Only do easy testcases")
    parser.add_argument("-d", dest="hard_only", action="store_true", help="Only do hard testcases")
    parser.add_argument("-s", dest="short", action="store_true", help="Print metrics only without breakdown")
    parser.add_argument("-f", dest="diff_only", action="store_true", help="Print only the differences")
    parser.add_argument("-x", dest="export_mode", action="store_true", help="Export mode")
    return parser


def _collect(
    total_metrics: Counter,
    conversation_breakdown: List[Tuple[str, str]],
    results_dir: str,
    short: bool,
    export_mode: bool,
    do_print: bool,
) -> Tuple[Dict[str, float], Dict[str, str], List[Tuple[str, str]]]:
    if len(total_metrics) == 0:
        print("No conversations found:", results_dir)
        raise SystemExit
    metrics = {
        "num_conversations": total_metrics["num_conversations"],
        "precision": 0.0 if total_metrics["predictions"] == 0 else total_metrics["matches"] / total_metrics["predictions"],
        "recall": total_metrics["matches"] / total_metrics["ground_truths"],
        "action_precision": total_metrics["valid_actions"] / total_metrics["actions"],
        "bad_action_rate": total_metrics["bad_actions"] / total_metrics["actions"],
        "success_rate": total_metrics["success"] / total_metrics["num_conversations"],
        "soft_success_rate": total_metrics["soft_success"] / total_metrics["num_conversations"],
        "api_name_recall": total_metrics["api_name_recall"] / total_metrics["num_conversations"],
        "api_name_precision": total_metrics["api_name_precision"] / total_metrics["num_conversations"],
    }
    if export_mode:
        table = {
            "success rate": [f'{metrics["success_rate"]:.1%}'],
            "soft success rate": [f'{metrics["soft_success_rate"]:.1%}'],
            "precision": [f'{metrics["precision"]:.1%}'],
            f'recall ({total_metrics["ground_truths"]})': [f'{metrics["recall"]:.1%}'],
            "bad action rate": [f'{metrics["bad_action_rate"]:.1%}'],
        }
    else:
        table = {
            "success rate": [f'{metrics["success_rate"]:.1%} ({metrics["soft_success_rate"]:.1%})'],
            "precision": [f'{metrics["precision"]:.1%} ({total_metrics["predictions"]})'],
            f'recall ({total_metrics["ground_truths"]})': [f'{metrics["recall"]:.1%}'],
            "bad action rate": [f'{metrics["bad_action_rate"]:.1%} ({total_metrics["actions"]})'],
            "api-name p": [f'{metrics["api_name_precision"]:.1%}'],
            "api-name r": [f'{metrics["api_name_recall"]:.1%}'],
        }
    if do_print:
        print(tabulate(table, headers="keys", tablefmt="github"))
    conversation_breakdown = sorted(conversation_breakdown, key=lambda x: x[0])
    if not short and do_print:
        print(
            "Result breakdown:\n"
            + tabulate(conversation_breakdown, headers=["Testcase", "Recall, BA Rate"], tablefmt="github")
        )
    return (metrics, {key: value[0] for key, value in table.items()}, conversation_breakdown)
    


def top_n_collect(
    results_dirs: List[str],
    tool_executor: Optional[ToolExecutor],
    easy_only: bool = True,
    short: bool = False,
    export_mode: bool = False,
    do_print: bool = True,
) -> Tuple[Dict[str, float], Dict[str, str], List[Tuple[str, str]]]:
    total_metrics = Counter()
    conversation_breakdown = []


    for file_name, file_path in get_names_and_paths(results_dirs[0]):
        if not file_name.endswith(".json"):
            continue
        if easy_only and not file_name.endswith("-easy.json"):
            continue
        if not easy_only and file_name.endswith("-easy.json"):
            continue

        conversation = build_best_conversation(results_dirs, file_name)
        conversation = tool_executor.evaluate_predictions(conversation)
        
        # with open(file_path, "r", encoding="utf-8") as reader:
        #     conversation = json.load(reader)
        # if tool_executor:
        conversation = tool_executor.evaluate_predictions(conversation)
        if conversation["metrics"]["success"]:
            repr = "✅"
        else:
            repr = f"❌ ({conversation['metrics']['recall']:.2f}, {conversation['metrics']['bad_action_rate']:.2f})"
        assert file_name.endswith(".json")
        conversation_breakdown.append((file_name[:-5], repr))
        total_metrics += conversation["metrics"]
        total_metrics["num_conversations"] += 1
        # if tool_executor:
        #     with open(file_path, "w", encoding="utf-8") as writer:
        #         json.dump(conversation, writer, indent=4)
        
    return _collect(
        total_metrics,
        conversation_breakdown,
        results_dirs[0],
        short,
        export_mode,
        do_print,
    )


def collect(
    results_dir: str,
    tool_executor: Optional[ToolExecutor],
    easy_only: bool = True,
    short: bool = False,
    export_mode: bool = False,
    do_print: bool = True,
) -> Tuple[Dict[str, float], Dict[str, str], List[Tuple[str, str]]]:
    total_metrics = Counter()
    conversation_breakdown = []
    for file_name, file_path in get_names_and_paths(results_dir):
        if easy_only and not file_name.endswith("-easy.json"):
            continue
        if not easy_only and file_name.endswith("-easy.json"):
            continue
        with open(file_path, "r", encoding="utf-8") as reader:
            conversation = json.load(reader)
            if tool_executor:
                conversation = tool_executor.evaluate_predictions(conversation)
            if conversation["metrics"]["success"]:
                repr = "✅"
            else:
                repr = f"❌ ({conversation['metrics']['recall']:.2f}, {conversation['metrics']['bad_action_rate']:.2f})"
            assert file_name.endswith(".json")
            conversation_breakdown.append((file_name[:-5], repr))
            total_metrics += conversation["metrics"]
            total_metrics["num_conversations"] += 1
        if tool_executor:
            with open(file_path, "w", encoding="utf-8") as writer:
                json.dump(conversation, writer, indent=4)
    if len(total_metrics) == 0:
        print("No conversations found:", results_dir)
        raise SystemExit
    metrics = {
        "num_conversations": total_metrics["num_conversations"],
        "precision": total_metrics["matches"] / total_metrics["predictions"],
        "recall": total_metrics["matches"] / total_metrics["ground_truths"],
        "action_precision": total_metrics["valid_actions"] / total_metrics["actions"],
        "bad_action_rate": total_metrics["bad_actions"] / total_metrics["actions"],
        "success_rate": total_metrics["success"] / total_metrics["num_conversations"],
        "soft_success_rate": total_metrics["soft_success"] / total_metrics["num_conversations"],
        "api_name_recall": total_metrics["api_name_recall"] / total_metrics["num_conversations"],
        "api_name_precision": total_metrics["api_name_precision"] / total_metrics["num_conversations"],
    }
    if export_mode:
        table = {
            "success rate": [f'{metrics["success_rate"]:.1%}'],
            "soft success rate": [f'{metrics["soft_success_rate"]:.1%}'],
            "precision": [f'{metrics["precision"]:.1%}'],
            f'recall ({total_metrics["ground_truths"]})': [f'{metrics["recall"]:.1%}'],
            "bad action rate": [f'{metrics["bad_action_rate"]:.1%}'],
        }
    else:
        table = {
            "success rate": [f'{metrics["success_rate"]:.1%} ({metrics["soft_success_rate"]:.1%})'],
            "precision": [f'{metrics["precision"]:.1%} ({total_metrics["predictions"]})'],
            f'recall ({total_metrics["ground_truths"]})': [f'{metrics["recall"]:.1%}'],
            "bad action rate": [f'{metrics["bad_action_rate"]:.1%} ({total_metrics["actions"]})'],
            "api-name p": [f'{metrics["api_name_precision"]:.1%}'],
            "api-name r": [f'{metrics["api_name_recall"]:.1%}'],
        }
    if do_print:
        print(tabulate(table, headers="keys", tablefmt="github"))
    conversation_breakdown = sorted(conversation_breakdown, key=lambda x: x[0])
    if not short and do_print:
        print(
            "Result breakdown:\n"
            + tabulate(conversation_breakdown, headers=["Testcase", "Recall, BA Rate"], tablefmt="github")
        )
    return (metrics, {key: value[0] for key, value in table.items()}, conversation_breakdown)


def to_variant(results_dir: str):
    return results_dir.rstrip("/").rsplit("/", 1)[-1]


def joint_print(
    results_dirs: List[str],
    tool_executor: Optional[ToolExecutor],
    easy_only: bool = True,
    short: bool = False,
    diff_only: bool = False,
    export_mode: bool = False,
):
    table = []
    breakdown = []
    variants = [to_variant(r) for r in results_dirs]
    for results_dir in results_dirs:
        _, _tb, _bd = collect(results_dir, tool_executor, easy_only, short, export_mode, do_print=False)
        tb = {"variant": to_variant(results_dir)}
        tb.update(_tb)
        table.append(tb)
        bd = list(zip(*_bd))
        if short:
            continue
        if breakdown:
            assert breakdown[0] == bd[0]
            breakdown.append(bd[1])
        else:
            breakdown = bd

    # check for shared results_dir names (excluding numbers)
    unique_results_dirs = set(r.split("/")[-1].rstrip("0123456789") for r in results_dirs)
    if len(unique_results_dirs) != len(results_dirs):
        for u_results_dir in unique_results_dirs:
            shared_dirs = [r for r in results_dirs if (u_results_dir in r and len(r.split("/")[-1].replace(u_results_dir, "")) <= 1)]

            if len(shared_dirs) > 1:
                _, _tb, _bd = top_n_collect(shared_dirs, tool_executor, easy_only, short, export_mode, do_print=False)
                as_var = to_variant(u_results_dir + f"-top-{len(shared_dirs)}")
                tb = {"variant": as_var}
                variants.append(as_var)
                tb.update(_tb)
                table.append(tb)
                bd = list(zip(*_bd))
                if short:
                    continue
                if breakdown:
                    assert breakdown[0] == bd[0]
                    breakdown.append(bd[1])
                else:
                    breakdown = bd
    
    print(tabulate(table, headers="keys", tablefmt="github"))


    if not short:
        # turn columns into rows
        breakdown = zip(*breakdown)
        if diff_only:
            # breakdown = [row for row in breakdown if len(set(row[1:])) > 1]
            breakdown = [row for row in breakdown if len(set(x[0] for x in row[1:])) > 1]
        print("Result breakdown:\n" + tabulate(breakdown, headers=["Testcase"] + variants, tablefmt="github"))


def main(flags: Optional[List[str]] = None):
    parser = get_arg_parser()
    args = parser.parse_args(flags)

    database_dir = str(TOOLTALK_EVAL_ROOT / "data" / "databases")
    tool_executor = None
    if args.rerun_eval:
        tool_executor = ToolExecutor(init_database_dir=database_dir)

    if len(args.results_dir) == 1:
        results_dir = args.results_dir[0]
        if not args.hard_only:
            print("=== EASY ===")
            collect(results_dir, tool_executor, easy_only=True, short=args.short, export_mode=args.export_mode)
        if not args.easy_only:
            print("=== HARD ===")
            collect(results_dir, tool_executor, easy_only=False, short=args.short, export_mode=args.export_mode)
        return

    if not args.hard_only:
        print("=== EASY ===")
        joint_print(
            args.results_dir,
            tool_executor,
            easy_only=True,
            short=args.short,
            diff_only=args.diff_only,
            export_mode=args.export_mode,
        )
    if not args.easy_only:
        print("=== HARD ===")
        joint_print(
            args.results_dir,
            tool_executor,
            easy_only=False,
            short=args.short,
            diff_only=args.diff_only,
            export_mode=args.export_mode,
        )


if __name__ == "__main__":
    main()