import argparse
import asyncio
import json
import logging
import os
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Optional

import openai
from tabulate import tabulate
from tooltalk.apis import ALL_APIS, APIS_BY_NAME, SUITES_BY_NAME
from tooltalk.evaluation import safely_divide
from tooltalk.evaluation.evaluate_openai import EvalModes
from tooltalk.evaluation.predictors.all import (
    AnthropicPredictor,
    CoherePredictor,
    MistralPredictor,
    OpenAIPredictor,
)
from tooltalk.evaluation.predictors.anthropic_predictor import (
    ENDPOINTS as ANTHROPIC_ENDPOINTS,
)
from tooltalk.evaluation.tool_executor import ToolExecutor
from tooltalk.utils.file_utils import get_names_and_paths
from tqdm import tqdm

logger = logging.getLogger(__name__)

TOOLTALK_EVAL_ROOT = Path(__file__).parents[3].absolute()


def get_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=["easy", "tooltalk", "assistance_subset"],
        default=["easy", "tooltalk", "assistance_subset"],
        help='Name(s) of the dataset for models to evaluate. Defaults to both "easy" and "tooltalk".',
    )
    parser.add_argument(
        "--model_provider",
        type=str,
        choices=["openai", "cohere", "mistral", "anthropic", "bedrock"],
        default="cohere",
        help="Model provider",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="command-r",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--api_mode",
        type=str,
        choices=["exact", "suite", "all"],
        default="all",
        help="API mode to use for evaluation, determines which api docs to include",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset evaluation writing over any cached results",
    )
    parser.add_argument(
        "--disable_documentation",
        action="store_true",
        help="Disable documentation sent to GPT-4 replacing with empty strings",
    )
    parser.add_argument(
        "--modes",
        choices=list(x.value for x in EvalModes),
        type=str,
        nargs="+",
        default=list(EvalModes),
        help="Evaluation modes",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode (more logs)")
    parser.add_argument(
        "--testcases",
        type=str,
        nargs="+",
        help="List of test-cases to run (will run everything by default)",
    )
    parser.add_argument(
        "--max_chain_of_actions_per_turn",
        type=int,
        default=10,
        help="Max number of actions per turn (to prevent infinite loops in case of multi-hop)",
    )
    parser.add_argument(
        "--force_single_hop",
        action="store_true",
        help="Force single-hop (i.e. force yielding a bot response to the user after each tool-use call). "
        "Defaults to False for all models except cohere, which defaults to True as user_followup takes precedence.",
    )
    parser.add_argument(
        "--user_followup",
        type=str,
        help="Insert a filler user followup after each hop (in case of multi-hop). Defaults to None, which signals single-hop.",
    )
    parser.add_argument(
        "--do_augmented_gen",
        action="store_false",
        dest="skip_augmented_gen",
        help="Do NOT skip the augmented-generation step",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--results_subfolder",
        type=str,
        help="Additional level of subfolder to save results in. "
        'You can put "/" at the end or at the beginning to indicate whether the subfolder should sit on top of '
        '{model_provider}/{model} or under (by default), e.g. "/abc" means "results/{model_provider}/{model}/abc". '
        'If you put "/" both at the beginning and at the end (i.e. "/abc/"), it will be "results/abc"',
    )
    parser.add_argument(
        "--task_and_context",
        type=str,
        default=None,
        help="File path to custom task and context model instructions in a txt file that go in the system prompt",
    )
    parser.add_argument(
        "--detailed_plans",
        action="store_true",
        help="Enable interactive planning mode",
    )

    parser.add_argument(
        "--sea",
        action="store_true",
        help="Enable deep planning mode",
    )
    return parser


async def run(
    args,
    predictor_class,
    tool_executor: Optional[ToolExecutor] = None,
    eval_root: Path = TOOLTALK_EVAL_ROOT,
):

    if tool_executor is None:
        tool_executor = ToolExecutor(
            init_database_dir=str(eval_root / "data" / "databases"),
            max_chain_of_actions_per_turn=(
                100 if args.force_single_hop else args.max_chain_of_actions_per_turn
            ),
        )

    results_root = eval_root / "results"
    if args.results_subfolder:
        if args.results_subfolder[0] == "/" and args.results_subfolder[-1] == "/":
            results_dir = results_root / args.results_subfolder[1:-1]
        elif args.results_subfolder[0] == "/":
            results_dir = results_root / args.model_provider / args.model / args.results_subfolder[1:]
        elif args.results_subfolder[-1] == "/":
            results_dir = results_root / args.results_subfolder[:-1] / args.model_provider / args.model
        else:
            results_dir = results_root / args.model_provider / args.model / args.results_subfolder
    else:
        results_dir = results_root / args.model_provider / args.model
    results_dir = str(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    logger.info("=== Evaluation Settings ===")
    logger.info(f"Model   : {args.model_provider} / {args.model}")
    logger.info(f"Results : {results_dir}")

    missed_testcases = set(args.testcases or [])
    for ds in args.dataset:
        if ds == "assistance_subset":
            eval_assistance_seeking_turn = True
        else:
            eval_assistance_seeking_turn = False

        if args.task_and_context:
            with open(args.task_and_context, "r", encoding="utf-8") as reader:
                task_and_context = reader.read()
            logger.info(
                f"Evaluate model with custom Task and Context model instructions: {task_and_context}"
            )
        else:
            task_and_context = None

        dataset_dir = str(eval_root / "data" / ds)
        logger.info(f"Dataset : {dataset_dir}")
        file_names_and_paths = get_names_and_paths(dataset_dir)
        if args.testcases:
            filtered_file_names_and_paths = []
            for testcase in args.testcases:
                for file_name, file_path in file_names_and_paths:
                    if testcase in file_name:
                        filtered_file_names_and_paths.append((file_name, file_path))
                        missed_testcases.discard(testcase)
            file_names_and_paths = filtered_file_names_and_paths

        total_metrics = Counter()
        conversation_breakdown = []
        for file_name, file_path in tqdm(file_names_and_paths):
            output_file_path = os.path.join(results_dir, file_name)
            if os.path.exists(output_file_path) and not args.reset:
                logger.info(f"Skipping {file_name} because it already exists")
                with open(output_file_path, "r", encoding="utf-8") as reader:
                    conversation_with_metrics = json.load(reader)
                total_metrics += conversation_with_metrics["metrics"]
                total_metrics["num_conversations"] += 1
                continue

            logger.info(f"Running {file_name}")
            with open(file_path, "r", encoding="utf-8") as reader:
                conversation = json.load(reader)

            if EvalModes.PREDICT in args.modes:
                logger.info("Running prediction...")
                if args.api_mode == "exact":
                    apis_used = [APIS_BY_NAME[api_name] for api_name in conversation["apis_used"]]
                elif args.api_mode == "suite":
                    apis_used = [
                        api for suite_name in conversation["suites_used"] for api in SUITES_BY_NAME[suite_name].apis
                    ]
                elif args.api_mode == "all":
                    apis_used = ALL_APIS
                else:
                    raise ValueError(f"Invalid api mode: {args.api_mode}")

                predictor_func = predictor_class(
                    model=args.model,
                    apis_used=apis_used,
                    disable_docs=args.disable_documentation,
                    debug=args.debug,
                    temperature=args.temperature,
                    is_detailed_plans=args.detailed_plans or args.sea,
                    is_sea=args.sea,
                )
                conversation = await tool_executor.run_conversation(
                    conversation,
                    predictor_func,
                    task_and_context=task_and_context,
                )

            if EvalModes.EVALUATE in args.modes:
                logger.info("Running evaluation...")
                if eval_assistance_seeking_turn:
                    conversation = tool_executor.evaluate_assistance_seeking(conversation)
                else:
                    conversation = tool_executor.evaluate_predictions(conversation)
                if eval_assistance_seeking_turn:
                    if conversation["metrics"]["f1_assistance"] == 1.0:
                        repr = "✅"
                    else:
                        repr = f"❌ ({conversation['metrics']['f1_assistance']:.2f})"
                else:
                    if conversation["metrics"]["success"]:
                        repr = "✅"
                    else:
                        repr = f"❌ ({conversation['metrics']['soft_success']:.2f})"
                logger.info(f"Conversation {file_name} pass: {repr}")
                assert file_name.endswith(".json")
                conversation_breakdown.append((file_name[:-5], repr))
                total_metrics += conversation["metrics"]
                total_metrics["num_conversations"] += 1

            if EvalModes.VALIDATE in args.modes:
                logger.info("Validating evaluation...")
                for turn in conversation["conversation"]:
                    if "predictions" not in turn:
                        continue
                    if eval_assistance_seeking_turn:
                        continue
                    for prediction in turn["predictions"]:
                        if prediction["role"] == "api":
                            assert "match" in prediction
                            assert "bad_action" in prediction

            with open(output_file_path, "w", encoding="utf-8") as writer:
                json.dump(conversation, writer, indent=4)

        logger.info("Finished processing conversations")
        if EvalModes.EVALUATE in args.modes:
            if total_metrics["num_conversations"] == 0:
                logger.info("No conversations found")
                continue
            metrics = {
                "num_conversations": total_metrics["num_conversations"],
                "precision": safely_divide(total_metrics["matches"], total_metrics["predictions"]),
                "recall": (
                    total_metrics["matches"] / total_metrics["ground_truths"] if total_metrics["ground_truths"] else 0
                ),
                "action_precision": safely_divide(total_metrics["valid_actions"], total_metrics["actions"]),
                "bad_action_rate": safely_divide(total_metrics["bad_actions"], total_metrics["actions"]),
                "success_rate": total_metrics["success"] / total_metrics["num_conversations"],
                "soft_success_rate": total_metrics["soft_success"] / total_metrics["num_conversations"],
                "api_name_precision": total_metrics["api_name_precision"] / total_metrics["num_conversations"],
                "api_name_recall": total_metrics["api_name_recall"] / total_metrics["num_conversations"],
                "api_name_jaccard": total_metrics["api_name_jaccard"] / total_metrics["num_conversations"],
            }
            if eval_assistance_seeking_turn:
                metrics["acc_asking_for_help_when_should"] = safely_divide(
                    total_metrics["acc_asking_for_help_when_should"], total_metrics["num_conversations"]
                )
                metrics["acc_asking_for_help_when_should_not"] = safely_divide(
                    total_metrics["acc_asking_for_help_when_should_not"], total_metrics["num_conversations"]
                )
                metrics["f1_assistance"] = safely_divide(total_metrics["f1_assistance"], total_metrics["num_conversations"])
            logger.info(f"Metrics: {json.dumps(metrics, indent=4)}")
            conversation_breakdown = sorted(conversation_breakdown, key=lambda x: x[0])
            print(
                "Result breakdown:\n"
                + tabulate(
                    conversation_breakdown,
                    headers=(
                        ["Testcase", "F1 Assistance"] if eval_assistance_seeking_turn else ["Testcase", "Soft Success"]
                    ),
                    tablefmt="github",
                )
            )
    if missed_testcases:
        logger.warning(f"Missed testcases: {missed_testcases}")


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, force=True)

    if args.model_provider == "openai":
        openai_key = os.environ.get("OPENAI_API_KEY", None)
        if openai_key is None:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        openai.api_key = openai_key
        predictor_class = partial(OpenAIPredictor, force_single_hop=args.force_single_hop)
    elif args.model_provider == "mistral":
        predictor_class = partial(MistralPredictor, force_single_hop=args.force_single_hop)
    elif args.model_provider == "anthropic":
        predictor_class = partial(
            AnthropicPredictor,
            route="anthropic",
            force_single_hop=args.force_single_hop,
        )
    elif args.model_provider == "bedrock":
        if args.model in ANTHROPIC_ENDPOINTS["bedrock"]:
            predictor_class = partial(
                AnthropicPredictor,
                route="bedrock",
                force_single_hop=args.force_single_hop,
            )
        else:
            raise NotImplementedError(f"Model {args.model} not supported by bedrock")
    else:
        assert args.model_provider == "cohere"
        predictor_class = partial(
            CoherePredictor,
            # user_followup=args.user_followup,
            force_single_hop=args.force_single_hop,
            skip_augmented_gen=args.skip_augmented_gen,
        )

    asyncio.run(run(args, predictor_class))


if __name__ == "__main__":
    main()
