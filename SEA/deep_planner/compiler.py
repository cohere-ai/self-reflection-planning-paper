
from .prompts import COMPILER_PROMPT


class Compiler():
    def __init__(self, n_plans: int, generate_func: callable):
        self.n_plans = n_plans
        self.generate = generate_func

        self.prompt = COMPILER_PROMPT(n_plans=n_plans)

    def compile(self, plan_feedback: dict, previous_convervation_str: str, tools_string: str) -> str:
        prompt = self.prompt.format(
            tools_str=tools_string,
            plans_and_feedback="\n\n".join([f"{plan}\nFeedback: {feedback}" for plan, feedback in plan_feedback.items()]),
            previous_conversation=previous_convervation_str
        )
        return self.generate(prompt, 0)