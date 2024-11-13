from typing import List
from .prompts import EVAL_PROMPT_TOOLS, EVAL_PROMPT_ADEQUACY, EVAL_PROMPT_BOTH

class Evaluator():
    def __init__(self, 
                 n_plans: int,
                 generate_func: callable,
                 is_split_feedback: bool = True):
        self.generate = generate_func
        self.is_split_feedback = is_split_feedback
        self.prompt_all = EVAL_PROMPT_BOTH(n_plans)
        self.prompt_tools = EVAL_PROMPT_TOOLS(n_plans)
        self.prompt_adequacy = EVAL_PROMPT_ADEQUACY(n_plans) 


    def evaluate(
            self, 
            plan_options: List[str], 
            previous_convervation_str: str, 
            tools_string: str
        ) -> str:
        feedback = {}
        plan_options = [f"Plan {i+1}: " + plan.strip() for i, plan in enumerate(plan_options)]
        for i, plan in enumerate(plan_options):
            other_plans = plan_options[:i] + plan_options[i+1:]
            other_plans = "\n\n".join(other_plans)

            if not self.is_split_feedback:
                prompt = self.prompt.format(
                    tools_str=self.tools_string,
                    previous_conversation=previous_convervation_str,
                    alternative_plans=other_plans,
                    plan=plan
                )
                feedback[plan] = self.generate(prompt, 0.0)
            else:
                prompt_tools = self.prompt_tools.format(
                    tools_str=tools_string,
                    previous_conversation=previous_convervation_str,
                    alternative_plans=other_plans,
                    plan=plan
                )
                prompt_adequacy = self.prompt_adequacy.format(
                    tools_str=tools_string,
                    previous_conversation=previous_convervation_str,
                    alternative_plans=other_plans,
                    plan=plan
                )
                feedback[plan] = f"{self.generate(prompt_tools, 0.0)}\n{self.generate(prompt_adequacy, 0.0)}"
            
        return feedback

