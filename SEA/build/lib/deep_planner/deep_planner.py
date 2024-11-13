import json
from deep_planner import Evaluator, Compiler

from typing import List, Optional

class DeepPlanner():
    def __init__(
            self, 
            n_plans: int, 
            temperature: float | List[float], 
            plan_generator_func: callable, 
            eval_generate_func: callable,
            compile_generate_func: callable,
            extract_plan_func: Optional[callable],
            is_split_feedback: Optional[bool] = True,
            is_printing: Optional[bool] = False
        ):
        self.n_plans = n_plans
        self.temperature = temperature
        self.generate_plan = plan_generator_func
        self.extract_plan = extract_plan_func
        self.evaluator = Evaluator(n_plans, eval_generate_func, is_split_feedback)
        self.compiler = Compiler(n_plans, compile_generate_func)
        self.is_printing = is_printing

        if isinstance(temperature, list):
            assert len(temperature) == n_plans, "Number of temperatures must match number of plans."


    def print(self, *args, **kwargs):
        if self.is_printing:
            print(*args, **kwargs)

    def sample_plans(self, prompt: Optional[str], messages: Optional[List[dict]]) -> List[str]:
        plan_options = []

        # Sample multiple unique plans
        temperatures = self.temperature if isinstance(self.temperature, list) else [self.temperature for _ in range(self.n_plans)]
        
        while len(plan_options) < self.n_plans:
            plan_temp = temperatures[len(plan_options)]
            
            if messages:
                self.print(f"Generating plans at temperature={self.temperature} from message...")
                plan = self.generate_plan(messages=messages, temperature=plan_temp)
            else:
                self.print(f"Generating plans at temperature={self.temperature} from prompt...")
                plan = self.generate_plan(prompt=prompt, temperature=plan_temp)

            if plan.strip() != "":
                if self.extract_plan:
                    plan = self.extract_plan(plan)

                if plan not in plan_options:
                    plan_options.append(plan)

        return plan_options


    def plan(
            self, 
            previous_conversation_str: str, 
            tools_string: str,
            planning_prompt: Optional[str] = None,  
            planning_messages: Optional[List[dict]] = None
        ) -> str:
        assert planning_prompt is not None or planning_messages is not None, "Either planning_prompt or planning_messages must be provided."
        assert planning_prompt is None or planning_messages is None, "Only one of planning_prompt or planning_messages should be provided."

        plans = self.sample_plans(prompt=planning_prompt, messages=planning_messages)

        self.print("Plans: ", plans)
        if len(plans) < 1:
            raise ValueError("No plans generated.")
        
        self.print("Evaluating plans...")
        feedback = self.evaluator.evaluate(plans, previous_conversation_str, tools_string)
        self.print("Plans and feedback: ", json.dumps(feedback))

        self.print("Compiling final plan...")
        final_plan = self.extract_plan(self.compiler.compile(feedback, previous_conversation_str, tools_string))
        self.print("Compiled plan: ", final_plan)

        return final_plan
    