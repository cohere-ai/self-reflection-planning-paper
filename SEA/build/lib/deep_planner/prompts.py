COMPILER_PROMPT = lambda n_plans:  "".join((
    "You are tasked with compiling a final plan to help an AI assistant solve a user query.",
    f"\nTo do this, you are provided with {n_plans} plan options, each of which is accompanied by a quality evaluation in the form of feedback." if n_plans > 1 else "\nTo do this, you are provided with an original plan, accompanied by a quality evaluation in the form of feedback.",
    "\nAdditionally, you are provided with a list of available tools specifications and the previous conversation between user and assistant which should be used to inform the final plan.",
    "\nYou should consider the feedback provided for each plan and either select or compile a final plan that is most likely to resolve the user's query." if n_plans > 1 else "\nYou should consider the feedback provided for the original plan and compile a final plan that is most likely to resolve the user's query.",
    "\nThe final plan should be clear and concise, with each step using exactly one tool that is necessary for solving the user query without any additional input required the user. Each step must also have a variable #V indicating the tool (from only those available) to be used and the parameters to be passed to the tool.",
    "\nDo not make use of tools that are not available. The final plan should only include interaction with the user if it is necessary for obtaining tool parameter values. If user is required for the parameters of the first tool to be used, the entire plan should be to ask the user for the necessary information.",
    "\nRespond with only the final plan.",
    "\n\nAvailable tools:",
    "\n{tools_str}"
    "\n\nPrevious conversation:",
    "\n{previous_conversation}",
    "\n\nPlans and feedback:" if n_plans > 1 else "Plan and feedback:",
    "\n{plans_and_feedback}",
    "\n\nFinal plan:\n",
))


EVAL_PROMPT_BOTH = lambda n_plans: "".join((
    "You are tasked evaluating a plan to help an AI assistant solve a user query.",
    "\nPlans may be given in structured step-by-step format, with each step being accompanied by a variable #V indicating the tool to be used and the parameters to be passed to the tool.",
    "\nYou should provide feedback on: 1) the efficacy of the plan in solving the user query (without any addition user input), 2) the availability of the tools selected, and 3) and the correctness of tool parameters specified.",
    "\nTo help you, you are provided with a list of available tools (and their specifications)" + (f", {n_plans-1} alternative plans for for comparison," if n_plans > 1 else "") + " and the previous conversation between the user and the AI assistant.",
    "\nRespond with only the plan feedback.",
    "\n\nAvailable tools:",
    "\n{tools_str}"
    "\n\nPrevious conversation:",
    "\n{previous_conversation}",
    "\n\nAlternative plans:" if n_plans > 1 else "",
    "\n{alternative_plans}" if n_plans > 1 else "",
    "\n\nPlan to evaluate:",
    "\n{plan}"
    "\n\nFeedback:\n",
))


EVAL_PROMPT_TOOLS = lambda n_plans: "".join((
    "You are tasked evaluating a plan to help an AI assistant solve a user query.",
    "\nPlans may be given in structured step-by-step format, with each step being accompanied by a variable #V indicating the tool to be used and the parameters to be passed to the tool.",
    "\nYou should provide feedback both on the availability of the tools selected and the correctness of parameters specified.",
    "\nTo help you, you are provided with a list of available tools (and their specifications)" + (f", {n_plans-1} alternative plans for for comparison," if n_plans > 1 else "") + " and the previous conversation between the user and the AI assistant.",
    "\nSpecifically, you should check that the tools used are available and that the parameters are correctly specified (i.e., both in the correct format and derived directly from the previous conversation without the use of placeholders or requiring any additional user input).",
    "\nRespond with only the plan feedback.",
    "\n\nAvailable tools:",
    "\n{tools_str}"
    "\n\nPrevious conversation:",
    "\n{previous_conversation}",
    "\n\nAlternative plans:" if n_plans > 1 else "",
    "\n{alternative_plans}" if n_plans > 1 else "",
    "\n\nPlan to evaluate:",
    "\n{plan}"
    "\n\nFeedback:\n",
))


EVAL_PROMPT_ADEQUACY = lambda n_plans: "".join((
    "You are tasked evaluating a plan to help an AI assistant solve a user query.",
    "\nPlans may be given in structured step-by-step format, with each step being accompanied by a variable #V indicating the tool to be used and the parameters to be passed to the tool.",
    "\nYou should provide feedback on the efficacy of the plan in solving the user query.",
    "\nTo help you, you are provided with a list of available tools (and their specifications)" + (f", {n_plans-1} alternative plans for for comparison," if n_plans > 1 else "") + " and the previous conversation between the user and the AI assistant.",
    "\nSpecifically, you should check that each step is necessary and makes use of the most suitable tools for solving the user query without any additional input required the user."
    "\nRespond with only the plan feedback.",
    "\n\nAvailable tools:",
    "\n{tools_str}"
    "\n\nPrevious conversation:",
    "\n{previous_conversation}",
    "\n\nAlternative plans:" if n_plans > 1 else "",
    "\n{alternative_plans}" if n_plans > 1 else "",
    "\n\nPlan to evaluate:",
    "\n{plan}"
    "\n\nFeedback:\n",
))