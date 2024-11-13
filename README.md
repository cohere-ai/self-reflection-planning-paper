# On the Efficacy of Self-reflection for Improving LLM Agent Planning

This repository contains the code for our paper: "On the Efficacy of Self-reflection for Improving LLM Agent Planning".

## Navigation
The root of the repository contains the following sub-repositories:
- `SEA/`: Contains the code for our self-relection framework, SEA (Sample-Evaluate-Aggregate).
- `ToolTalk/`: Contains our version of the [ToolTalk benchmark repository](https://github.com/microsoft/ToolTalk).
- `ToolSandbox/`: Contains our version of the [ToolSandbox benchmark repository](https://github.com/apple/ToolSandbox).


## Experiments

The `ToolTalk/` and `ToolSandbox/` sub-repositories contain the code necessary to run our experiments on each  benchmark, with instructions provided in the top the respective READMEs. Note that the different environments will likely be required to run each benchmark - after creating a new python environment, you will have to: 1) follow the benchmark installation instructions described in its README, and 2) locally install the SEA repository as a package by running `pip install .` within the `SEA/` subrepository.
The respective experimental results for each benchmark can be found in the `results` folder in the root of each sub-repository.

## Citation
```
```