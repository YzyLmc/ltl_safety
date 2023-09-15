# Safety Chip
Codebase for paper: *Plug in the Safety Chip: Enforcing Temporal Constraints for LLM Agents*
## Installation
1. Download [Virtual Home](https://github.com/xavierpuigf/virtualhome) simulator, and put them under `virtualhome_v2.3.0/simulation`.
2. Creating the environment by:
```
conda env create -f environment.yml
```
## Prepare for Experiments
Set environment variable for OpenAI API key for current shell
```
export OPENAI_API_KEY=<YOUR_API_KEY>
```
## Files
- `src`: all scripts for the project are stored here
    * `exp_virtualhome_manip.py`: script for virtualhome experiments
    * `exp_spot.py`: script for robot demo. Doesn't require virtualhome simulator installed
    * `constraint_module.py`: currently only used for translating natural language into LTL formulas
    * `lang2ltl.py`: library for enhanced version of lang2ltl framework. Can translate predifined predicates
    * `precond.py`: precondition module used for robot demo on baseline model and safety chip.
    * `get_embed.py`: encode action list or object list for grounding
    * `utils.py`
- `results`: experimental results. Two runs for robot demo. Two sets of results for virtualhome experiments: one with non-expert provided utterances, the other with expert provided utterances.
- `virtualhome_v2.3.0/dataset`: datasets for virtualhome exp and robot demo
- `robot`: nav graph scanned by Boston Dynamics Spot robot

## Running Experiments
1. for virtualhome experiments, install virtaulhome simulator, export your `OPENAI_API_KEY`. Then you should be able to run exps by (an example)
```
python src/exp_virtualhome_manip.py --exp rooms --example_fname 0_1 --constraint_num 5 --safety_level full
```
If you place the embedding files somewhere, don't forget to specify them using the flags.

2. Similarly, you can also generate plans for robot demo by:
```
python src/exp_spot.py --example_fname task1 --constraint_num 10 --safety_level full
```
