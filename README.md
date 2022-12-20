# Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters

Code for the paper "Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters" by [Boshi Wang](https://boshi-wang.github.io/), [Sewon Min](https://shmsw25.github.io/), [Xiang Deng](https://xiang-deng.github.io/), [Jiaming Shen](https://mickeysjm.github.io/), [You Wu](https://scholar.google.com/citations?user=i8TKyfIAAAAJ&hl=en), [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz) and [Huan Sun](http://web.cse.ohio-state.edu/~sun.397/).


## Repo Tour
    .
    ├── grade-school-math/                       # GSM8K dataset, from https://github.com/openai/grade-school-math
    ├── indices_800.json                         # Indices for the 800 GSM8K test examples used for evaluation 
    ├── Bamboogle Prerelease - Sheet1.csv        # Bamboogle dataset, from https://github.com/ofirpress/self-ask
    ├── Bamboogle Prerelease - Sheet1_inter.csv  # Annotated intermediate bridging entities for Bamboogle
    ├── utils.py                                 # Helper functions
    ├── prompts_*/                               # Full prompts for all settings in our experiments
    ├── main_*.py                                # Scripts for getting model predictions via OpenAI API
    ├── eval_*.ipynb                             # Evaluation scripts, including cached evaluation results
    └── result_*/                                # Cached model prediction results 

## Usage
First put your OpenAI API key in a file named ```api_key.txt```.

### Run LLM generation
Details could be found in the param descriptions in ```main_*.py```. For example, to run the invalid reasoning setting on GSM8K and Bamboogle:
```bash
python main_gsm8k.py --prompt_dir prompts_arithmetic/invalid_reasoning.txt --eng text-davinci-002 --num_test 800 --seed 1357 --temp 0.0 --test_ind indices_800.json
```
```bash
python main_bamboogle.py --prompt_dir prompts_bamboogle/invalid_reasoning.txt --eng text-davinci-002 --num_test -1 --seed 1357 --temp 0.0
```

### Evaluation
```eval_*.ipynb``` contains the scripts and cached evaluation results.
