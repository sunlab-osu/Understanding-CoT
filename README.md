# Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters

Code, model input/output and cached evaluation results for our [ACL-23](https://2023.aclweb.org/) paper ["Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters"](https://arxiv.org/abs/2212.10001) by [Boshi Wang](https://boshi-wang.github.io/), [Sewon Min](https://shmsw25.github.io/), [Xiang Deng](https://xiang-deng.github.io/), [Jiaming Shen](https://mickeysjm.github.io/), [You Wu](https://scholar.google.com/citations?user=i8TKyfIAAAAJ&hl=en), [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz) and [Huan Sun](http://web.cse.ohio-state.edu/~sun.397/).


## Overview

While [Chain-of-Thought (CoT) prompting](https://arxiv.org/abs/2201.11903) can improve reasoning in large LMs, there is little understanding of what makes it effective. We perform a series of ablation studies on two representive benchmarks where CoT brings large improvements, which reveal the impact of different aspects of CoT demonstrations. We find that 
- CoT reasoning is possible with invalid demonstrations - prompting with invalid reasoning steps can achieve over 80-90% of the performance obtained using CoT under various metrics, while still generating coherent lines of reasoning during inference.
- Other aspects of the rationales, such as being relevant to the query and correctly ordering the reasoning steps, are much more important for effective CoT reasoning.

Overall, these findings open up new questions regarding LLMs' capability to learn to reason in context, and reflections on benchmarking few-shot reasoning.


## Citation

If you find our code or paper useful, please cite the paper:
```
@inproceedings{wang2023towards,
  title={Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters},
  author={Wang, Boshi and Min, Sewon and Deng, Xiang and Shen, Jiaming and Wu, You and Zettlemoyer, Luke and Sun, Huan},
  booktitle={The 61st Annual Meeting of the Association for Computational Linguistics},
  year={2023}
}
```

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
