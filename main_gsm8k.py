import os
import openai
import json
import jsonlines
import re
import numpy as np
from utils import read_jsonl, extract_answer
import argparse

def get_answer_from_gpt(prompt, question, eng='text-davinci-002', max_tokens=256, temperature=0.0):
    response = openai.Completion.create(
        engine=eng,
        prompt=prompt + "\n\nQ: {}\nA:".format(question),
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["Q:"]
    )
    return response['choices'][0]['text'].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dir", default=None, type=str, required=True, help="directory to prompt file (.txt)")
    parser.add_argument("--eng", default=None, type=str, required=True, help="engine")
    parser.add_argument("--num_test", default=400, type=int, help="number of samples tested. -1 if on all test samples")
    parser.add_argument("--seed", default=1357, type=int, help="random seed")
    parser.add_argument("--temp", default=0.0, type=float, help="temperature for generation")
    parser.add_argument("--max_tokens", default=256, type=int, help="max # of tokens for generation")
    parser.add_argument("--test_ind", default=None, type=str, help="dir to test indices. If not provided, randomly choose.")
    parser.add_argument("--suffix", default="", type=str, help="")

    args = parser.parse_args()
    print(args)

    # load prompts
    file = args.prompt_dir
    assert file.endswith(".txt")
    prompt_name = os.path.basename(file)[:-4]
    print(file, prompt_name)
    with open(file, "r", encoding='utf-8') as f:
        prompt = f.read().strip()

    test_data = read_jsonl("grade-school-math/grade_school_math/data/test.jsonl")
    qa_pairs = [(instance['question'], extract_answer(instance['answer'])) for instance in test_data]
    print("loading dataset complete. altogether", len(qa_pairs), "questions")

    # scale down. -1 if not.
    NUM_TEST = args.num_test
    if NUM_TEST == -1:
        qa_pairs_test = qa_pairs
    else:
        if args.test_ind is None:
            np.random.seed(args.seed)
            rand_indices = np.random.choice(len(qa_pairs), NUM_TEST, replace=False)
            qa_pairs_test = [qa_pairs[i] for i in rand_indices]
        else:
            with open(args.test_ind, "r") as f:
                test_ind = json.load(f)
            assert len(test_ind) == NUM_TEST
            qa_pairs_test = [qa_pairs[i] for i in test_ind]

    print("testing on", len(qa_pairs_test), "samples")

    with open("api_key.txt", "r") as f:
        openai.api_key = f.read().strip()

    file_name = "result_gsm8k/gsm8k_{}.eng{}.sample{}.seed{}.temp{}.{}.jsonl".format(
        prompt_name, args.eng, NUM_TEST, args.seed, args.temp, args.suffix)
    writer = jsonlines.open(file_name, mode='w')

    count = 0
    for (question, answer) in qa_pairs_test:
        count += 1
        print("currently", prompt_name, "#", count)
        result = dict()
        result['question'] = question
        result['answer'] = answer
        max_tokens = args.max_tokens
        if prompt_name == 'standard':
            max_tokens = 15
        result['ans_'+prompt_name] = get_answer_from_gpt(prompt, question,
                                eng=args.eng, max_tokens=max_tokens, temperature=args.temp)
        writer.write(result)

    # the last element is the prompt
    writer.write(prompt)
    writer.close()


if __name__ == '__main__':
    main()
