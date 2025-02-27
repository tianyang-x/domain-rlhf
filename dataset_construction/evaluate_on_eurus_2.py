import json
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from evaluation_utils.code_util import evaluate_code
from evaluation_utils.math_util import evaluate_math
import argparse
from tqdm import tqdm

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="The model to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--k", type=int, default=10, help="Number of completions to generate for each prompt")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of parallel tensors to use for generation")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for sampling")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to evaluate")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="File to save evaluation results")
    # NOTE: The dataset asks for CoT output. This is very token-consuming. If you limit it to a small number, the model will not be able to generate the answer.
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate.")
    return parser.parse_args()

def generate_batch_completions(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    all_completions = []
    for output in outputs:
        completions = [output.outputs[i].text for i in range(len(output.outputs))]
        all_completions.append(completions)
    return all_completions

def evaluate(k_completions, ability, ground_truth):
    all_results = []
    for completion in k_completions:
        if ability == "math":
            is_correct, _, _ = evaluate_math(completion, ground_truth)
            if is_correct:
                all_results.append(1)
            else:
                all_results.append(0)
        elif ability == "coding":
            success, _ = evaluate_code(completion, ground_truth)
            if success:
                all_results.append(1)
            else:
                all_results.append(0)
    return all_results

def main():
    args = read_args()
    ds = load_dataset("PRIME-RL/Eurus-2-RL-Data")
    
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    k = args.k
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, n=k, max_tokens=4096)
    
    batch_size = args.batch_size
    results = {}
    complete_results = {}
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    num = 0
    for split in ["train", "validation"]:
        results[split] = []
        complete_results[split] = []
        print(f"Running evaluation on {split} split")
        num_samples = args.num_samples if args.num_samples > 0 else len(ds[split])
        for i in tqdm(range(0, num_samples, batch_size)):
            batch_samples = ds[split][i: i + batch_size]
            prompts = []
            for sample in batch_samples['prompt']:
                message = tokenizer.apply_chat_template(
                    sample,
                    tokenize=False,
                    add_generation_prompt=False
                )
                prompts.append(message)
            
            batch_outputs = generate_batch_completions(llm, prompts, sampling_params)
            
            for data_source, prompt, ability, reward_model, extra_info, completions in zip(
                batch_samples["data_source"],
                batch_samples["prompt"],
                batch_samples["ability"],
                batch_samples["reward_model"],
                batch_samples["extra_info"],
                batch_outputs
            ):
                sample = {
                    "data_source": data_source,
                    "prompt": prompt,
                    "ability": ability,
                    "reward_model": reward_model,
                    "extra_info": extra_info,
                }
                
                ability = sample.get("ability", "")
                ground_truth = sample.get("reward_model", {}).get("ground_truth", "")
                correctness = evaluate(completions, ability, ground_truth)
                sample_result = {
                    **sample,
                    "correctness": correctness,
                }
                results[split].append(sample_result)
                complete_sample_result = sample_result.copy()
                complete_sample_result["completions"] = completions
                complete_results[split].append(complete_sample_result)
                
                num += 1
                if args.num_samples > 0 and num >= args.num_samples:
                    break
            if args.num_samples > 0 and num >= args.num_samples:
                break
        if args.num_samples > 0 and num >= args.num_samples:
            break
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    with open("complete_" + args.output_file, "w", encoding="utf-8") as f:
        json.dump(complete_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()