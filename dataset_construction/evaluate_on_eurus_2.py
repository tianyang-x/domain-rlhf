import os
import json
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from util.evaluation_utils.code_util import evaluate_code
from util.evaluation_utils.math_util import evaluate_math
import argparse

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B", help="The model to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--k", type=int, default=8, help="Number of completions to generate for each prompt")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of parallel tensors to use for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for sampling")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to evaluate; if -1, use the entire shard")
    parser.add_argument("--output_folder", type=str, default=".", help="Base folder to save evaluation results")
    # NOTE: token consuming
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of tokens to generate.")
    parser.add_argument("--log_per_step", type=int, default=1000, help="Log results every N samples")
    # NOTE: new features: resuming, shard splitting
    parser.add_argument("--resume", action="store_true", help="Resume evaluation from existing output files if available")
    parser.add_argument("--num_shards", type=int, default=3, help="Total number of shards to split the dataset")
    parser.add_argument("--shard_index", type=int, default=0, help="Index of the shard to process (0-indexed)")
    return parser.parse_args()

def generate_batch_completions(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    all_completions = []
    for output in outputs:
        completions = [output.outputs[i].text for i in range(len(output.outputs))]
        all_completions.append(completions)
    return all_completions

def process_dialogue(dialogue: dict, tokenizer, start_header_text, end_header_text, eot_text):
    prompt_template = ""
    if tokenizer.bos_token_id is not None:
        prompt_template += f"{tokenizer.decode([tokenizer.bos_token_id])}"

    prompts = dialogue
    if prompts[-1]["role"] == "assistant":
        prompts = prompts[:-1]
    for message in prompts:
        prompt_template += f"{start_header_text}{message['role']}{end_header_text}\n{message['content']}{eot_text}\n"
    # append bot token
    prompt_template += f"{start_header_text}assistant{end_header_text}\n"

    return prompt_template

def evaluate(k_completions, ability, ground_truth):
    all_results = []
    for completion in k_completions:
        if ability == "math":
            is_correct, _, _ = evaluate_math(completion, ground_truth)
            all_results.append(1 if is_correct else 0)
        elif ability == "coding":
            success, _ = evaluate_code(completion, ground_truth)
            all_results.append(1 if success else 0)
    return all_results

def main():
    args = read_args()
    ds = load_dataset("PRIME-RL/Eurus-2-RL-Data")
    
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, n=args.k, max_tokens=args.max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    for split in ["train", "validation"]:
        print(f"Running evaluation on {split} split")
        dataset_full = ds[split]
        total_samples = len(dataset_full)
        shard_start = (args.shard_index * total_samples) // args.num_shards
        shard_end = ((args.shard_index + 1) * total_samples) // args.num_shards
        split_ds = dataset_full.select(range(shard_start, shard_end))
        
        if args.num_samples > 0:
            num_samples = min(args.num_samples, len(split_ds))
        else:
            num_samples = len(split_ds)
        
        result_file = f"evaluation_results_{split}_shard{args.shard_index}_of_{args.num_shards}.json"
        result_file = os.path.join(args.output_folder, result_file)
        complete_result_file = f"complete_evaluation_results_{split}_shard{args.shard_index}_of_{args.num_shards}.json"
        complete_result_file = os.path.join(args.output_folder, complete_result_file)
        
        if args.resume and os.path.exists(result_file):
            with open(result_file, "r", encoding="utf-8") as f:
                results_split = json.load(f)
            if os.path.exists(complete_result_file):
                with open(complete_result_file, "r", encoding="utf-8") as f:
                    complete_results_split = json.load(f)
            else:
                complete_results_split = []
            start_index = len(results_split)
            print(f"Resuming {split} split from sample index {start_index}")
        else:
            results_split = []
            complete_results_split = []
            start_index = 0
        
        sample_counter = start_index
        for i in tqdm(range(start_index, num_samples, args.batch_size), desc=f"Processing {split} split"):
            batch_samples = split_ds[i: i + args.batch_size]
            prompts = []
            for sample in batch_samples['prompt']:
                message = process_dialogue(sample, tokenizer, start_header_text="<|start_header_id|>", end_header_text="<|end_header_id|>", eot_text="<|eot_id|>")
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
                sample_dict = {
                    "data_source": data_source,
                    "prompt": prompt,
                    "ability": ability,
                    "reward_model": reward_model,
                    "extra_info": extra_info,
                }
                ground_truth = sample_dict.get("reward_model", {}).get("ground_truth", "")
                correctness = evaluate(completions, ability, ground_truth)
                sample_result = {**sample_dict, "correctness": correctness}
                results_split.append(sample_result)
                complete_sample_result = sample_result.copy()
                complete_sample_result["completions"] = completions
                complete_results_split.append(complete_sample_result)
                sample_counter += 1
            
            if (i - start_index) % args.log_per_step == 0:
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(results_split, f, ensure_ascii=False, indent=2)
                with open(complete_result_file, "w", encoding="utf-8") as f:
                    json.dump(complete_results_split, f, ensure_ascii=False, indent=2)
        
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results_split, f, ensure_ascii=False, indent=2)
        with open(complete_result_file, "w", encoding="utf-8") as f:
            json.dump(complete_results_split, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()