import string
import json
import argparse
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
class GPTTokenizer:
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2", max_length=1e5)

    def tokenize(self, s):
        tokens = self.gpt_tokenizer.tokenize(s)
        tokens = [t.lstrip("Ġ") for t in tokens]
        return tokens

xlingual_tokenizer = GPTTokenizer()
xlingual_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer)

def normalize_answer(s):
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_punc(lower(s)))

def exact_match(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def rouge(prediction, ground_truth):
    scores = xlingual_rouge_scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)

def compute_metrics(predictions, references):
    assert len(predictions) == len(references)
    em = sum(metric_max_over_ground_truths(exact_match, pred, gold) 
             for pred, gold in zip(predictions, references))
    rougeL = sum(metric_max_over_ground_truths(rouge, pred, gold) 
                for pred, gold in zip(predictions, references))
    return {
        "exact_match": round(100 * em / len(references), 4),
        "rougeL": round(100 * rougeL / len(references), 4)
    }

def compute_grouped_metrics(predictions, references, groups):
    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        examples_by_group.setdefault(group, []).append((pred, gold))
    return {
        f"{metric}_{group}": value
        for group in examples_by_group
        for metric, value in compute_metrics(*zip(*examples_by_group[group])).items()
    }

def parse_args():
    parser = argparse.ArgumentParser()
    file = "ponss_k_32"
    parser.add_argument("--model_path", default=f"./output/{file}_meta-llama_Llama-3.2-3B-Instruct")
    parser.add_argument("--test_prompts_file", default="filtered_test_prompts.jsonl")
    parser.add_argument("--reference_file", default="filtered_test_references.jsonl")
    parser.add_argument("--prediction_file", default=f"predictions_{file}.jsonl")
    parser.add_argument("--output_file", default=f"results_{file}.json")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()

def generate_predictions(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Count lines for progress bar
    with open(args.test_prompts_file) as f:
        total_prompts = sum(1 for _ in f)

    with open(args.prediction_file, "w") as fout, open(args.test_prompts_file) as fin:
        for line in tqdm(fin, total=total_prompts, desc="Generating predictions", unit="prompt"):
            prompt_data = json.loads(line)
            generated = generator(
                prompt_data["prompt"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )[0]['generated_text']
            
            prediction = generated.split("output : ")[-1].strip().split('\n')[0]
            prediction = prediction.replace('"', '').replace("'", "").strip()
            
            fout.write(json.dumps({
                "id": prompt_data["id"],
                "prediction": prediction
            }) + "\n")

def main():
    args = parse_args()
    CUDA_VISIBLE_DEVICES=1
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("Generating predictions...")
    generate_predictions(args)

    print("\nEvaluating results...")
    eval_instances = {}
    with open(args.reference_file) as fin:
        for line in tqdm(fin, desc="Loading references", unit="ref"):
            instance = json.loads(line)
            eval_instances[instance["id"]] = instance

    all_predictions = {}
    with open(args.prediction_file) as fin:
        for line in tqdm(fin, desc="Loading predictions", unit="pred"):
            pred = json.loads(line)
            all_predictions[pred["id"]] = pred["prediction"]

    all_results = {}
    print("\nEvaluating xlingual track:")
    instance_ids = [id for id, inst in eval_instances.items() if inst.get("track", "xlingual") == "xlingual"]
    
    references = []
    predictions = []
    for id in tqdm(instance_ids, desc="Processing instances", unit="instance"):
        references.append(eval_instances[id]["references"])
        predictions.append(all_predictions.get(id, ""))

    if missing := len([p for p in predictions if not p]):
        print(f"Warning: {missing} empty predictions found")

    print("\nCalculating metrics:")
    metrics = compute_metrics(predictions, references)
    print(f"\nExact Match: {metrics['exact_match']}%")
    print(f"ROUGE-L: {metrics['rougeL']}%")
    all_results.update(metrics)

    if instance_ids and "task_category" in eval_instances[instance_ids[0]]:
        categories = [eval_instances[id]["task_category"].lower().replace(" ", "_") 
                     for id in instance_ids]
        grouped = compute_grouped_metrics(predictions, references, categories)
        all_results.update(grouped)

    if args.output_file:
        with open(args.output_file, "w") as fout:
            json.dump(all_results, fout, indent=2)
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()
