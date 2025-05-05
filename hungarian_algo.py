import json
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

def load_train_tasks(splits_dir):
    """Load task names from train_tasks.txt"""
    train_tasks_path = os.path.join(splits_dir, "train_tasks.txt")
    with open(train_tasks_path, "r", encoding="utf-8") as f:
        return [line.strip().split('_')[0] for line in f if line.strip()]

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def train_prompt(E_definition, positive_examples, negative_examples, instance):
    prompt = f'''Definition : {E_definition}
    
    Positive Example 1—
    input : {positive_examples[0]["input"]}
    output : {positive_examples[0]["output"]}
    explanation : {positive_examples[0]["explanation"]}

    Negative Example 1—
    input : {negative_examples[0]["input"]}
    output : {negative_examples[0]["output"]}
    explanation : {negative_examples[0]["explanation"]}

    Now complete the following example—
    input : {instance["input"]}
    output : '''
    return prompt

def hungarian_matching(dataset_path, num_samples, model, splits_dir, max_instances_per_task=5000):
    """Perform Hungarian algorithm matching with global optimization"""
    tasks = []
    all_instances = []
    
    # Load training tasks
    print("splits_dir :- ",splits_dir)
    train_tasks = load_train_tasks(splits_dir)
    print(train_tasks)
    print(f"Processing {len(train_tasks)} training tasks (max {max_instances_per_task} instances per task)")
    
    # First pass: collect all tasks and instances
    for task_file in tqdm(os.listdir(dataset_path), desc="Processing tasks"):
        tasks = []
        all_instances = []
        if not task_file.endswith(".json"):
            continue
        if not any(task_file.startswith(task_id) for task_id in train_tasks):
            continue
            
        task_path = os.path.join(dataset_path, task_file)
        task_data = load_json(task_path)
        print(task_data)
        exit()
        items = task_data if isinstance(task_data, list) else [task_data]
        
        for item in items:
            E_definition = item["Definition"][0]
            tasks.append({
                "task_id": task_file,
                "definition": E_definition,
                "pos_examples": item["Positive Examples"],
                "neg_examples": item["Negative Examples"]
            })
            
            # Collect instances into global pool
            instances = item["Instances"][:max_instances_per_task]
            for instance in instances:
                if isinstance(instance["output"], list):
                    instance["output"] = instance["output"][0]
                if "input" not in instance or "output" not in instance:
                    continue  # Skip invalid instances
                all_instances.append(instance)
    
        # Precompute embeddings for all instances and tasks
        print("Encoding instances...")
        instance_embeddings = []
        for instance in tqdm(all_instances, desc="Instances"):
            text = instance["input"] + instance["output"]
            emb = model.encode(text, convert_to_tensor=True).cpu().numpy()
            instance_embeddings.append(emb)
        
        print("Encoding tasks...")
        task_embeddings = []
        for task in tqdm(tasks, desc="Tasks"):
            emb = model.encode(task["definition"], convert_to_tensor=True).cpu().numpy()
            task_embeddings.append(emb)
        
        # Build global utility matrix
        print("Building utility matrix...")
        num_tasks = len(tasks)
        num_instances = len(all_instances)
        utility_matrix = np.zeros((num_tasks, num_instances))
        
        for i in range(num_tasks):
            task_emb = task_embeddings[i]
            for j in range(num_instances):
                instance_emb = instance_embeddings[j]
                similarity = np.dot(task_emb, instance_emb) / (
                    np.linalg.norm(task_emb) * np.linalg.norm(instance_emb) + 1e-8
                )
                utility_matrix[i, j] = similarity
        
        # Build cost matrix for Hungarian algorithm (max weight → min cost)
        cost_matrix = np.zeros((num_tasks * num_samples, num_instances))
        for i in range(num_tasks):
            for k in range(num_samples):
                row = i * num_samples + k
                cost_matrix[row, :] = -utility_matrix[i, :]  # Negate for minimization
        
        # Run Hungarian algorithm
        print("Running Hungarian assignment...")
        task_rows, instance_cols = linear_sum_assignment(cost_matrix)
        
        # Collect valid assignments
        output_list = []
        used_instances = set()
        for row, col in zip(task_rows, instance_cols):
            task_idx = row // num_samples
            instance_idx = col
            if instance_idx in used_instances:
                continue  # Ensure no duplicates
            used_instances.add(instance_idx)
            
            task = tasks[task_idx]
            instance = all_instances[instance_idx]
            score = utility_matrix[task_idx, instance_idx]
            
            output_list.append({
                "task_id": task["task_id"],
                "id": str(instance.get("id", "unknown")),
                "input_prompt": train_prompt(
                    task["definition"],
                    task["pos_examples"],
                    task["neg_examples"],
                    instance
                ),
                "output": str(instance["output"]),
                "score": float(score)
            })
        
    return output_list

def training_subset(dataset_path, num_samples, model_name, train_dataset_path, splits_dir):
    """Generate training subset using Hungarian algorithm"""
    model = SentenceTransformer(model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')
    
    output_list = hungarian_matching(dataset_path, num_samples, model, splits_dir)
    
    with open(train_dataset_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)
    
    print(f"Wrote {len(output_list)} examples to {train_dataset_path}")

if __name__ == "__main__":
    train_dataset_path = "/home/om/natural-instructions/train_datasets/hungarian_matching.json"
    dataset_path = "/home/om/natural-instructions/tasks"
    splits_dir = "../splits/default"
    num_samples = 16
    model_name = "intfloat/multilingual-e5-large-instruct"

    training_subset(
        dataset_path=dataset_path,
        num_samples=num_samples,
        model_name=model_name,
        train_dataset_path=train_dataset_path,
        splits_dir=splits_dir
    )
    print("Training subset created with global Hungarian matching.")




