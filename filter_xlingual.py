import json
import argparse

def filter_prompts_by_reference_ids(prompts_file, references_file, output_file):
    """Keep only prompts whose IDs exist in the filtered references file"""
    # Load reference IDs
    reference_ids = set()
    with open(references_file, 'r', encoding='utf-8') as ref_file:
        for line in ref_file:
            try:
                entry = json.loads(line)
                reference_ids.add(entry["id"])
            except (json.JSONDecodeError, KeyError):
                print(f"Skipping invalid reference entry: {line.strip()}")
    
    # Filter prompts
    kept = 0
    with open(prompts_file, 'r', encoding='utf-8') as prompts, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in prompts:
            try:
                prompt_entry = json.loads(line)
                if prompt_entry["id"] in reference_ids:
                    outfile.write(json.dumps(prompt_entry, ensure_ascii=False) + '\n')
                    kept += 1
            except (json.JSONDecodeError, KeyError):
                print(f"Skipping invalid prompt entry: {line.strip()}")
    
    print(f"Kept {kept} prompts matching reference IDs")

def main():
    parser = argparse.ArgumentParser(description='Filter prompts by reference IDs')
    parser.add_argument('--prompts', required=True, help='Input prompts JSONL file')
    parser.add_argument('--references', required=True, help='Filtered references JSONL file')
    parser.add_argument('--output', required=True, help='Output JSONL file path')
    args = parser.parse_args()

    filter_prompts_by_reference_ids(args.prompts, args.references, args.output)

if __name__ == "__main__":
    main()
