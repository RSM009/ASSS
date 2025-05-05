import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def extract_input_output_for_finetune(input_filename, output_filename):
    """
    Extracts 'input_prompt' and 'output' fields from a JSONL file,
    ignoring entries with 'track' == 'default', and writes cleaned data to a new JSON file.
    """
    extracted_data = []

    try:
        with open(input_filename, "r", encoding="utf-8") as infile:
            for line_num, line in enumerate(infile, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)

                    # Skip entries with track == "default"
                    if entry.get("track") == "default":
                        continue

                    if "input_prompt" in entry and "output" in entry:
                        raw_input = entry["input_prompt"].strip()

                        # Handle string or list outputs
                        output = entry["output"]
                        if isinstance(output, list):
                            output_text = " ".join(output).strip()
                        else:
                            output_text = str(output).strip()

                        extracted_data.append({
                            "input": raw_input,
                            "output": output_text
                        })
                    else:
                        logging.warning(f"Line {line_num}: Missing 'input_prompt' or 'output'")

                except json.JSONDecodeError as e:
                    logging.error(f"Line {line_num}: JSON decode error - {e}")

        # Write cleaned data to output file
        with open(output_filename, "w", encoding="utf-8") as outfile:
            json.dump(extracted_data, outfile, ensure_ascii=False, indent=2)

        logging.info(f"Successfully processed {len(extracted_data)} entries into '{output_filename}'")

    except FileNotFoundError:
        logging.error(f"Input file '{input_filename}' not found.")
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")

# 🔧 Set file paths here
if __name__ == "__main__":
    base_filename = "ponss_k32_2"
    input_json_file = f"{base_filename}.json"
    output_json_file = f"train_data_before_split_{base_filename}.json"

    extract_input_output_for_finetune(input_json_file, output_json_file)
