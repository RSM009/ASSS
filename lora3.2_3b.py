import argparse
import time
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import load_dataset
from trl import SFTTrainer

tokenizer = None  # Global tokenizer for formatting function

def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || All params: {all_param} || Trainable%: {100 * trainable_params / all_param:.2f}"
    )

def formatting_prompts_func_few_shot(example):
    output_texts = []
    inputs, outputs = example['input'], example['output']
    global tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer is not initialized before formatting.")
    for user_input, assistant_output in zip(inputs, outputs):
        messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_output}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        output_texts.append(text)
    return output_texts

def initialize_model_and_tokenizer_lora(model_name, adapter_path="", quantization_config=None):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer pad_token to eos_token")
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()

    if model.config.pad_token_id != tokenizer.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Set model pad_token_id to: {tokenizer.pad_token_id}")

    if adapter_path:
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path).eval()

    return tokenizer, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--eval_dataset_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    torch.manual_seed(42)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer, model = initialize_model_and_tokenizer_lora(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        quantization_config=bnb_config,
    )

    print("Loading datasets...")
    train_dataset = load_dataset("json", data_files=args.train_dataset_path, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_dataset_path, split="train")
    eval_dataset = eval_dataset.select(range(min(1000, len(eval_dataset))))
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = None
    if not args.adapter_path:
        print("Applying new LoRA config...")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    else:
        print("Using pre-loaded adapter.")

    print_trainable_parameters(model)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=3,
        bf16=use_bf16,
        fp16=not use_bf16,
        save_steps=500,
        eval_steps=500,
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        report_to="tensorboard",  # or "tensorboard" if needed
    )
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func_few_shot,
        peft_config=peft_config,
    )

    print("Starting Training...")
    trainer.train()
    print("Saving Final Adapter...")
    trainer.save_model(args.output_dir)
    print("Training Complete.")
