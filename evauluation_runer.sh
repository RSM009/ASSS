
export CUDA_VISIBLE_DEVICES=1
python evaluation.py \
  --reference_file ./natural-instructions/eval/leaderboard/test_references.jsonl \
  --temperature 0.5 \
  --max_new_tokens 40 

