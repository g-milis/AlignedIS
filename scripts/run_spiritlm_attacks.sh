dataset_name_list=('finance_qa')

for dataset_name in "${dataset_name_list[@]}"; do
    python -m evaluations.run_and_score_attacks \
    --model_str spirit-lm-base-7b \
    --generations_file results/$dataset_name/spirit_lm_base_7b/alignedis/audio_generation.txt \
    --attack_configs evaluations/configs/attack_light_configs.json
done
