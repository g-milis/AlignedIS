dataset_name_list=('finance_qa')

for dataset_name in "${dataset_name_list[@]}"; do
    python -m evaluations.run_and_score_attacks \
    --model_str SpeechGPT-7B-cm \
    --generations_file results/$dataset_name/SpeechGPT_7B_cm/alignedis/audio_generation.txt \
    --attack_configs evaluations/configs/attack_light_configs.json
done
