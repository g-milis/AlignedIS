dataset_name_list=('mmw_book_report' 'mmw_story' 'mmw_fake_news' 'dolly_cw' 'longform_qa' 'finance_qa' 'librispeech')
# Run on just one dataset instead of all
dataset_name_list=('finance_qa')

# NOTE: For reweight type, use either "alignedis" or "baselines"
reweight_types=("alignedis" "baselines")

# You may set max_generations to a higher number, or remove it to sample the whole dataset

for dataset_name in "${dataset_name_list[@]}"; do
    for reweight in "${reweight_types[@]}"; do
        python -m experiments \
        --res_dir results \
        --model_str SpeechGPT-7B-cm \
        --reweight_type $reweight \
        --dataset_name $dataset_name \
        --max_generations 2
    done
done
