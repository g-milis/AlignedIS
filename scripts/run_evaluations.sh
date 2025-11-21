# Set fpr to 1%
fpr=0.01

# The `score_path` argument is the scoring file created at the end of generation

# For AlignedIS
python evaluations/get_aligned_acc.py \
    --fpr_thres $fpr \
    --score_path results/finance_qa/spirit_lm_base_7b/alignedis/score.txt

# For baselines
python evaluations/get_baselines_acc.py \
    --fpr_thres $fpr \
    --score_path results/finance_qa/spirit_lm_base_7b/baselines/score.txt
