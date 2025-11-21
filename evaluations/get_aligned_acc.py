import os
import numpy as np
from collections import defaultdict
import math
import re
import numpy as np
import pandas as pd


def read_json_lines_to_df(file_path):
    df = pd.read_json(open(file_path, "r", encoding="utf8"), lines=True)
    return df


def get_split_fpr(n, m, split_num):
    res = 0
    res = sum(
        math.comb(n, i) * pow(split_num - 1, n - i)
        for i in range(m, n + 1)
    )
    res /= split_num ** n
    return res


def extract_n_value(text):
    pattern = re.search(r"AlignedIS_Reweight\(n=(\d+)", text)
    if pattern:
        return int(pattern.group(1))
    raise NotImplementedError


def get_save_path(score_path,fpr_thres):
    save_dir = os.path.dirname(score_path)
    file_name = score_path.split('/')[-1].split('.')[0] + '_' + str(fpr_thres).replace('.', '_') + '.txt'
    return os.path.join(save_dir, file_name)


def generate_result(score_path, save_path, fpr_thres, len_limit, seed=None, shuffle=True):
    df = read_json_lines_to_df(score_path)

    for reweight in ["AlignedIS_Reweight"]:
        for which_output in ["retokenized_output", "raw_output"]:
            current_df = df[df["which_output"] == which_output]

            tot_cnt = defaultdict(int)
            acc_cnt = defaultdict(int)
            fpr_list = defaultdict(list)

            for _, row in current_df.iterrows():
                res_dict = row
                cur_len = res_dict['lens']

                wp = res_dict['watermark_processor']

                if cur_len < len_limit:
                    continue
                if f"reweight={reweight}" not in wp:
                    continue

                if seed is not None:
                    if f"seed={seed}" not in wp:
                        continue
                    if f"shuffle={shuffle}" not in wp:
                        continue

                raw_score = res_dict["raw_scores"]
                cur_n = extract_n_value(wp)

                cur_fpr = get_split_fpr(int(cur_len), int(raw_score), split_num=cur_n)
                fpr_list[cur_n].append(cur_fpr)

                if cur_fpr <= fpr_thres:
                    acc_cnt[cur_n] += 1
                tot_cnt[cur_n] += 1

            lines = []
            for wp in sorted(tot_cnt.keys()):
                lines.append(f"Which output: {which_output} " + '-' * (50 - len(which_output)) + '\n')
                lines.append(f"{reweight}(n={wp}, seed={seed}, shuffle={shuffle})\n")
                lines.append(f'Total cnt: {tot_cnt[wp]}\n' )
                lines.append(f'Median p-value: {np.median(fpr_list[wp])}\n')
                lines.append(f'TPR@FPR={fpr_thres}: {acc_cnt[wp] / tot_cnt[wp]}\n')


            print("".join([line for line in lines]))

            with open(save_path, 'a') as f:
                f.writelines(lines)

    return lines


def get_lines(score_path, fpr_thres, len_limit, seed=None, shuffle=True):
    save_path = get_save_path(score_path, fpr_thres)
    
    if not os.path.exists(save_path):
        save_dir='/'.join(save_path.split('/')[:-1])
        os.makedirs(save_dir,exist_ok=True)

    lines = generate_result(score_path, save_path, fpr_thres, len_limit, seed, shuffle)
        
    with open(save_path,'r') as f:
        lines = f.readlines()
    return lines


def get_result_dict(score_path, fpr_thres, len_limit=10, seed=None, shuffle=True):
    lines = get_lines(score_path, fpr_thres, len_limit, seed, shuffle)

    line_num = len(lines)
    assert line_num % 5 == 0
    res_num = line_num // 5
    res_list = []
    for res_idx in range(res_num):
        cur_n = extract_n_value(lines[res_idx * 5 + 1].strip())
        median_p = float((lines[res_idx * 5 + 3].strip()).split(':')[-1])
        tpr = float((lines[res_idx * 5 + 4].strip()).split(':')[-1])
        res_list.append((cur_n, {'median_p': median_p, 'tpr': tpr}))
    
    # Sort by cur_n
    res_list.sort(key=lambda x: int(x[0]))

    # Build sorted dict
    res_dict = {f"AlignedIS(n={n})": vals for n, vals in res_list}
    return res_dict


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=str)
    parser.add_argument("--fpr_thres", type=float)
    args = parser.parse_args()

    get_result_dict(args.score_path, args.fpr_thres)
