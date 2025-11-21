import os
import numpy as np
from collections import defaultdict
import scipy.stats as stats
import re
import math
import pandas as pd


def read_json_lines_to_df(file_path):
    df = pd.read_json(open(file_path, "r", encoding="utf8"), lines=True)
    return df


def print_dict(res_dict):
    for k in sorted(res_dict.keys()):
        print(k)
        print(res_dict[k])


def z_to_fpr(z):
    return stats.norm.sf(z)


def fpr2thres(fpr):
    return (-0.5*math.log(fpr))**0.5


def thres2fpr(thres):
    return math.exp(-2*thres**2)


def get_save_path(score_path,fpr_thres):
    save_dir = os.path.dirname(score_path)
    file_name = score_path.split('/')[-1].split('.')[0] + '_' + str(fpr_thres).replace('.', '_') + '.txt'
    return os.path.join(save_dir, file_name)


def get_KGW_res(score_path,fpr,save_path,len_limit=10):
    df = read_json_lines_to_df(score_path)
    for which_output in ["retokenized_output", "raw_output"]:
        current_df = df[df["which_output"] == which_output]

        tot_cnt=defaultdict(int)
        acc_cnt=defaultdict(int)
        z_score_list=defaultdict(list)
        fpr_list=defaultdict(list)

        for _, row in current_df.iterrows():
            res_dict = row
            if "lens" in res_dict.keys():
                cur_len=res_dict['lens']
            else:
                cur_len=res_dict['output_len']
            wp=res_dict['watermark_processor']
            if cur_len<len_limit:
                continue
            if ("John" not in wp) and ("Unigram" not in wp):
                continue
            
            raw_score=res_dict['raw_scores']
            z_score=2*(raw_score-0.5*cur_len)/cur_len**0.5
            tot_cnt[wp]+=1
            z_score_list[wp].append(z_score)
            cur_fpr=z_to_fpr(z_score)
            
            fpr_list[wp].append(cur_fpr)
            if cur_fpr<=fpr:
                acc_cnt[wp]+=1

        lines = []
        for wp in sorted(tot_cnt.keys()):
            if 'John' in wp:
                delta = re.findall(r"delta=(\d+\.?\d*)", wp)[0]
                wp_name=f'KGW($\\delta$={delta})'
            elif 'Unigram' in wp:
                delta = re.findall(r"delta=(\d+\.?\d*)", wp)[0]
                wp_name=f'Unigram($\\delta$={delta})'
            else:
                raise NotImplementedError

            lines.append(f"Which output: {which_output} " + '-' * (50 - len(which_output)) + '\n')
            lines.append(wp_name + '\n')
            lines.append(f'Total cnt: {tot_cnt[wp]}\n' )
            lines.append(f'Median p-value: {np.median(fpr_list[wp])}\n')
            lines.append(f'TPR@FPR={fpr}: {acc_cnt[wp] / tot_cnt[wp]}\n')
            lines.append('\n')

        print("".join([line for line in lines]))

        with open(save_path, 'a') as f:
            f.writelines(lines)
    return res_dict


def get_dip_res(score_path,fpr,save_path,len_limit=10):
    df = read_json_lines_to_df(score_path)
    for which_output in ["retokenized_output", "raw_output"]:
        current_df = df[df["which_output"] == which_output]
    
        tot_cnt=defaultdict(int)
        acc_cnt=defaultdict(int)
        beta_score_list=defaultdict(list)
        threshold=fpr2thres(fpr)
        
        for _, row in current_df.iterrows():
            res_dict = row
            if "lens" in res_dict.keys():
                cur_len=res_dict['lens']
            else:
                cur_len=res_dict['output_len']
            wp=res_dict['watermark_processor']
            if cur_len<len_limit:
                continue
            if 'Dip' not in wp:
                continue
            
            beta_score=res_dict['beta_score']
            beta_score_list[wp].append(beta_score)
            
            tot_cnt[wp]+=1
            if beta_score>threshold:
                acc_cnt[wp]+=1

        lines = []
        for wp in sorted(tot_cnt.keys()):
            alpha = re.findall(r"alpha=(\d+\.?\d*)", wp)[0]
            if abs(float(alpha)-0.5)<1e-5:
                wp_name='$\\gamma$-reweight'
            else:
                wp_name=f'DiPmark($\\alpha$={alpha})'
            median_beta=np.median(beta_score_list[wp])

            lines.append(f"Which output: {which_output} " + '-' * (50 - len(which_output)) + '\n')
            lines.append(wp_name + '\n')
            lines.append(f'Total cnt: {tot_cnt[wp]}\n' )
            lines.append(f'Median p-value: {thres2fpr(median_beta)}\n')
            lines.append(f'TPR@FPR={fpr}: {acc_cnt[wp] / tot_cnt[wp]}\n')
            lines.append('\n')

        print("".join([line for line in lines]))

        with open(save_path, 'a') as f:
            f.writelines(lines)
    return res_dict


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=str)
    parser.add_argument("--fpr_thres", type=float)
    args = parser.parse_args()

    save_path = get_save_path(args.score_path, args.fpr_thres)

    get_KGW_res(args.score_path, args.fpr_thres, save_path)
    get_dip_res(args.score_path, args.fpr_thres, save_path)
