#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import FloatTensor, LongTensor
from transformers import LogitsProcessor

from .base import AbstractReweight, AbstractWatermarkKey
from typing import List
from .dipmark import Dip_Reweight
from .aligned import  AlignedIS_Reweight
from .unigram import Unigram_Reweight


class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        private_key: any,
        reweight: AbstractReweight, # sample strategy
        watermark_key_list: List[AbstractWatermarkKey]
    ):
        self.watermark_key_list=watermark_key_list
        self.private_key=private_key
        self.reweight=reweight

    def format(self):
        if isinstance(self.reweight, Dip_Reweight):
            return f"DiP_alpha_{repr(self.reweight.alpha)}".replace(".", "_")
        elif isinstance(self.reweight, AlignedIS_Reweight):
            return f"AlignedIS_n_{repr(self.reweight.n)}".replace(".", "_")
        elif isinstance(self.reweight, Unigram_Reweight):
            return f"Unigram_{repr(self.gamma)}_delta_{repr(self.delta)}".replace(".", "_")
        else:
            return "unknown"

    def __repr__(self):
        watermark_str=', '.join([repr(watermark_key) for watermark_key in self.watermark_key_list])
        
        res_str=f"WatermarkLogitsProcessor(private_key={repr(self.private_key)}, reweight={repr(self.reweight)}, watermark_key_list=[{watermark_str}])"
    
        return res_str

    def get_rng_seed(self, key_list) -> any:
        import hashlib
        m = hashlib.sha256()
        # m.update(self.private_key)    
        for key in key_list:
            m.update(key)
        full_hash = m.digest()
        seed = int.from_bytes(full_hash, "big") % (2**32 - 1)
        return seed

    
    def reset_watermark_key(self,batch_size):
        for watermark_key in self.watermark_key_list:
            watermark_key.reset(batch_size)

    def _get_codes(self, input_ids: LongTensor):
        batch_size = input_ids.size(0)

        mask=[]
        seeds=[]
        for batch_idx in range(batch_size):
            cur_mask=0
            key_list=[self.private_key]
            for watermark_key in self.watermark_key_list:
                cur_wm_mask,cur_wm_key=watermark_key.generate_key_and_mask(input_ids[batch_idx],batch_idx)
                if cur_wm_key is not None:
                    key_list.append(cur_wm_key)
                cur_mask=(cur_mask or cur_wm_mask)
            mask.append(cur_mask)
            seeds.append(self.get_rng_seed(key_list))
        return mask, seeds

    def _core(self, input_ids: LongTensor, scores: FloatTensor):
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=scores.device).manual_seed(seed) for seed in seeds
        ]
        mask = torch.tensor(mask, device=scores.device,dtype=torch.bool)
        
        # from_random is only called here, creating the Aligned_WatermarkCode class
        if isinstance(self.reweight,AlignedIS_Reweight):
            watermark_code = self.reweight.watermark_code_type.from_random(
                rng, scores.size(1),self.reweight.n
            )
        else:
            watermark_code = self.reweight.watermark_code_type.from_random(
                rng, scores.size(1)
            )
        # Call the reweight
        reweighted_scores = self.reweight.reweight_logits(watermark_code, scores)
        return mask, reweighted_scores

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        mask, reweighted_scores = self._core(input_ids, scores)
        return torch.where(mask[:, None], scores, reweighted_scores)

    def get_green_token_quantile(self, input_ids: LongTensor, vocab_size, current_token,debug=False):
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]
        assert isinstance(self.reweight,Dip_Reweight)
        mask = torch.tensor(mask, device=input_ids.device)
        watermark_code = self.reweight.watermark_code_type.from_random(
            rng, vocab_size
        )
        
        # calculate the score here
        token_quantile = [(torch.where(watermark_code.shuffle[i] == current_token[i])[0]+1)/vocab_size
                        for i in range(input_ids.shape[0])]
        
        return token_quantile
    
    def get_unigram_score(self, input_ids: LongTensor, vocab_size, current_token,debug=False):
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]
        assert isinstance(self.reweight,Unigram_Reweight)
        mask = torch.tensor(mask, device=input_ids.device)
        watermark_code = self.reweight.watermark_code_type.from_random(
            rng, vocab_size
        )
        
        green_list_size=round(self.reweight.gamma*vocab_size)
        scores=[torch.tensor(current_token[i] in watermark_code.shuffle[i][:green_list_size]).float() for i in range(input_ids.shape[0])]
        
        return scores

    def get_cluster_n_res(self,input_ids: LongTensor, vocab_size, current_token, cur_n, cluster_dict):
        assert isinstance(self.reweight, AlignedIS_Reweight)
        assert self.reweight.n == cur_n

        mask, seeds = self._get_codes(input_ids)
        rng = [torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds]
        mask = torch.tensor(mask, device=input_ids.device)
        # from_random is only called here, creating the Aligned_WatermarkCode class
        watermark_code = self.reweight.watermark_code_type.from_random(
            rng, vocab_size, self.reweight.n
        )

        if vocab_size % cur_n == 0:
            splits = torch.arange(start=0, end=vocab_size).reshape(cur_n, vocab_size // cur_n).to(input_ids.device)
        else:
            splits = [[] for _ in range(cur_n)]

            # Assign audio tokens to the correct splits
            for token_id, split_index in cluster_dict.items():
                splits[split_index].append(token_id)

            # Assign remaining tokens using block-based splitting
            remaining_tokens = sorted(set(range(vocab_size)) - set(cluster_dict.keys()))
            for n_idx in range(cur_n):
                start = round(len(remaining_tokens) * n_idx / cur_n)
                end = round(len(remaining_tokens) * (n_idx + 1) / cur_n)
                splits[n_idx].extend(remaining_tokens[start:end])

        scores = []
        for bsz_idx in range(input_ids.shape[0]):
            cur_k = watermark_code.split_k[bsz_idx]
            if self.reweight.shuffle:
                current_split = watermark_code.shuffle[bsz_idx][splits[cur_k]]
            else:
                current_split = splits[cur_k]

            if current_token[bsz_idx] in current_split:
                scores.append(1)
            else:
                scores.append(0)

        return scores


class WatermarkLogitsProcessor_Baseline(LogitsProcessor):
    # def __init__(self):
    #     super().__init__()
    #     raise NotImplementedError

    def format(self):
        return "unwatermarked"
    
    def __repr__(self):
        return f"WatermarkLogitsProcessor_Baseline()"


    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        return scores