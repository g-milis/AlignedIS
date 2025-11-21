import torch
import os


def get_wps(reweight_type, context_length=1, model_str=None):
    from watermarks import (
        WatermarkLogitsProcessor,
        PrevN_ContextCodeExtractor,
        WatermarkLogitsProcessor_Baseline,
        NGramHashing,
        Dip_Reweight,
        AlignedIS_Reweight,
        Unigram_Reweight,
    )
    
    from ..lm_watermarking.watermark_processor import (
        WatermarkLogitsProcessor as WatermarkLogitsProcessor_John,
    )

    import random
    import copy

    random.seed(42)
    private_key = random.getrandbits(1024).to_bytes(128, "big")
    
    watermark_key_list = [
        NGramHashing(PrevN_ContextCodeExtractor(context_length), ignore_history=False)
    ]

    reweight_list=[]
    if reweight_type=='alignedis':
        reweight_list = [
            AlignedIS_Reweight(20, seed=0, shuffle=False, model_str=model_str),
        ]

    elif reweight_type=='baselines':
        reweight_list=[
            Dip_Reweight(alpha=0.5),
            Dip_Reweight(alpha=0.4),
            Dip_Reweight(alpha=0.3),
            Unigram_Reweight(delta=1.0,gamma=0.5),
            Unigram_Reweight(delta=1.5,gamma=0.5),
            Unigram_Reweight(delta=2.0,gamma=0.5)
        ]

    elif reweight_type=='ablation' or reweight_type=='ablation_new':
        reweight_list = [
            AlignedIS_Reweight(n) for n in [2, 5, 8, 10, 13, 15, 17, 20, 40, 60, 80, 100]
        ]

    else:
        print('Unknown reweight_type: ',reweight_type)
        raise AttributeError
        
    wm_wps = []
    
    '''
    Commmon WatermarkLogitsProcessor
    '''
    for wm_key in watermark_key_list:
        for reweight in reweight_list:
            wm_wps.append(
                WatermarkLogitsProcessor(
                    private_key,
                    reweight=copy.deepcopy(reweight),
                    watermark_key_list=[copy.deepcopy(wm_key)],
                )
            )
            
    john_wps = [
            WatermarkLogitsProcessor_John(
                vocab_size=0,  # placeholder
                gamma=0.5,
                delta=delta,
                seeding_scheme="simple_1",
            )
            for delta in [1.0, 1.5, 2.0]
        ]

    baseline_wp = WatermarkLogitsProcessor_Baseline() # no watermark baseline  
    if reweight_type == "baselines":
        return [*wm_wps, *john_wps, baseline_wp]
    else:
        return [*wm_wps]


def get_num_gpus():
    import torch

    num_gpus = torch.cuda.device_count()
    return num_gpus


def batched_wp_task_worker(tq, get_in_ds, reweight_type, dataset_name, context_length=1, batch_size=8, max_generations=None, model_str="spirit-lm-base-7b"):
    ds = get_in_ds(dataset_name=dataset_name, max_generations=max_generations, extend="speech" in model_str.lower())

    from .common import get_wps

    wps = get_wps(reweight_type=reweight_type, context_length=context_length, model_str=model_str)

    from tqdm import tqdm

    # cnt=0
    print("tqdm for loading the task queue (batch x wp):")
    for batch in tqdm(ds.iter(batch_size=batch_size), total=len(ds) // batch_size):
        for wp in wps:
            tq.put({"batch": batch, "watermark_processor": wp})


def merged_task_worker(
    get_in_ds,
    output_filepath,
    tq,
    model_str,
    batch_size=8,
    watermark_only=False,
    wh_only=False,
    no_gumbel=False,
    beta_only=False,
    dataset_name=None,
    max_generations=None
):
    in_ds = get_in_ds(dataset_name=dataset_name, max_generations=max_generations, extend="speech" in model_str.lower())

    from datasets import load_dataset

    out_ds = load_dataset("json", data_files={"test": output_filepath})["test"]
    out_ds = out_ds.sort("id")

    dss, wps = add_reference(in_ds, out_ds)

    from tqdm import tqdm

    for ds, wp_str in zip(dss, wps):
        if watermark_only:
            if "John" in wp_str or "None" == wp_str:
                continue
        if wh_only:
            if ", True)" in wp_str:
                continue
        if no_gumbel:
            if "Gumbel" in wp_str:
                continue
        if beta_only:
            if "Beta_Reweight" not in wp_str:
                continue
        print("tqdm for loading the task queue (batch, no wp):")
        for batch in tqdm(ds.iter(batch_size=batch_size), total=len(ds) // batch_size):
            tq.put(batch)


def log(line: dict, f):
    import json

    f.write(json.dumps(line))
    f.write("\n")
    f.flush()


def simple_store_worker(path, rq, rqe):
    import os

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    from queue import Empty

    with open(path, "w") as f:
        while not (rqe.is_set() and rq.empty()):
            try:
                result = rq.get(timeout=1)
            except Empty as e:
                continue
            assert isinstance(result, dict)
            if result == {}:
                continue
            if isinstance(next(iter(result.values())), list):
                assert all([isinstance(v, list) for v in result.values()])
                lens = [len(v) for v in result.values()]
                assert all([l == lens[0] for l in lens])
                for i in range(lens[0]):
                    log({k: v[i] for k, v in result.items()}, f)
            else:
                log(result, f)


from typing import Union


def set_spawn():
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def remove_tailing_pad_s(s: str):
    while s.endswith("<pad>"):
        s = s[:-5]
    return s
    #  index = s.find("<pad>")
    #  if index == -1:
    #      return s
    #  else:
    #      return s[:index]


def remove_tailing_pad(strs: list[str]):
    return [remove_tailing_pad_s(s) for s in strs]


def remove_text_worker(tq, tqe, rq):
    from queue import Empty

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        for f in ["input", "output", "raw_output", "retokenized_output", "reference", "display_output"]:
            if f in batch:
                del batch[f]
        rq.put(batch)


def spiritlm_worker(
    tq,
    tqe,
    rq,
    gpu_id,
    model_str,
    save_dir="results",
    generation_kwargs={},
    decoder_only=False,
    tokenization_kwargs={},
):
    from transformers import set_seed, GenerationConfig, LogitsProcessorList
    from models.spiritlm.model.spiritlm_model import Spiritlm, OutputModality, GenerationInput, ContentType
    import soundfile as sf


    model = Spiritlm(model_str, f"cuda:{gpu_id}")
    generation_config = GenerationConfig(
        temperature=1.0,
        max_new_tokens=512,
        do_sample=True
    )

    from queue import Empty
    model.model.eval()
    with torch.no_grad():
        while not (tqe.is_set() and tq.empty()):
            try:
                task = tq.get(timeout=1)
            except Empty as e:
                continue
            batch = task["batch"]
            wp = task["watermark_processor"]
            lps = []
            if wp is not None:
                if "reset_watermark_key" in dir(wp):
                    batch_size = len(batch["id"])
                    wp.reset_watermark_key(batch_size)
                if "vocab_size" in dir(wp):
                    if model_str=="spirit-lm-base-7b":
                        wp.vocab_size = 32512
                    elif model_str=="spirit-lm-expressive-7b":
                        wp.vocab_size = 32768
                    else:
                        print('Unknown model_str: ',model_str)
                        print('Please set vocab size.')
                        raise NotImplementedError

                lps.append(wp)

            # for reproducibility and sufficient randomness
            import hashlib

            hash = hashlib.sha256()
            hash.update((str(batch["id"]) + repr(wp)).encode("utf-8"))
            seed = hash.digest()
            seed = int.from_bytes(seed, "big") % (2**32 - 1)

            set_seed(seed)

            try:
                response, outputs_ids = model.generate(
                    interleaved_inputs=[
                        GenerationInput(
                            content=batch['input'][0],
                            content_type=(ContentType.SPEECH if batch['input'][0].startswith("/") else ContentType.TEXT),
                        )
                    ],
                    output_modality=OutputModality.SPEECH,
                    logits_processor=LogitsProcessorList(lps),
                    generation_config=generation_config
                )

                response = response[0].content

                filename = f"{save_dir}/{wp.format()}_{batch['id'][0]}.wav"
                sf.write(filename, response, 16000)

                retokenized_output = model.speech_tokenizer(torch.from_numpy(response))
                retokenized_output_ids = torch.tensor(model.tokenizer(retokenized_output).input_ids).unsqueeze(0)
                wp_str = repr(wp)

                rq.put(
                    {
                        "raw_output": outputs_ids.tolist(),
                        "retokenized_output": retokenized_output_ids.tolist(),
                        "output_len": [len(outputs_ids.squeeze().tolist())],
                        "retokenized_len": [len(retokenized_output_ids.squeeze().tolist())],
                        "id": batch["id"],
                        "reference_id": batch["reference_id"],
                        "watermark_processor": [wp_str],
                        "wav_path": [filename]
                    }
                )
            except Exception as e:
                wp_str = repr(wp)
                print('Caught error in :',repr(wp))
                print('error info:', )
                print(e)
                
                rq.put(
                    {
                        "raw_output": [[1]],
                        "retokenized_output": [[1]],
                        "output_len": [0],
                        "retokenized_len": [0],
                        "id": batch["id"],
                        "reference_id": batch["reference_id"],
                        "watermark_processor": [wp_str],
                        "wav_path": [""]
                    }
                )


def speechgpt_worker(
    tq,
    tqe,
    rq,
    gpu_id,
    model_str,
    save_dir="results",
    generation_kwargs={},
    decoder_only=False,
    tokenization_kwargs={},
):
    from transformers import set_seed, GenerationConfig, LogitsProcessorList
    from models.speechgpt import SpeechGPT
    import soundfile as sf

    model = SpeechGPT(model_str, device=f"cuda:{gpu_id}")
    generation_config = GenerationConfig(
        min_new_tokens=400,
        max_new_tokens=600,
        temperature=0.8,
        repetition_penalty=1.2,
        top_k=60,
        top_p=0.8,
        do_sample=True,
        eos_token_id=[2, 33001, 33005]
    )


    from queue import Empty
    model.model.eval()
    with torch.no_grad():
        while not (tqe.is_set() and tq.empty()):
            try:
                task = tq.get(timeout=1)
            except Empty as e:
                continue
            batch = task["batch"]
            wp = task["watermark_processor"]
            lps = []
            if wp is not None:
                if "reset_watermark_key" in dir(wp):
                    batch_size = len(batch["id"])
                    wp.reset_watermark_key(batch_size)
                if "vocab_size" in dir(wp):
                    wp.vocab_size = 33006

                lps.append(wp)

            # for reproducibility and sufficient randomness
            import hashlib

            hash = hashlib.sha256()
            hash.update((str(batch["id"]) + repr(wp)).encode("utf-8"))
            seed = hash.digest()
            seed = int.from_bytes(seed, "big") % (2**32 - 1)
            set_seed(seed)

            try:
                outputs_ids, generated_units, wav = model(
                    ["Read this sentence aloud, this is input: " + batch['input'][0]],
                    generation_config=generation_config,
                    logits_processor=LogitsProcessorList(lps),
                    enforce_speech=True
                )

                wav = wav.numpy()

                filename = f"{save_dir}/{wp.format()}_{batch['id'][0]}.wav"
                sf.write(filename, wav, 16000)

                # Retokenize Speech (units)
                retokenized_output = model.s2u.from_wav(torch.from_numpy(wav))
                # Convert speech units to LLM vocabulary (tokens)
                retokenized_output_ids = torch.tensor(model.tokenizer(retokenized_output).input_ids).unsqueeze(0)

                wp_str = repr(wp)
                rq.put(
                    {
                        "raw_output": outputs_ids,
                        "retokenized_output": retokenized_output_ids.tolist(),
                        "output_len": [len(outputs_ids[0])],
                        "retokenized_len": [len(retokenized_output_ids.squeeze())],
                        "id": batch["id"],
                        "reference_id": batch["reference_id"],
                        "watermark_processor": [wp_str],
                        "wav_path": [filename]
                    }
                )
            except Exception as e:
                wp_str = repr(wp)
                print('Caught error in:', repr(wp))
                print('error info:', )
                print(e)

                rq.put(
                    {
                        "raw_output": [[1]],
                        "retokenized_output": [[1]],
                        "output_len": [0],
                        "retokenized_len": [0],
                        "id": batch["id"],
                        "reference_id": batch["reference_id"],
                        "watermark_processor": [wp_str],
                        "wav_path": [""]
                    }
                )


def add_reference(in_ds, out_ds):
    """assuming ordered by ids"""
    wp_types = set(out_ds["watermark_processor"])

    s_out_dss = []
    for wp_type in wp_types:
        s_out_ds = out_ds.filter(lambda x: x["watermark_processor"] == wp_type)
        s_out_dss.append(s_out_ds)
    from datasets import concatenate_datasets

    return s_out_dss, wp_types


# calc p val for EXP_edit and ITS_edit
def get_p_val_id(vocab_size, output_ids, wp, device, test_config={}, la_wp=None, eps=0,gamma=0):
    assert eps <= 1
    assert eps >= 0
    
    decoder_input_ids=output_ids.to(device)
    label_attention_mask=torch.ones_like(decoder_input_ids).to(device)
    
    if eps > 0:
        decoder_input_ids = random_paraphrase(decoder_input_ids, vocab_size,device, eps)
    
    p_val=wp.get_p_val(decoder_input_ids,vocab_size,gamma=gamma)
    
    return p_val, label_attention_mask


@torch.no_grad()
def get_unigram_score_id(vocab_size, output_ids, wp, device, test_config={}, la_wp=None, eps=0):
    assert eps <= 1
    assert eps >= 0
    
    decoder_input_ids=output_ids.to(device)
    label_attention_mask=torch.ones_like(decoder_input_ids).to(device)
    
    if eps > 0:
        decoder_input_ids = random_paraphrase(decoder_input_ids, vocab_size,device, eps)

    scores = torch.zeros(decoder_input_ids.shape, device=device)
    for i in range(decoder_input_ids.size(1) - 1):
        pre = decoder_input_ids[:, : i + 1]
        cur_token = decoder_input_ids[:, i + 1]
        out = wp.get_unigram_score(pre, vocab_size, cur_token)
        scores[:, i+1] = torch.stack(out).reshape(-1)

    return scores, label_attention_mask


# for Dipmark and gamma-reweight
@torch.no_grad()
def get_quantile_id(vocab_size, output_ids, wp, device, test_config={}, la_wp=None, eps=0):
    assert eps <= 1
    assert eps >= 0
    
    decoder_input_ids = output_ids.to(device)

    label_attention_mask = torch.ones_like(decoder_input_ids).to(device)

    if eps > 0:
        decoder_input_ids = random_paraphrase(decoder_input_ids, vocab_size,device, eps)

    quantile = torch.zeros(decoder_input_ids.shape, device=device)
    for i in range(decoder_input_ids.size(1) - 1):
        pre = decoder_input_ids[:, : i + 1]
        cur_token = decoder_input_ids[:, i + 1]
        out = wp.get_green_token_quantile(pre, vocab_size, cur_token)
        quantile[:, i+1] = torch.stack(out).reshape(-1)

    return quantile, label_attention_mask

# for Splitmark
@torch.no_grad()
def get_split_res_id(vocab_size, output_ids, wp, device, test_config={}, la_wp=None, eps=0,split_num=None):
    assert eps <= 1
    assert eps >= 0
    decoder_input_ids=output_ids.to(device)
    label_attention_mask=torch.ones_like(decoder_input_ids).to(device)

    if eps > 0:
        decoder_input_ids = random_paraphrase(decoder_input_ids, vocab_size,device, eps)

    scores = torch.zeros(decoder_input_ids.shape, device=device)
    for i in range(decoder_input_ids.size(1) - 1):
        pre = decoder_input_ids[:, : i + 1]
        cur_token = decoder_input_ids[:, i + 1]
        out = wp.get_n_res(pre, vocab_size, cur_token,cur_n=split_num)
        scores[:, i+1] = torch.tensor(out).reshape(-1)
    return scores, label_attention_mask
    

@torch.no_grad()
def get_cluster_split_res_id(vocab_size, output_ids, wp, device, test_config={}, la_wp=None, eps=0, split_num=None):
    assert eps <= 1
    assert eps >= 0
    decoder_input_ids = output_ids.to(device)
    label_attention_mask=torch.ones_like(decoder_input_ids).to(device)

    if eps > 0:
        decoder_input_ids = random_paraphrase(decoder_input_ids, vocab_size,device, eps)

    scores = torch.zeros(decoder_input_ids.shape, device=device)
    for i in range(decoder_input_ids.size(1) - 1):
        pre = decoder_input_ids[:, :(i + 1)]
        cur_token = decoder_input_ids[:, i + 1]
        out = wp.get_cluster_n_res(pre, vocab_size, cur_token, split_num, wp.reweight.clusters)
        scores[:, i + 1] = torch.tensor(out).reshape(-1)
    return scores, label_attention_mask


from ..lm_watermarking.watermark_processor import (
    WatermarkLogitsProcessor as WatermarkLogitsProcessor_John,
)

@torch.no_grad()
def get_green_token_scores_id(
    vocab_size, output_ids, wp, device, eps=0
):
    assert eps <= 1
    assert eps >= 0
    
    decoder_input_ids = output_ids.to(device)
    label_attention_mask = torch.ones_like(decoder_input_ids)

    if eps > 0:
        decoder_input_ids = random_paraphrase(decoder_input_ids, vocab_size, device,eps)

    scores = torch.zeros(decoder_input_ids.shape, device=device)

    assert decoder_input_ids.shape[0] == 1
    for i in range(decoder_input_ids.size(1) - 1):
        pre = decoder_input_ids[0, : i + 1]
        cur_token = decoder_input_ids[0, i + 1]
        assert wp.select_green_tokens
        green_token_ids=wp._get_greenlist_ids(pre)
        if cur_token in green_token_ids:
            scores[0,i+1]=1
    return scores, label_attention_mask



@torch.no_grad()
def get_synthid_text_scores_id(
    vocab_size, output_ids, wp, device, eps=0
):
    # raise NotImplementedError
    assert eps <= 1
    assert eps >= 0
    
    decoder_input_ids = output_ids.to(device)
    label_attention_mask = torch.ones_like(decoder_input_ids)

    if eps > 0:
        decoder_input_ids = random_paraphrase(decoder_input_ids, vocab_size, device,eps)

    scores = torch.zeros(decoder_input_ids.shape, device=device)

    assert decoder_input_ids.shape[0] == 1
    for i in range(decoder_input_ids.size(1) - 1):
        pre = decoder_input_ids[:, : i + 1]
        cur_token = decoder_input_ids[:, i + 1]
        cur_score=wp.get_synthid_text_res(pre, vocab_size, cur_token)
        scores[:,i+1]=cur_score
        
    return scores, label_attention_mask

def beta_score_worker(
    tq, tqe, rq, gpu_id, oracle_model_str, eps=0, decoder_only=False, tokenization_kwargs={}
):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed, AutoModelForCausalLM, LlamaTokenizer
    from transformers import LogitsProcessorList, TemperatureLogitsWarper, GenerationConfig


    # if decoder_only:
    #     model = AutoModelForCausalLM.from_pretrained(oracle_model_str).to(
    #         f"cuda:{gpu_id}"
    #     )
    # else:
    #     model = AutoModelForSeq2SeqLM.from_pretrained(oracle_model_str).to(
    #         f"cuda:{gpu_id}"
    #     )
    # tokenizer = AutoTokenizer.from_pretrained(oracle_model_str)

    from models.spiritlm.model.spiritlm_model import Spiritlm, OutputModality, GenerationInput, ContentType


    device = f"cuda:{gpu_id}"
    if 'spirit-lm' in oracle_model_str:
        # model = Spiritlm(oracle_model_str,f"cuda:{gpu_id}")#.to(f"cuda:{gpu_id}")
        # generation_config = GenerationConfig(
        #     temperature=1,
        #     # top_p=0.95,
        #     # top_k=0,
        #     max_new_tokens=10,
        #     do_sample=True
        # )
        # tokenizer = AutoTokenizer.from_pretrained(model_str)
        # vocab_size = max(model.tokenizer.get_vocab().values())
        if oracle_model_str == "spirit-lm-base-7b":
            vocab_size = 32512
        elif oracle_model_str == "spirit-lm-expressive-7b":
            vocab_size = 32768
        else:
            print('Unknown oracle_model_str: ',oracle_model_str)
            print('Please set vocab size.')
            raise NotImplementedError
    elif "SpeechGPT" in oracle_model_str:
        vocab_size = 33006

        # del model
    else:
        print('Unknown oracle_model_str: ',oracle_model_str)
        raise NotImplementedError
    #     print(vocab_size)
    from queue import Empty
    
    
    def chernoff_bound(quantiles,lens):
        avg_score=torch.sum(quantiles>0.5,dim=-1)/lens
        bound=(((1-avg_score)**(avg_score-1))/(2*avg_score**avg_score))**lens
        return avg_score,bound

    def score_func(quantiles, lens, mode='linear'):
        """
        params:
            quantiles: [batch_size, max_len]
        """
        if mode == "linear":
            return torch.sum(quantiles, dim=-1), lens / 2

        if mode == "test":
            return torch.sum(quantiles > 0.5, dim=-1), lens / 2
        
        if mode=='scaled_sigmoid':
            left=-10
            right=10
            return torch.sum(torch.sigmoid(quantiles*(right-left)+left),dim=-1),lens/2
        
        if mode == 'scaled_log':
            left=1
            right=torch.e
            return torch.sum(torch.log(quantiles*(right-left)+left),dim=-1),1/(torch.e-1)*lens
        
        return NotImplementedError

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        assert len(set(batch["watermark_processor"])) == 1

        wp_str = batch["watermark_processor"][0]

        from watermarks import (
            WatermarkLogitsProcessor,
            PrevN_ContextCodeExtractor,
            WatermarkLogitsProcessor_Baseline,
            NGramHashing,
            FixedKeySet,
            PositionHashing,
            KeySequence,
            NoKey,
            Dip_Reweight,
            AlignedIS_Reweight,
            Unigram_Reweight,
        )

        wp = eval(wp_str)
        # print(wp)
        # raise 
        # wp.reset_watermark_key(len(batch["watermark_processor"]))
        # wp.ignore_history = True

        la_wp = None

        # Evaluate for both pure generated ids and retokenized from audio
        if "retokenized_output" in batch.keys():
            output_kinds = ["raw_output", "retokenized_output"]
        elif "raw_output" in batch.keys():
            output_kinds = ["raw_output"]
        else:
            output_kinds = ["output"]

        for output_kind in output_kinds:

            output_ids=torch.tensor(batch[output_kind]).to(device)
            
            if 'AlignedIS_Reweight' in wp_str:
                wp = eval(wp_str)
                wp.reset_watermark_key(len(batch["watermark_processor"]))
                wp.ignore_history = True
                
                import re
                def extract_n_value(text):
                    pattern = re.search(r"AlignedIS_Reweight\(n=(\d+)", text)
                    if pattern:
                        return int(pattern.group(1))
                    return None
                
                cur_n=extract_n_value(wp_str)
                assert cur_n is not None
                scores, label_attention_mask = get_cluster_split_res_id(
                    vocab_size, output_ids, wp, device, la_wp=la_wp, eps=eps, split_num=cur_n
                )
            
                assert label_attention_mask.shape[0] == 1
                label_attention_mask[0, :2] = 0
                
                scores = scores * label_attention_mask
                raw_scores = scores.sum(dim=-1)
                seq_len = torch.sum(label_attention_mask, dim=-1, keepdim=False)
                
                rq.put(
                    {
                        **batch,
                        "which_output": [output_kind],
                        "lens": seq_len.cpu().tolist(),
                        "raw_scores": raw_scores.cpu().tolist()
                    }
                )
                
            elif "John" in wp_str:
                wp = eval(wp_str)
                scores, label_attention_mask = get_green_token_scores_id(vocab_size,output_ids, wp, device, eps=eps)
                
                assert label_attention_mask.shape[0] == 1
                label_attention_mask[0, :2] = 0 

                scores = scores * label_attention_mask
                seq_len = torch.sum(label_attention_mask, dim=-1, keepdim=False)
                raw_scores = scores.sum(dim=-1)
                rq.put(
                    {
                        **batch,
                        "which_output": [output_kind],
                        "lens": seq_len.cpu().tolist(),
                        "raw_scores": raw_scores.cpu().tolist()
                    }
                )
            elif 'SynthID_Text' in wp_str:
                wp = eval(wp_str)
                wp.reset_watermark_key(len(batch["watermark_processor"]))
                wp.ignore_history = True
                scores, label_attention_mask = get_synthid_text_scores_id(vocab_size,output_ids, wp, device, eps=eps)
                
                assert label_attention_mask.shape[0] == 1
                label_attention_mask[0, :2] = 0 # for fair comparision

                scores = scores * label_attention_mask
                seq_len = torch.sum(label_attention_mask, dim=-1, keepdim=False)
                raw_scores = scores.sum(dim=-1)
                rq.put(
                    {
                        **batch,
                        "which_output": [output_kind],
                        "lens": seq_len.cpu().tolist(),
                        "raw_scores": raw_scores.cpu().tolist()
                    }
                )
                
                
                
            elif ('ITS_edit' in wp_str) or ('EXP_edit' in wp_str):
                wp = eval(wp_str)
                

                if 'ITS_edit' in wp_str:
                    gamma=0.4
                elif 'EXP_edit' in wp_str:
                    gamma=0.0
                else:
                    print('Unknown wp_str: ',wp_str)
                    exit(1)
                
                p_val, label_attention_mask = get_p_val_id(
                    vocab_size, output_ids, wp, device, la_wp=la_wp, eps=eps,gamma=gamma
                )
                assert label_attention_mask.shape[0] == 1
                # label_attention_mask[0, :5] = 0 # for fair comparision

                # scores = scores * label_attention_mask
                seq_len = torch.sum(label_attention_mask, dim=-1, keepdim=False)
                
                rq.put(
                    {
                        **batch,
                        "which_output": [output_kind],
                        "lens": seq_len.cpu().tolist(),
                        "p_val": p_val.view(1).cpu().tolist()
                    }
                )  
                
            elif 'Unigram' in wp_str:
                wp = eval(wp_str)
                wp.reset_watermark_key(len(batch["watermark_processor"]))
                wp.ignore_history = True

                scores, label_attention_mask = get_unigram_score_id(
                    vocab_size, output_ids, wp, device, la_wp=la_wp, eps=eps
                )
                
                assert label_attention_mask.shape[0] == 1
                label_attention_mask[0, :2] = 0 # for fair comparision

                scores = scores * label_attention_mask
                seq_len = torch.sum(label_attention_mask, dim=-1, keepdim=False)
                raw_scores = scores.sum(dim=-1)
                rq.put(
                    {
                        **batch,
                        "which_output": [output_kind],
                        "lens": seq_len.cpu().tolist(),
                        "raw_scores": raw_scores.cpu().tolist()
                    }
                )  

            elif ("Beta" in wp_str) or ("Dip_" in wp_str):

                wp = eval(wp_str)
                wp.reset_watermark_key(len(batch["watermark_processor"]))
                wp.ignore_history = True
                
                quantiles, label_attention_mask = get_quantile_id(
                    vocab_size, output_ids, wp, device, la_wp=la_wp, eps=eps
                )
                assert label_attention_mask.shape[0] == 1
                
                label_attention_mask[0, :2] = 0

                quantiles = quantiles * label_attention_mask
                cum_label_attention_mask = torch.cumsum(label_attention_mask, dim=-1)
                lens = cum_label_attention_mask[:, -1]

                raw_score, expected_value = score_func(quantiles, lens, mode="test")
                seq_len = torch.sum(label_attention_mask, dim=-1, keepdim=False)

                assert torch.all(seq_len == lens)
                final_score = (raw_score - expected_value) / torch.sqrt(seq_len)

                rq.put(
                    {
                        **batch,
                        "which_output": [output_kind],
                        "lens": lens.cpu().tolist(),
                        "beta_score": final_score.cpu().tolist(),
                    }
                )
            elif wp_str=='None' or 'WatermarkLogitsProcessor_Baseline' in wp_str:
                rq.put(
                    {
                        **batch,
                        "which_output": [output_kind],
                    }
                )
                
            else:
                print(f"Unknown Watermark Processor: {wp_str}")
                raise NotImplementedError

