import os
import re
import json
import torch
import glob
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from transformers import EncodecModel, AutoProcessor

import models.unit_to_token_map as token_map

from .attacks.AudioMarkBench.no_box.nobox_audioseal_librispeech import (
    pert_time_stretch,
    pert_Gaussian_noise,
    pert_background_noise,
    pert_opus,
    pert_encodec,
    pert_quantization,
    pert_highpass,
    pert_lowpass,
    pert_smooth,
    pert_echo,
    pert_mp3
)

def all_attacks_auto(   
    wav_id=0,     
    waveform=None,
    model_str="",
    wav_dir="",
    attack_configs="",
    sr=16000,
    speech_tokenizer=None,
    tokenizer=None,
    model_encodec=None, 
    processor_encodec=None,
    watermark_processor=None
):
    with open(attack_configs) as f:
        attack_configs = json.load(f)

    # Mapping attack names to corresponding functions
    attack_functions = {
        "time_dilation": pert_time_stretch,
        "gaussian_noise": pert_Gaussian_noise,
        "background_noise": pert_background_noise,
        "encodec": lambda waveform, magnitude: pert_encodec(waveform, 24000, magnitude, model_encodec, processor_encodec),
        "quantization": pert_quantization,
        "highpass": pert_highpass,
        "lowpass": pert_lowpass,
        "smooth": pert_smooth,
        "echo": pert_echo,
        "opus": lambda waveform, magnitude: pert_opus(waveform, bitrate=1000 * magnitude, quality=1, cache=wav_dir),
        "mp3": pert_mp3
    }

    for attack_type, params in attack_configs.items():
        param_name, magnitudes = list(params.items())[0]
        
        for magnitude in magnitudes:
            wav_subdir = os.path.join(wav_dir, f"{attack_type}_{magnitude}")
            os.makedirs(wav_subdir, exist_ok=True)

            attack_results = []

            pert_wav_filename = os.path.join(wav_subdir, f"{wav_id}_{magnitude}.wav")
            if os.path.exists(pert_wav_filename):
                continue

            attack_func = attack_functions[attack_type]

            if attack_type in ["time_dilation"]:
                waveform_pert, _ = attack_func(waveform, sr, magnitude)
            elif attack_type in ["highpass", "lowpass"]:
                waveform_pert = attack_func(waveform, magnitude, sr)
            else:
                waveform_pert = attack_func(waveform, magnitude)

            waveform_pert = waveform_pert.squeeze()
            sf.write(pert_wav_filename, waveform_pert, 16000)

            if "spirit-lm" in model_str:
                retokenized_output = speech_tokenizer(waveform_pert)
            elif "SpeechGPT" in model_str:
                retokenized_output = speech_tokenizer.from_wav(waveform_pert)

            retokenized_output_ids = tokenizer(retokenized_output).input_ids

            attack_results.append({
                "id": wav_id,
                "attack_type": attack_type,
                "strength": f"{param_name}_{magnitude}",
                "output": retokenized_output_ids,
                "watermark_processor": watermark_processor
            })
        
            results_file = os.path.join(wav_dir, f"{attack_type}_{param_name}_{magnitude}.txt")
            with open(results_file, "a") as f:
                for item in attack_results:
                    f.write(json.dumps(item) + "\n")


def call_vocoders(units, model_str, vocoder, device):
    if "spirit-lm" in model_str:
        wav = torch.tensor(vocoder(units))

    elif "SpeechGPT" in model_str:
        unit = [int(num) for num in re.findall(r"<(\d+)>", units)]
        x = {"code": torch.LongTensor(unit).view(1, -1).to(device)}
        wav = vocoder(x, True).detach().cpu()

    return wav


def wp2name(wp):
    def extract_n_value(text):
        pattern = re.search(r"AlignedIS_Reweight\(n=(\d+)", text)
        if pattern:
            return int(pattern.group(1))
        raise NotImplementedError

    if 'John' in wp:
        if 'delta=0.5' in wp:
            return 'KGW_0_5'
        elif 'delta=1.0' in wp:
            return 'KGW_1_0'
        elif 'delta=1.5' in wp:
            return 'KGW_1_5'
        elif 'delta=2.0' in wp:
            return 'KGW_2_0'
        else:
            raise NotImplementedError
    elif 'Unigram' in wp:
        if 'delta=0.5' in wp:
            return 'Unigram_0_5'
        elif 'delta=1.0' in wp:
            return 'Unigram_1_0'
        elif 'delta=1.5' in wp:
            return 'Unigram_1_5'
        elif 'delta=2.0' in wp:
            return 'Unigram_2_0'
        else:
            raise NotImplementedError
        
    elif 'Dip_Reweight' in wp:
        if 'alpha=0.5' in wp:
            return 'Dipmark_0_5'
        elif 'alpha=0.4' in wp:
            return 'Dipmark_0_4'
        elif 'alpha=0.3' in wp:
            return 'Dipmark_0_3'
        else:
            raise NotImplementedError
    elif 'STA' in wp:
        return 'STA'
    elif 'ITS' in wp:
        return 'ITS'
    elif '_Baseline' in wp:
        return 'Baseline'
    elif wp=='None':
        return 'None'
    elif "Aligned" in wp:
        return f"AlignedIS_{extract_n_value(wp)}"
    else:
        raise NotADirectoryError


def check_files_equal_lines(file1, file2):
    if not os.path.exists(file1) or not os.path.exists(file2):
        return False

    with open(file1) as f1, open(file2) as f2:
        lines1 = sum(1 for _ in f1)
        lines2 = sum(1 for _ in f2)

    return lines1 == lines2


def attack(args, base_dir):
    # Read results file
    generations = pd.read_json(
        open(args.generations_file, encoding="utf8"), lines=True
    )

    wm_processors = {wp: wp2name(wp) for wp in generations["watermark_processor"].unique()}

    wav_dirs = []

    with torch.no_grad():
        for wp, wp_name in wm_processors.items():
            current_df = generations[generations["watermark_processor"] == wp]
            wp_wav_dir = os.path.join(base_dir, wp_name)
            wav_dirs.append(wp_wav_dir)

            wav_subdir = os.path.join(wp_wav_dir, "clean")
            os.makedirs(wav_subdir, exist_ok=True)

            for generation in tqdm(current_df.itertuples(index=False)):
                wav_id = generation.id
                token_sequence = generation.raw_output

                try:
                    filename = os.path.join(wav_subdir, f"{wav_id}.wav")
                    if os.path.exists(filename):
                        continue

                    if "spirit" in args.model_str:
                        # TODO: clip in generation
                        units = token_map.token_to_unit_spiritlm(token_sequence[2:])
                    elif "SpeechGPT" in args.model_str:
                        # TODO remove 1
                        units = token_map.token_to_unit_speechgpt(token_sequence[1:])
                    wav = call_vocoders(units, args.model_str, vocoder, device)

                    sf.write(filename, wav, 16000)

                    all_attacks_auto(
                        wav_id=wav_id,
                        waveform=wav[None, ...],
                        model_str=args.model_str,
                        wav_dir=wp_wav_dir,
                        watermark_processor=generation.watermark_processor,
                        attack_configs=args.attack_configs,
                        speech_tokenizer=speech_tokenizer,
                        tokenizer=tokenizer,
                        model_encodec=model_encodec,
                        processor_encodec=processor_encodec
                    )
                except Exception as e:
                    print(e)

    return wav_dirs


def attack_posthoc(args, base_dir):
    wm_processors = ["audioseal_cuda", "wavmark"]

    wav_dirs = []

    for wm in wm_processors:
        wav_dirs += glob.glob(f"{base_dir}/attacks_new/{wm}")

        with torch.no_grad():
            for wav_dir in wav_dirs:

                wp_wav_dir = os.path.join(wav_dir, "attacks")

                for file in tqdm(os.listdir(wav_dir)):
                    if not file.endswith("wav"): continue

                    wav_id = file.replace(".wav", "")
                    wav, _ = sf.read(os.path.join(wav_dir, file))

                    wav = torch.tensor(wav, dtype=torch.float)

                    all_attacks_auto(
                        wav_id=wav_id,
                        waveform=wav[None, ...],
                        model_str=args.model_str,
                        wav_dir=wp_wav_dir,
                        watermark_processor=wm,
                        attack_configs=args.attack_configs,
                        speech_tokenizer=speech_tokenizer,
                        tokenizer=tokenizer,
                        model_encodec=model_encodec,
                        processor_encodec=processor_encodec
                    )

    return wav_dirs


if __name__ == "__main__":
    import argparse
    from experiments import audio_generation as ag
    from experiments.audio_generation.common import set_spawn

    set_spawn()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_str", type=str)
    parser.add_argument("--generations_file", type=str)
    parser.add_argument("--attack_configs", type=str, default="evaluations/configs/attack_base_configs.json")
    args = parser.parse_args()


    # Prepare saving directory for wav files
    base_dir = os.path.join(os.path.dirname(args.generations_file), "attacks")
    os.makedirs(base_dir, exist_ok=True)

    # if check_all_dirs(base_dir):
    #     print("Already done", base_dir)
    #     exit(0)
    # else:
    #     print("Something missing!")
    #     exit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the ENCODeC model and processor
    model_encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
    processor_encodec = AutoProcessor.from_pretrained("facebook/encodec_24khz")

    # Prepare model components
    if "spirit-lm" in args.model_str:
        from models.spiritlm.model.spiritlm_model import Spiritlm
        model = Spiritlm(args.model_str, device)
        speech_tokenizer = model.speech_tokenizer
        tokenizer = model.tokenizer
        vocoder = lambda x: speech_tokenizer.decode(x)
        del model

    elif "SpeechGPT" in args.model_str:
        from models.speechgpt import SpeechGPT
        model = SpeechGPT(args.model_str, device=device)
        speech_tokenizer = model.s2u
        tokenizer = model.tokenizer
        vocoder = model.vocoder
        del model

    # Run attacks
    wav_dirs = attack(args, base_dir)

    # wav_dirs = attack_posthoc(args, os.path.dirname(args.generations_file))

    # Trick to avoid it as an argument
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.generations_file))))


    for wav_dir in wav_dirs:

        for jsonl_file in os.listdir(wav_dir):
            if "scores" not in jsonl_file and jsonl_file.endswith("txt"):
                output_path = os.path.join(wav_dir, jsonl_file)
                score_save_path = output_path.replace(".txt", "_scores.txt")
                if not check_files_equal_lines(output_path, score_save_path):
                    print("Running score evaluation for:", output_path)
                    ag.evaluate_score.pipeline(
                        output_path=output_path, 
                        score_save_path=score_save_path,
                        eps=0,
                        model_str=args.model_str,
                        dataset_name=dataset_name
                    )
