# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the FAIR Noncommercial Research License
# found in the LICENSE file in the root directory of this source tree.

import json
import os
import re
import torch
from typing import List

from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

from .utils.speech2unit.speech2unit import Speech2Unit


NAME = "SpeechGPT"
META_INSTRUCTION = "You are an AI assistant whose name is SpeechGPT.\n- SpeechGPT is a intrinsic cross-modal conversational language model that is developed by Fudan University.  SpeechGPT can understand and communicate fluently with human through speech or text chosen by the user.\n- It can perceive cross-modal inputs and generate cross-modal outputs.\n"


def extract_text_between_tags(text, tag1='[SpeechGPT] :', tag2='<eoa>'):
    pattern = f'{re.escape(tag1)}(.*?){re.escape(tag2)}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        response = match.group(1)
    else:
        response = ""
    return response


class SpeechGPT:
    def __init__(
        self,
        model_name_or_path: str,
        lora_weights: str=None,
        device="cpu",
        s2u_dir: str="models/checkpoints/",
        vocoder_dir: str="models/speechgpt/utils/vocoder"
    ):
        self.meta_instruction = META_INSTRUCTION
        self.template= "[Human]: {question} <eoh>. [SpeechGPT]: "

        self.device = device

        # Speech2unit
        self.s2u = Speech2Unit(ckpt_dir=s2u_dir)

        # Model
        model_name_or_path = os.path.join(os.getcwd(), "models/checkpoints", model_name_or_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        if lora_weights is None:
            lora_weights = model_name_or_path.replace("7B-cm", "7B-com")

        self.model = PeftModel.from_pretrained(
            self.model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.model.half()
        self.model.eval()

        # Tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left"

        # Vocoder
        with open(os.path.join(vocoder_dir, "config.json")) as f:
            vocoder_cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoder(
            os.path.join(vocoder_dir, "vocoder.pt"),
            vocoder_cfg
        ).to(self.device)

        self.non_speech_token_ids = [[x] for x in range(3, 32000) if x not in [584, 29871]]


    def preprocess(
        self,
        raw_text: str,
    ):
        processed_parts = []
        for part in raw_text.split("is input:"):
            if os.path.isfile(part.strip()) and os.path.splitext(part.strip())[-1] in [".wav", ".flac", ".mp4"]:
                processed_parts.append(self.s2u(part.strip(), merged=True))
            else:
                processed_parts.append(part)
        processed_text = "is input:".join(processed_parts)

        prompt_seq = self.meta_instruction + self.template.format(question=processed_text)
        return prompt_seq


    def postprocess(
        self,
        response: str,
    ):
        question = extract_text_between_tags(response, tag1="[Human]", tag2="<eoh>")
        answer = extract_text_between_tags(response + '<eoa>', tag1=f"[SpeechGPT] :", tag2="<eoa>")
        tq = extract_text_between_tags(response, tag1="[SpeechGPT] :", tag2="; [ta]") if "[ta]" in response else ''
        ta = extract_text_between_tags(response, tag1="[ta]", tag2="; [ua]") if "[ta]" in response else ''
        ua = extract_text_between_tags(response + '<eoa>', tag1="[ua]", tag2="<eoa>") if "[ua]" in response else ''
        if "<sosp>" in response:
            speech = extract_text_between_tags(response, tag1="<sosp>", tag2="<eosp>")
        else:
            speech = " ".join([num for num in re.findall(r'(<\d+>)', response)]).replace("<sosp>", "")
        return {"question": question, "answer": answer, "textQuestion": tq, "textAnswer": ta, "unitAnswer": ua, "speech": speech}


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


    def forward(
        self,
        prompts: List[str],
        enforce_speech=False,
        **kwargs
    ):
        with torch.no_grad():
            # Preprocess
            preprocessed_prompts = []
            for prompt in prompts:
                preprocessed_prompts.append(self.preprocess(prompt))

            input_ids = self.tokenizer(preprocessed_prompts, return_tensors="pt", padding=True).input_ids
            for input_id in input_ids:
                if input_id[-1] == 2:
                    input_id = input_id[:, :-1]
                if len(input_id) > 512:
                    input_id = input_id[:512]

            input_ids = input_ids.to(self.device)

            bad_words_ids = self.non_speech_token_ids if enforce_speech else None

            # Generate
            generated_ids = self.model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                bad_words_ids=bad_words_ids,
                **kwargs
            ).sequences

            # # Get token ids from <sosp> to <eosp> in the response
            # generated_answer_ids = [ids_list[(ids_list.index(33000) + 1):ids_list.index(33001)] for ids_list in generated_ids.cpu().tolist()]

            generated_answer_ids = generated_ids.cpu().tolist()

            # Decode response and postprocess
            raw_responses = self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
            responses = [self.postprocess(x) for x in raw_responses]
            assert len(responses) == 1, "Assumed unbatched generation."

            # Create wav
            wav = None
            x = {"code": torch.tensor([0])}

            generated_answer_ids = []

            for _, response in enumerate(responses):
                if response["speech"] != '':
                    unit = [int(num) for num in re.findall(r'<(\d+)>', response["speech"])]
                    generated_answer_ids.append([32000 + u for u in unit])
                    x = {"code": torch.LongTensor(unit).view(1, -1).to(self.device)}
                    wav = self.vocoder(x, True).detach().cpu()

            if wav is None:
                print("Bad generation:", generated_ids, raw_responses)

        return generated_answer_ids, x["code"].squeeze().cpu(), wav
    
    def get_audio_embeddings(self):
        return self.vocoder.model.dict.weight.detach()
