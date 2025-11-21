# Robust Distortion-Free Watermark for Autoregressive Audio Generation Models (NeurIPS 2025)


[![Paper](https://img.shields.io/badge/arXiv-2312.06613-brightgreen)](https://arxiv.org/abs/2510.21115)
&nbsp; [![Project WebPage](https://img.shields.io/badge/Project-webpage-blue)](https://g-milis.github.io/audio_watermark.html)

Official implementation of Aligned-IS, a novel, robust, and distortion-free watermark, specifically crafted for autoregressive audio generation models. Discrete autoregressive models for contiunuous modalities suffer from retokenization mismatch, which other techniques mitigate by finetuning. Our method utilizes a clustering approach that treats tokens within the same cluster equivalently, effectively countering the retokenization mismatch issue in a training-free manner.


## Quick Start

### Installation

Our current setup has been verified with older libraries, but you can try any setup supporting `fairseq==0.12.2`, `transformers==4.31.0`, and CUDA.

Create an environment:
```
conda create -n aligned python=3.9 pip=24.0
conda activate aligned
pip install -r requirements.txt
```

### Datasets
We have left the Finance QA dataset in `datasets` for a quicker demo. For the rest:
1. Download [Dolly dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/main/databricks-dolly-15k.jsonl).
2. Get the full [WaterBench dataset](https://github.com/THU-KEG/WaterBench/tree/main/data/WaterBench).
3. For LibriSpeech, we used the dev subset found [here](https://www.openslr.org/12).

Bear in mind, you might need to verify the paths and loading logic for your custom datasets. All data loading and preprocessing happens in `experiments/audio_generation/generation_dataset.py`, whose functions are imported in `experiments/audio_generation/__init__.py`.


### Models
1. For SpiritLM, request permission to [download it](https://ai.meta.com/resources/models-and-libraries/spirit-lm-downloads/), for more information [check here](https://github.com/facebookresearch/spiritlm/blob/main/checkpoints/README.md).
2. For SpeechGPT, see the original [repo](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt).

Place all assets under `models/checkpoints` or a custom directory which you can soft link to `models/checkpoints` for ease of use with the current settings. To add more models, look into `models/clusters.py`. You need to save their text embedding layer and cluster them accordingly.


### Experiments
Run watermarking experiments:
```
./scripts/run_spiritlm_exp.sh
```
or
```
./scripts/run_speechgpt_exp.sh
```
Those scripts produce a generation log (the raw generated sequences as well as a retokenized version of them) as well as a scoring file containing the watermark score for each generation. To aggregate the results and get the detection rates, run (make sure to adjust the score path):
```
./scripts/run_evaluations.sh
```
You may also adjust the arguments in each script. 

Optionally, you can run robustness experiments, which takes as input the generation logs and perturbs the waveforms, then creates scoring files accordingly. You may need to inspect the attack configs argument:
```
./scripts/run_<model>_attacks.sh
```

## Methodology breakdown

The codebase is designed for running multiple experiments in parallel. You may isolate the watermarking and scoring code and adapt it to your workflow. 

The watermarking reweight code is in `watermarks/aligned.py`, which is wrapped in a `LogitsProcessor` class in `watermarks/transformers.py`, ensuring compatibility with huggingface `transformers` autoregressive models. You may inspect the actual usage with the models in `experiments/audio_generation/common.py:spiritlm_worker` and `experiments/audio_generation/common.py:speechgpt_worker`.

For detection, `watermarks/transformers.py:WatermarkLogitsProcessor.get_cluster_n_res` performs the counting of how many tokens belong to their cluster sampled at each step.


## Acknowledgements

The watermarking code is based on [DipMark](https://github.com/yihwu/DiPmark), and the attack suite is mostly from [AudioMarkBench](https://github.com/mileskuo42/AudioMarkBench).


## Citation

If you find this work useful for your research, please cite our paper (official NeurIPS citation not yet available):

```
@article{wu2025robust,
  title={Robust Distortion-Free Watermark for Autoregressive Audio Generation Models},
  author={Wu, Yihan and Milis, Georgios and Chen, Ruibo and Huang, Heng},
  journal={arXiv preprint arXiv:2510.21115},
  year={2025}
}
```
