# Port-LLM
Yali Zhang, Haifan Yin, Weidong Li, Emil Björnson, Mérouane Debbah, "Port-LLM: A Port Prediction Method for Fluid Antenna based on Large Language Models"，available online: [[paper](https://arxiv.org/abs/2502.09857)].

## Dependencies and Installation
* Python 3.9
* Pytorch 2.1.0
* NVIDIV GPU + CUDA 12.1
* Anaconda (conda 24.5.0)

## Dataset
The dataset used in this code is derived from the CDL-D channel model. Comprehensive information regarding the parameter configurations for dataset generation is available in [our paper](https://arxiv.org/abs/2502.09857).

## Get started
* The code for model training is in the `train`, while the code utilized for testing the model across multiple antennas at the base station can be found in the `Multiantenna_test`. `other_NN_based_models` contains the other neural network-based models compared in our paper and `plot_figs` includes some data file processing and graphing code utilized in our research.
* The trained model obtained by utilizing the codes in `train` is employed for the purposes of performance evaluation and comparative analysis in `Multiantenna_test` and `other_NN_based_models`.
* Please be advised that when utilizing this code, it is essential to modify the file paths within the code to correspond with your specific file locations.
* The pre-trained GPT-2 model utilized in the code is available for download from the official website [[Hugging Face](https://huggingface.co/models)].

## Attention!
* To reduce the number of our model parameters, we have further refined the Port-LLM model described in the original paper. The code for the improved Port-LLM model, based on LoRA fine-tuning, is presented in `Updata_code\model_lore.py`.
* To more fully leverage the powerful natural language processing capabilities of large language models and further compare the effectiveness of prompt fine-tuning versus LoRA fine-tuning in the models studied here, we propose a prompt-based fine-tuning model named Prompt-Port-LLM in the updated version of our paper. The specific implementation code for this model can be found at: `Updata_code\model_prompt.py`.

## Citation
If you find this repo helpful, please cite our paper.
```
@article{zhang2025portllm,
  title={Port-LLM: A Port Prediction Method for Fluid Antenna based on Large Language Models},
  author={Yali Zhang, Haifan Yin, Weidong Li, Emil Björnson, Mérouane Debbah},
  journal={arXiv preprint arXiv:2502.09857},
  year={2025}
}
```
