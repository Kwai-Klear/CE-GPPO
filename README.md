# Klear-Reasoner: 8B Open-Reasoning Model with Gradient-Preserving RL  
*Advancing mathematical and programmatic reasoning through long CoT SFT + GPPO*

---

<!-- ## 🚀 Quick Links
| Resource | Link |
|---|---|
| 🤗 Model Hub | [Klear-Reasoner-8B](https://huggingface.co/klear-team/klear-reasoner-8b) |
| 📄 Technical Report | [arXiv:250x.xxxxx](https://arxiv.org/abs/250x.xxxxx) |
| 🐛 Issues & Discussions | [GitHub Issues](https://github.com/klear-team/klear-reasoner/issues) |
| 📧 Contact | klear-reasoner@kuaishou.com |

--- -->

## 📌 Overview
Klear-Reasoner-8B is an 8-billion-parameter reasoning model that achieves **state-of-the-art** performance on challenging **math and coding benchmarks**:

| Benchmark | AIME 2024 | AIME 2025 | LiveCodeBench V5 | LiveCodeBench V6 |
|---|---|---|---|---|
| **Score** | **90.5 %** | **83.2 %** | **66.0 %** | **58.1 %** |

The model combines:
1. **Quality-centric long CoT SFT** – distilled from DeepSeek-R1-0528.
2. **Gradient-Preserving Clipping Policy Optimization (GPPO)** – a novel RL method that **keeps gradients from clipped tokens** to boost exploration & convergence.

---

<!-- ## 🛠️ Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
``` -->

<!-- ### 2. Load Model & Generate
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "klear-team/klear-reasoner-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "Prove that for all positive integers n, n^3 + 2n is divisible by 3."
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=8192,
    temperature=0.6,
    top_p=0.95,
    do_sample=True
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

--- -->

## 📊 Benchmark Results (Pass@1)

| Model | AIME2024<br>avg@64 | AIME2025<br>avg@64 | HMMT2025<br>avg@64 | LCB V5<br>avg@8 | LCB V6<br>avg@8 |
|-------|--------------------|--------------------|--------------------|-----------------|-----------------|
| AReal-boba-RL-7B | 61.9 | 48.3 | 29.4 | 34.3 | 31.0† |
| MiMo-7B-RL | 68.2 | 55.4 | 35.7 | 57.8 | 49.3 |
| Skywork-OR1-7B | 70.2 | 54.6 | 35.7 | 47.6 | 42.7 |
| AceReason-Nemotron-1.1-7B | 72.6 | 64.8 | 42.9 | 57.2 | 52.1 |
| POLARIS-4B-Preview ♠ | 81.2 | _79.4_ | 58.7 | 58.5† | 53.0† |
| Qwen3-8B | 76.0 | 67.3 | 44.7† | 57.5 | 48.4† |
| Deepseek-R1-0528-Distill-8B ♣ | _86.0_ | 76.3 | 61.5 | 61.0† | 51.6† |
| OpenReasoning-Nemotron-7B ♣ | 84.7 | 78.2 | 63.5 | _65.6_† | _56.3_† |
| Klear-Reasoner-8B-SFT | 75.6 | 70.1 | 57.6 | 58.5 | 49.6 |
| Klear-Reasoner-8B | 83.2 | 75.6 | 60.3 | 61.6 | 53.1 |
| *w/ 64K Inference Budget* ♣ | **90.5** | **83.2** | **70.8** | **66.0** | **58.1** |

> We report the average `pass@1` results (avg@_n_), with all other evaluation metrics following the DeepSeek-R1 assessment framework (temperature=0.6, top_p=0.95).  
> LCB stands for LiveCodeBench. By default, we include official performance data provided by model developers when available.  
> Otherwise, † indicates our evaluation results based on the officially recommended configurations.  
> Models marked with ♠ indicate a maximum inference length of 96K tokens, while those with ♣ denote a 64K maximum inference length; all other models use a 32K inference length.  
> The best score on a given dataset is marked in **bold**, and the second best is _underlined_.


---

## 🧪 Reproducing the Results
We provide full training scripts & data configs:

```bash
git clone https://github.com/suu990901/Klear_Reasoner.git
cd Klear_Reasoner
pip install -r requirements.txt

# RL
bash recipe/dapo/perf_run_dapo_ours_code.sh

```

---

<!-- ## 🔍 Key Techniques
| Component | Description |
|---|---|
| **GPPO** | Gradient-Preserving Clipping Policy Optimization. Retains clipped-token gradients; stabilizes training while boosting exploration. |
| **Soft Reward** | For code tasks, reward = (passed tests / total tests) instead of binary 0/1. |
| **Zero-Advantage Filtering** | Removes prompt groups whose advantages are all zero → clearer gradients. |
| **YaRN Extension** | Extends context to 64 K tokens during inference for better long-CoT reasoning. | -->


<!-- --- -->

<!-- ## 🤝 Citation
```bibtex
@misc{klear2025reasoner,
  title={Klear-Reasoner: Advancing Reasoning Capability via Gradient-Preserving Clipping Policy Optimization},
  author={Su, Zhenpeng and Pan, Leiyu and Bai, Xue and Liu, Dening and Dong, Guanting and Huang, Jiaming and Hu, Wenping and Zhang, Fuzheng and Zhou, Guorui},
  year={2025},
  eprint={250x.xxxxx},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
``` -->

<!-- --- -->

<!-- ## 📄 License
Apache 2.0. See [LICENSE](LICENSE) for details.

--- -->
