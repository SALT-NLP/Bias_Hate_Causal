# 📜 Mitigating Biases in Hate Speech Detection from A Causal Perspective

🔍 This is the official code and data repository for the EMNLP 2023 findings paper titled "Mitigating Biases in Hate Speech Detection from A Causal Perspective".
Authors: [Zhehao Zhang](https://zzh-sjtu.github.io/zhehaozhang.github.io/), [Jiaao Chen](https://cs.stanford.edu/people/jiaaoc/), [Diyi Yang](https://cs.stanford.edu/~diyiy/)

## 🌟 Abstract

<details><summary>Abstract</summary>

Nowadays, many hate speech detectors are built to automatically detect hateful content. However, their training sets are sometimes skewed towards certain stereotypes (e.g., race or religion-related). As a result, the detectors are prone to depend on some shortcuts for predictions. Previous works mainly focus on token-level analysis and heavily rely on human experts' annotations to identify spurious correlations, which is not only costly but also incapable of discovering higher-level artifacts. In this work, we use grammar induction to find grammar patterns for hate speech and analyze this phenomenon from a causal perspective. Concretely, we categorize and verify different biases based on their spuriousness and influence on the model prediction. Then, we propose two mitigation approaches including Multi-Task Intervention and Data-Specific Intervention based on these confounders.
Experiments conducted on 9 hate speech datasets demonstrate the effectiveness of our approaches.

</details>

## 📂 Folder Structure

- Folder Descriptions

## 📊 Data Description

All datasets are sourced from [hatespeechdata.com](https://hatespeechdata.com/).

📥 **Download Data**:
- **Multimodal Meme Dataset**: [Download](https://github.com/bharathichezhiyan/Multimodal-Meme-Classification-Identifying-Offensive-Content-in-Image-and-Text)
- **Hate Speech Dataset from a White Supremacist Forum**: [Download](https://github.com/Vicomtech/hate-speech-dataset)
- **Fox News User Comments**: [Download](https://github.com/sjtuprog/fox-news-comments)
- **HASOC19**: [Download](https://hasocfire.github.io/hasoc/2019/dataset.html)
- **ETHOS**: [Download](ETHOS)
- **Twitter Sentiment Analysis**: [Download](https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech)
- **Anatomy of Online Hate**: [Download](https://www.dropbox.com/s/21wtzy9arc5skr8/ICWSM18%20-%20SALMINEN%20ET%20AL.xlsx?dl=0)
- **Hate Speech on Twitter**: [Download](https://github.com/ziqizhang/data)
- **Hate-Offensive (AHS)**: [Download](https://github.com/t-davidson/hate-speech-and-offensive-language#automated-hate-speech-detection-and-the-problem-of-offensive-language)
## 🚀 How to Run

### 🧠 Multi-Task Intervention (MTI)
- For the Multi-Task Intervention, we utilize two methods: masked language modeling (MLM) and multi-task learning (MTL). The implementation of MLM is adopted from the code provided in the [Hugging Face documentation on Masked Language Modeling](https://huggingface.co/docs/transformers/tasks/masked_language_modeling).
#### Running the Masked Language Modeling

To execute the script, use the command below:
```
python masked_LM.py [arguments]
```
##### Arguments

- `--device`: Specifies the device to use (e.g., 'cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
- `--model_checkpoint`: The model checkpoint for initialization. Default value is 'bert-base-cased'.
- `--output_dir`: The directory where the trained model will be saved. Default is set to 'MLM_large_corpus'.
- `--evaluation_strategy`: Specifies the evaluation strategy. The default strategy is 'epoch'.
- `--learning_rate`: Learning rate for training. Default value is 2e-5.
- `--weight_decay`: Weight decay during training. Default value is 0.01.
- `--train_batch_size`: Size of the batch during training. Default value is 64.
- `--eval_batch_size`: Batch size during evaluation. Default is set to 64.
- `--fp16`: Flag to enable half precision training. Set to False by default.
- `--datasets`: Comma-separated list of dataset names to be used. This argument is mandatory.
- `--logging_divisor`: Determines the logging frequency. Logs will be generated every `len(train_data) // logging_divisor` steps. The default value is 4.
- `--num_epochs`: Number of epochs for training. Default value is 5.
- `--save_strategy`: Strategy employed for model saving. Default is 'epoch'.


### 🌐 Data-Specific Intervention (DSI)
- Instructions for DSL

### ❗ Hate Speech Detection
- Instructions for Hate Speech Detection
