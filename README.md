# 📜 Mitigating Biases in Hate Speech Detection from A Causal Perspective

🔍 This is the official code and data repository for the EMNLP 2023 findings paper titled ["Mitigating Biases in Hate Speech Detection from A Causal Perspective"](https://aclanthology.org/2023.findings-emnlp.440/).
Authors: [Zhehao Zhang](https://zzh-sjtu.github.io/zhehaozhang.github.io/), [Jiaao Chen](https://cs.stanford.edu/people/jiaaoc/), [Diyi Yang](https://cs.stanford.edu/~diyiy/)

## 🌟 Abstract

<details><summary>Abstract</summary>

Nowadays, many hate speech detectors are built to automatically detect hateful content. However, their training sets are sometimes skewed towards certain stereotypes (e.g., race or religion-related). As a result, the detectors are prone to depend on some shortcuts for predictions. Previous works mainly focus on token-level analysis and heavily rely on human experts' annotations to identify spurious correlations, which is not only costly but also incapable of discovering higher-level artifacts. In this work, we use grammar induction to find grammar patterns for hate speech and analyze this phenomenon from a causal perspective. Concretely, we categorize and verify different biases based on their spuriousness and influence on the model prediction. Then, we propose two mitigation approaches including Multi-Task Intervention and Data-Specific Intervention based on these confounders.
Experiments conducted on 9 hate speech datasets demonstrate the effectiveness of our approaches.

</details>

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

#### Running the Multi-Task Learning

You can simply concatenate the 9 datasets above and randomly shuffle the large dataset before inputting it into the basic classification pipeline.


### 🌐 Data-Specific Intervention (DSI)
- To perform DSI, we should first mitigate two types of correlations described in Section 3.1: Token-level Spurious Correlations and Sentence-level Spurious Correlations,

#### Finding Token-level Spurious Correlations

```
python token_bias.py [arguments]
```
##### Arguments

- `--data_file`: Specifies the dataset file path.
- `--tokenizer`: The tokenizer that you want to use. We use the simple word tokenizer in this work.


#### Finding Sentence-level Spurious Correlations

To find the Sentence-level Spurious Correlations, we follow a previous work (Finding Dataset Shortcuts with Grammar Induction EMNLP2022) and utilize the code from the original GitHub repository at [ShortcutGrammar](https://github.com/princeton-nlp/ShortcutGrammar). 

To integrate this into your workflow:

1. **PCFG Grammar Induction**:
    - Begin by using the hate speech data to perform PCFG (Probabilistic Context-Free Grammar) grammar induction. This can be done using the scripts (src/run.py) provided in the repository above.

2. **Visualizing Sentence-level Biases**:
    - Once you have obtained the output grammar from the PCFG grammar induction, you can use the `sentence_bias.ipynb` notebook to visualize and analyze the sentence-level biases in the data.

After that you can generate counterfactual data according to the prompt design in A.4 using an LLM.

### Classification pipeline
- After combine the counterfactual data with the original dataset and get the model checkpoint after MTI, we can use the following code to do the hate speech detection (formulated as a binary classification problem).

#### Training the Hate Speech Detection Model

To train the hate speech detection model, use the following script:

```
python classification.py [arguments]
```

##### Arguments

- `--seed`: Seed for random number generation to ensure reproducibility. Default is 60.
- `--epoch`: Number of training epochs. Default is 3.
- `--bz`: Batch size for training and evaluation. Default is 10.
- `--data`: The dataset to be used. It should be the same as the folder name.
- `--lr`: Learning rate for the optimizer. Default is 2e-5.
- `--wd`: Weight decay for the optimizer. Default is 0.
- `--norm`: Max gradient norm for gradient clipping. Default is 0.8.
- `--gpu`: GPU device ID for CUDA_VISIBLE_DEVICES. Default is '6'.
- `--model_checkpoint`: Pretrained model checkpoint for initialization. Default is "bert-base-cased". You can change it to the model after MTI to get the best performance."

##### 🔧 Hyperparameter Tuning with Weights & Biases (wandb)

Utilize Weights & Biases for hyperparameter tuning by defining and running a wandb sweep in your script. Configure the sweep with your desired parameters, such as learning rate and batch size, and use wandb's dashboard to monitor and analyze the results. Detailed instructions and examples can be found in the [official wandb documentation](https://docs.wandb.ai/).


#### Evaluating the Model

Once the model is trained, you can evaluate its performance on the test set using the metrics like accuracy, micro-F1, and macro-F1 scores. The evaluation is automatically done at the end of the training script and the results are displayed.

#### Saving and Loading Models

- The trained model can be saved using PyTorch's `save_pretrained` method and can be loaded using `from_pretrained` for further analysis or deployment.

### 📝 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{zhang-etal-2023-mitigating,
    title = "Mitigating Biases in Hate Speech Detection from A Causal Perspective",
    author = "Zhang, Zhehao  and
      Chen, Jiaao  and
      Yang, Diyi",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.440",
    pages = "6610--6625",
    abstract = "Nowadays, many hate speech detectors are built to automatically detect hateful content. However, their training sets are sometimes skewed towards certain stereotypes (e.g., race or religion-related). As a result, the detectors are prone to depend on some shortcuts for predictions. Previous works mainly focus on token-level analysis and heavily rely on human experts{'} annotations to identify spurious correlations, which is not only costly but also incapable of discovering higher-level artifacts. In this work, we use grammar induction to find grammar patterns for hate speech and analyze this phenomenon from a causal perspective. Concretely, we categorize and verify different biases based on their spuriousness and influence on the model prediction. Then, we propose two mitigation approaches including Multi-Task Intervention and Data-Specific Intervention based on these confounders. Experiments conducted on 9 hate speech datasets demonstrate the effectiveness of our approaches.",
}
```
## ✉️ Contact Information

For any inquiries or further information regarding this project, feel free to reach out to the leading author:

- **Zhehao Zhang**: [zhehao.zhang.gr@dartmouth.edu](mailto:zhehao.zhang.gr@dartmouth.edu) | [Website](https://zzh-sjtu.github.io/zhehaozhang.github.io/)

We welcome questions, feedback, and collaboration requests!


