# 📜 Mitigating Biases in Hate Speech Detection from A Causal Perspective

🔍 This is the official code and data repository for the EMNLP 2023 findings paper titled "Mitigating Biases in Hate Speech Detection from A Causal Perspective".

## 🌟 Abstract

<details><summary>Abstract</summary>

Nowadays, many hate speech detectors are built to automatically detect hateful content. However, their training sets are sometimes skewed towards certain stereotypes (e.g., race or religion-related). As a result, the detectors are prone to depend on some shortcuts for predictions. Previous works mainly focus on token-level analysis and heavily rely on human experts' annotations to identify spurious correlations, which is not only costly but also incapable of discovering higher-level artifacts. In this work, we use grammar induction to find grammar patterns for hate speech and analyze this phenomenon from a causal perspective. Concretely, we categorize and verify different biases based on their spuriousness and influence on the model prediction. Then, we propose two mitigation approaches including Multi-Task Intervention and Data-Specific Intervention based on these confounders.
Experiments conducted on 9 hate speech datasets demonstrate the effectiveness of our approaches.

</details>

## Folder Structure

