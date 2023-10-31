import numpy as np
import argparse

def bias(text_train, label_train, tokenizer, token_or_word='word', top_n=100, hate_label=1):
    """
    Calculate the top biased words in a dataset based on Pointwise Mutual Information (PMI).

    :param text_train: list of texts
    :param label_train: corresponding labels (0 for non-hate, 1 for hate)
    :param tokenizer: tokenizer with a tokenize method
    :param token_or_word: whether to tokenize by word or using a provided tokenizer ('word' or 'token')
    :param top_n: number of top biased words to return
    :param hate_label: label considered as 'hate'
    :return: list of top_n biased words
    """

    # Tokenize or split texts based on the choice
    if token_or_word == 'word':
        processed_texts = [item.split(' ') for item in text_train]
    elif token_or_word == 'token':
        processed_texts = [tokenizer.tokenize(item) for item in text_train]
    else:
        raise ValueError("token_or_word must be either 'word' or 'token'")

    # Initialize dictionaries
    dic_word_hate, dic_word_non_hate, dic_word = {}, {}, {}
    dic_class = {0: 0, 1: 0}

    # Populate dictionaries based on labels and words
    for index, words in enumerate(processed_texts):
        label = label_train[index]
        dic_class[label] += 1
        
        for word in words:
            # Increment general word dictionary
            dic_word[word] = dic_word.get(word, 0) + 1

            if label == hate_label:
                dic_word_hate[(word, label)] = dic_word_hate.get((word, label), 0) + 1
            else:
                dic_word_non_hate[(word, label)] = dic_word_non_hate.get((word, label), 0) + 1

    # Compute total counts for normalization purposes
    sum_value_hate = sum(dic_word_hate.values())
    sum_value_non_hate = sum(dic_word_non_hate.values())
    sum_word = sum(dic_word.values()) + 100

    # Calculate PMI for words in hate-labeled texts
    final_dic_hate = {}
    hate_portion = (dic_class[1] + 100) / (dic_class[1] + dic_class[0] + 100)
    
    for (word, cl), value in dic_word_hate.items():
        if value > 5:
            value += 100
            PMI = np.log2((value / (sum_value_hate + sum_value_non_hate + 200)) /
                          (((dic_word[word] + 100) / sum_word) * hate_portion))
            
            # Only store positive PMI values
            if PMI > 0:
                final_dic_hate[word] = PMI

    # Get the top biased words
    sorted_PMI_hate = sorted(final_dic_hate.items(), key=lambda x: x[1], reverse=True)
    biased_words = [pair[0] for pair in sorted_PMI_hate[:top_n]]

    return biased_words


def main():
    parser = argparse.ArgumentParser(description='Calculate biased words based on PMI.')
    parser.add_argument('data_file', type=str, help='Path to the data file with label and content.')
    parser.add_argument('tokenizer', type=str, help='Tokenizer choice.')
    args = parser.parse_args()

    # Read the file and split label and content
    with open(args.data_file, 'r') as file:
        lines = file.readlines()
        label_train = [int(line.split('\t')[0]) for line in lines]
        text_train = [line.split('\t')[1].strip() for line in lines]

    biased_words = bias(text_train, label_train, args.tokenizer, "word")
    print("Biased Words:", biased_words)

if __name__ == '__main__':
    main()