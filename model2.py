import gensim
from gensim import downloader
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import f1_score
from torch.optim import Adam
import FFnn

def load_data(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    sentence = []
    all_tags = []
    sentence_tags = []
    words = []
    word_to_label = {}

    for line in lines:
        if line == '\n':
            sentences.append(sentence)
            all_tags.append(sentence_tags)
            sentence = []
            sentence_tags = []
        else:
            if len(line.split('\t')) > 1:
                word, tag = line.split('\t')
                word = word.lower()
                tag = "0" if tag[:-1] == "O" else "1"
                sentence.append(word)
                sentence_tags.append(tag)
                words.append(word)
                word_to_label[word] = tag

    return sentences, all_tags, words, word_to_label


def build_set(words, model, word_to_label, length):
    train_set = []
    train_tags = []
    not_appeared_positive = []
    not_appeared_negative = []
    avg_positive_vec = np.zeros(length)
    avg_negative_vec = np.zeros(length)
    num_known_positive = 0
    num_known_negative = 0
    for word in words:
        if word not in model.key_to_index:
            if word_to_label[word] == '1':
                not_appeared_positive.append(word)
            else:
                not_appeared_negative.append(word)
        else:
            word_vec = model[word]

            if word_to_label[word] == '1':
                num_known_positive += 1
                avg_positive_vec = [sum(x) for x in zip(avg_positive_vec, word_vec)]
            else:
                num_known_negative += 1
                avg_negative_vec = [sum(x) for x in zip(avg_negative_vec, word_vec)]

            train_set.append(word_vec)
            train_tags.append(word_to_label[word])

    for word in not_appeared_positive:
        train_set.append([x / num_known_positive for x in avg_positive_vec])
        train_tags.append('1')
    for word in not_appeared_negative:
        train_set.append([x / num_known_negative for x in avg_negative_vec])
        train_tags.append('0')

    return train_set, train_tags


def train_and_predict(model, k, length):
    # Load training data
    sentences, all_tags, words, word_to_label = load_data('train.tagged')
    train_all_words = set(words)
    train_set, train_tags = build_set(train_all_words, model, word_to_label, length)

    # Load test data
    test_sentences, test_tags, test_words, test_word_to_label = load_data('dev.tagged')
    test_all_words = set(test_words)
    test_set, test_tags = build_set(test_all_words, model, test_word_to_label, length)

    datasets = {'train': (train_set, train_tags), 'test': (test_set, test_tags)}
    # train model with FF neural network

    # predict
    predictions = 0

    # calculate F1 score
    f1_score_value = f1_score(test_tags, predictions, average='binary', pos_label='1')
    print(f"f1 score: {f1_score_value}")


def main():
    pass


if __name__ == '__main__':
    main()