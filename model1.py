from gensim import downloader
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import re

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
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
    set_data = []
    set_tags = []

    for word in words:
        if word not in model.key_to_index:
            word_vec = np.zeros(length)
        else:
            word_vec = model[word]

        set_data.append(word_vec)
        set_tags.append(word_to_label[word])

    return set_data, set_tags


def train_and_predict(model, k, length):
    # Load training data
    sentences, all_tags, words, word_to_label = load_data('train.tagged')
    train_all_words = set(words)

    # generate train set
    train_set, train_tags = build_set(train_all_words, model, word_to_label, length)
    # train knn
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_set, train_tags)

    # load test data
    test_sentences, test_tags, test_words, test_word_to_label = load_data('dev.tagged')
    test_all_words = set(test_words)

    # generate test set
    test_set, test_tags = build_set(test_all_words, model, test_word_to_label, length)

    # predict
    predictions = knn.predict(test_set)

    # calculate F1 score
    f1_score_value = f1_score(test_tags, predictions, average='binary', pos_label='1')
    print(f"f1 score: {f1_score_value}")


def main():
    for k in [1, 3, 5]:
        print(f"num of neighbors: {k}")

        for model_name in ['word2vec-google-news-300','glove-wiki-gigaword-50','glove-wiki-gigaword-100','glove-wiki-gigaword-200',
                           'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200']:
            print(f"model: {model_name}")
            # extract number from model name
            dim = int(re.findall(r'\d+', model_name)[-1])
            model = downloader.load(model_name)
            train_and_predict(model, k, dim)
            print()


if __name__ == '__main__':
    main()
