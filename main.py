import gensim
from gensim import downloader
import numpy as np
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


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
        if line == '\t\n':
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
                sentence.append(word)
                sentence_tags.append(tag)
                words.append(word)
                word_to_label[word] = tag

    return sentences, all_tags, words, word_to_label


def main():
    # Load training data
    sentences, all_tags, words, word_to_label = load_data('train.tagged')
    all_words = set(words)

    # define word2vec model
    model = Word2Vec(sentences=sentences, vector_size=10, window=2, min_count=1, workers=4)

    # generate train set
    train_set = []
    train_tags = []

    for word in all_words:
        if word not in model.wv.key_to_index:
            word_vec = np.random.rand(10)
        else:
            word_vec = model.wv[word]
        train_set.append(word_vec)
        train_tags.append(word_to_label[word])

    # train knn
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_set, train_tags)

    # load test data
    test_sentences, test_tags, test_words, test_word_to_label = load_data('dev.tagged')
    test_all_words = set(test_words)

    # generate test set
    test_set = []
    test_tags = []

    for word in test_all_words:
        if word not in model.wv.key_to_index:
            word_vec = np.random.rand(10)
        else:
            word_vec = model.wv[word]

        test_set.append(word_vec)
        test_tags.append(test_word_to_label[word])

    # predict
    predictions = knn.predict(test_set)
    print(set(predictions))
    print(set(test_tags))
    # calculate F1 score
    f1_score_value = f1_score(test_tags, predictions, average='binary', pos_label='1')
    print(f"f1 score: {f1_score_value}")


if __name__ == '__main__':
    main()