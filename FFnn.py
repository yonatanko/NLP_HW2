from torch import nn
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import model2
from gensim import downloader
import numpy as np
import re
from torch.optim import Adam


class NerNN(nn.Module):
    def __init__(self, vec_dim, num_classes, hidden_dim=3):
        super(NerNN, self).__init__()
        self.first_layer = nn.Linear(vec_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.Tanh()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        x = self.first_layer(input_ids)
        x = self.activation(x)
        x = self.second_layer(x)
        if labels is None:
            return x, None
        loss = self.loss(x, labels)
        return x, loss


class NerDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, model, model_name):
        self.file_path = file_path
        self.sentences, self.labels, words, word_to_label= model2.load_data(file_path)
        self.labels = [item for sublist in self.labels for item in sublist]
        self.tags_to_idx = {tag: idx for idx, tag in enumerate(sorted(list(set(self.labels))))}
        self.idx_to_tags = {idx: tag for idx, tag in self.tags_to_idx.items()}
        representation, labels = [], []
        for sentence, label in zip(self.sentences, self.labels):
            cur_rep = []
            for word in sentence:
                if word not in model.key_to_index:
                    continue
                vec = model[word]
                cur_rep.append(vec)
            if len(cur_rep) == 0:
                continue
            cur_rep = np.stack(cur_rep).mean(axis=0)
            representation.append(cur_rep)
            labels.append(label)
        self.labels = labels
        representation = np.stack(representation)
        self.tokenized_sen = representation
        self.vector_dim = int(re.findall(r'\d+', model_name)[-1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        cur_sen = self.tokenized_sen[idx]
        cur_sen = torch.FloatTensor(cur_sen.squeeze())
        label = self.labels[idx]
        label = self.tags_to_idx[label]
        return {"input_ids": cur_sen, "labels": label}


def train(model, data_sets, optimizer, num_epochs: int, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True, num_workers=4),
                    "dev": DataLoader(data_sets["dev"], batch_size=batch_size, shuffle=False, num_workers=4)}

    model.to(device)
    max_f1 = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} of {num_epochs}")
        print("-" * 10)
        for phase in ["train", "dev"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            labels, preds = [], []
            for batch in data_loaders[phase]:
                batch_size = 0
                for k, v in batch.items():
                    batch[k] = v.to(device)
                    batch_size = v.shape[0]
                optimizer.zero_grad()
                if phase == "train":
                    outputs, loss = model(**batch)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs, loss = model(**batch)

                pred = outputs.argmax(dim=-1).clone().detach().cpu()
                labels += batch["labels"].cpu().view(-1).tolist()
                preds += pred.view(-1).tolist()
                running_loss += loss.item() * batch_size

            epoch_loss = running_loss / len(data_sets[phase])
            # why f1 score is 0? because the labels are not balanced (only 1% of the labels are 1) so the model predicts 0 all the time and the f1 score is 0 (because the recall is 0) and the accuracy is 0.99 (because the model predicts 0 all the time) so the model is not good at all (because it predicts 0 all the time)
            epoch_f1 = f1_score(labels, preds)
            epoch_accuracy = accuracy_score(labels, preds)

            print(f"{phase} Loss: {epoch_loss}, F1: {epoch_f1}, accuracy: {epoch_accuracy}")

            # update max f1 score
            if phase == "dev" and epoch_f1 > max_f1:
                max_f1 = epoch_f1
                with open("model.pkl", "wb") as f:
                    torch.save(model, f)
        print()

    print(f"Max F1: {max_f1:.4f}")


def main():
    for model_name in ["glove-wiki-gigaword-50", "glove-wiki-gigaword-100", "glove-wiki-gigaword-200",
                       "word2vec-google-news-300", "glove-twitter-25", "glove-twitter-50", "glove-twitter-100"]:
        model = downloader.load(model_name)
        print(f"Model: {model_name}")
        train_set = NerDataset("train.tagged", None, model, model_name)
        test_set = NerDataset("dev.tagged", None, model, model_name)
        nn_model = NerNN(num_classes=2, vec_dim=train_set.vector_dim)
        optimizer = Adam(params=nn_model.parameters())
        datasets = {"train": train_set, "dev": test_set}
        train(model=nn_model, data_sets=datasets, optimizer=optimizer, num_epochs=5)
        print()

if __name__ == "__main__":
    main()
