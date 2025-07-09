# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, model, word_embeddings):
        self.model = model
        self.word_embeddings = word_embeddings

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        # Get embeddings and average them
        # word_idxs = torch.tensor([self.word_embeddings.word_indexer.index_of(w) for w in ex_words])
        word_idxs = []
        for w in ex_words:
            idx = self.word_embeddings.word_indexer.index_of(w)
            if idx == -1:
                word_idxs.append(1)
            else:
                word_idxs.append(idx)
        word_idxs = torch.tensor(word_idxs).unsqueeze(0)

        with torch.no_grad():
            output = self.model(word_idxs)
            prediction = torch.argmax(output).item()
        return prediction

class DAN(nn.Module):
        def __init__(self, emb_layer, hidden_size, out_size):
            super(DAN, self).__init__()
            self.emb_layer = emb_layer
            self.hidden_size = hidden_size
            self.V = nn.Linear(self.emb_layer.embedding_dim, hidden_size)
            self.g = nn.Tanh() # or nn.ReLU()
            self.W = nn.Linear(hidden_size, out_size)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, batch_word_idxs):
            padded_word_idxs = torch.tensor(batch_word_idxs, dtype=torch.long)
            embeddings = self.emb_layer(padded_word_idxs)

            batch_embs = embeddings.mean(axis=1)

            # return self.softmax(self.W(self.g(self.V(batch_embs))))
            return self.W(self.g(self.V(batch_embs)))

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """

    # 1. Define a subclass of nn.Module that does your prediction. This should return a log-probability
    # distribution over class labels. Your module should take a list of word indices as input and embed them
    # using a nn.Embedding layer initialized appropriately.
    # 2. Compute your classification loss based on the prediction. In lecture, we saw using the negative log
    # probability of the correct label as the loss. You can do this directly, or you can use a built-in loss
    # function like NLLLoss or CrossEntropyLoss. Pay close attention to what these losses expect as
    # inputs (probabilities, log probabilities, or raw scores).
    # 3. Call network.zero grad() (zeroes out in-place gradient vectors), loss.backward (runs the
    # backward pass to compute gradients), and optimizer.step to update your parameters.
    
    # hyperparameters
    lr = 0.0005
    num_epochs = 4

    inp = word_embeddings.get_embedding_length()
    hid = 128
    if inp == 50:
        hid = 32
    
    out = 2

    emb_layer = word_embeddings.get_initialized_embedding_layer(frozen=False, padding_idx= 0)

    dan = DAN(emb_layer, hid, out)
    optimizer = torch.optim.Adam(dan.parameters(), lr=lr)

    batch_size = 32
    start = 0
    end = 0
    batches = []
    for i in range(1, int(np.ceil(len(train_exs)/batch_size))):
        start = (i-1) * batch_size
        end = min(len(train_exs), i * batch_size)

        batch = train_exs[start:end]
        batches.append(batch)


    loss_fn = nn.CrossEntropyLoss()
    classifier = None
    for epoch in range(0, num_epochs):
        total_loss = 0
        # for ex in train_exs:
        #     word_idxs = []
        #     for w in ex.words:
        #         idx = word_embeddings.word_indexer.index_of(w)
        #         if idx == -1:
        #             word_idxs.append(1)
        #         else:
        #             word_idxs.append(idx)
        #     word_idxs = torch.tensor(word_idxs).unsqueeze(0)

        #     gold_label = torch.zeros(2)
        #     gold_label[ex.label] = 1
            
        #     probs = dan(word_idxs)
        #     loss = torch.neg(torch.log(probs.squeeze(0)).dot(gold_label))
            
        #     dan.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     total_loss += loss.item()
        np.random.shuffle(batches)
        for batch in batches:
            batch_word_idxs= []
            # batch_labels = []

            for ex in batch:
                word_idxs = []
                for w in ex.words:
                    idx = word_embeddings.word_indexer.index_of(w)
                    if idx == -1:
                        word_idxs.append(1)
                    else:
                        word_idxs.append(idx)
                # word_idxs = torch.tensor(word_idxs).unsqueeze(0)

                # gold_label = torch.zeros(2)
                # gold_label = [0,0]
                # gold_label[ex.label] = 1

                batch_word_idxs.append(word_idxs)
                # batch_labels.append(gold_label)
            m = np.max([len(x) for x in batch_word_idxs])

            for i in range(len(batch_word_idxs)):
                padding = m - len(batch_word_idxs[i])
                batch_word_idxs[i] = batch_word_idxs[i] + list(np.zeros(padding))
            # gold_labels_tensor = torch.tensor(batch_labels)
            gold_labels_tensor = torch.tensor([ex.label for ex in batch], dtype=torch.long)
            logits = dan(batch_word_idxs)
            # loss = torch.neg(torch.sum(torch.log(probs) * gold_labels_tensor, dim=1)).mean()
            loss = loss_fn(logits, gold_labels_tensor)
            
            dan.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        classifier = NeuralSentimentClassifier(dan, word_embeddings)
        predictions = classifier.predict_all([ex.words for ex in dev_exs], has_typos=False)

        num_correct = 0
        num_pos_correct = 0
        num_pred = 0
        num_gold = 0
        num_total = 0
        golds = [ex.label for ex in dev_exs]
        if len(golds) != len(predictions):
            raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
        for idx in range(0, len(golds)):
            gold = golds[idx]
            prediction = predictions[idx]
            if prediction == gold:
                num_correct += 1
            if prediction == 1:
                num_pred += 1
            if gold == 1:
                num_gold += 1
            if prediction == 1 and gold == 1:
                num_pos_correct += 1
            num_total += 1
        acc = float(num_correct) / num_total
        output_str = "Accuracy: %i / %i = %f" % (num_correct, num_total, acc)
        prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
        rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
        # output_str += ";\nPrecision (fraction of predicted positives that are correct): %i / %i = %f" % (num_pos_correct, num_pred, prec)
        # output_str += ";\nRecall (fraction of true positives predicted correctly): %i / %i = %f" % (num_pos_correct, num_gold, rec)
        # output_str += ";\nF1 (harmonic mean of precision and recall): %f;\n" % f1
        # print(output_str)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_exs)}, Accuracy: %i / %i = %f, F1 : %f" % (num_correct, num_total, acc, f1))


    return NeuralSentimentClassifier(dan, word_embeddings)