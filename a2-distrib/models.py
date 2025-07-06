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
        word_embeddings_list = [self.word_embeddings.get_embedding(word) for word in ex_words]
        avg_embedding = torch.tensor(np.mean(word_embeddings_list, axis=0), dtype= torch.float32)
        
        with torch.no_grad():
            output = self.model(avg_embedding)
            prediction = torch.argmax(output).item()
        return prediction

class DAN(nn.Module):
        def __init__(self, input_size, hidden_size, out_size):
            super(DAN, self).__init__()
            # self.embedding_dim = embedding_dim
            self.hidden_size = hidden_size

            self.V = nn.Linear(input_size, hidden_size)
            self.g = nn.Tanh() # or nn.ReLU()
            self.W = nn.Linear(hidden_size, out_size)
            self.softmax = nn.Softmax(dim=0)

        def forward(self, x):
            return self.softmax(self.W(self.g(self.V(x))))

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
    lr = 0.001
    num_epochs = 50

    inp = word_embeddings.get_embedding_length()
    hid = 128
    out = 2

    dan = DAN(inp, hid, out)
    optimizer = torch.optim.Adam(dan.parameters(), lr=lr)

    for epoch in range(0, num_epochs):
        total_loss = 0
        for ex in train_exs:
            embeddings = [word_embeddings.get_embedding(w) for w in ex.words]

            avg_embedding = torch.tensor(np.mean(embeddings, axis=0), dtype= torch.float32)

            dan.zero_grad()
            gold_label = torch.zeros(2)
            gold_label[ex.label] = 1

            probs = dan.forward(avg_embedding)

            loss = torch.neg(torch.log(probs).dot(gold_label))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_exs)}")

    return NeuralSentimentClassifier(dan, word_embeddings)