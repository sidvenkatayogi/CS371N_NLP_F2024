# models.py

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import random
from transformer import PositionalEncoding
from datetime import datetime

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(nn.Module):
    def __init__(self, vocab_index, vocab_size, d_model, num_layers, num_classes, chunk_len):
        super(NeuralLanguageModel, self).__init__()
        self.vocab_index = vocab_index
        self.vocab_size = vocab_size
        self.chunk_len = chunk_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.posenc = PositionalEncoding(d_model, chunk_len + 1)
        enc_layer = nn.TransformerEncoderLayer(d_model= d_model, nhead= 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers= num_layers)
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, input_tensor):
        mask = nn.Transformer.generate_square_subsequent_mask(input_tensor.shape[1])
        embedded = self.posenc(self.embedding(input_tensor))
        output = self.transformer(embedded, mask=mask)
        output = self.output_layer(output)
        return nn.functional.log_softmax(output, dim=2)

    def get_next_char_log_probs(self, context):
        self.eval()
        with torch.no_grad():
            context = ' ' + context
            context_indices = [self.vocab_index.index_of(c) for c in context[-self.chunk_len:]]
            context_tensor = torch.tensor([context_indices], dtype= torch.long)
            log_probs = self.forward(context_tensor)
            return log_probs[0, -1, :].detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        self.eval()
        log_probs = 0.0
        for i, char in enumerate(next_chars):
            current_context = context + next_chars[:i]
            log_probs += self.get_next_char_log_probs(current_context[-self.chunk_len:])[self.vocab_index.index_of(char)]
        return log_probs

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    s = datetime.now()
    # Hyperparameters
    d_model = 128
    vocab_size = len(vocab_index)
    num_classes = vocab_size
    num_layers = 2
    lr = 5e-4
    num_epochs = 10
    chunk_len = 16
    batch_size = 256

    model = NeuralLanguageModel(vocab_index, vocab_size, d_model, num_layers, num_classes, chunk_len)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fcn = nn.NLLLoss()
    
    train_examples = []
    for i in range(0, len(train_text) - chunk_len):
        train_examples.append([vocab_index.index_of(c) for c in train_text[i:i+chunk_len+1]])

    for t in range(num_epochs):
        model.train()
        es = datetime.now()
        epoch_loss = 0.0

        random.seed(t)
        random.shuffle(train_examples)
        
        for i in range(0, len(train_examples), batch_size):
            batch_examples = train_examples[i:i+batch_size]
            
            inputs = torch.tensor([[vocab_index.index_of(' ')] + ex[:-1] for ex in batch_examples], dtype= torch.long)
            targets = torch.tensor([ex[:] for ex in batch_examples], dtype= torch.long)

            optimizer.zero_grad()

            log_probs = model.forward(inputs)
        
            loss = loss_fcn(log_probs.view(-1, num_classes), targets.view(-1))
        
            loss.backward()
            optimizer.step()
        
            epoch_loss += loss.item() * len(batch_examples)

        ee = datetime.now()
        print(f"Epoch {t+1} loss: {epoch_loss/len(train_examples)}, time: {ee-es}")
        
        model.eval()
    e = datetime.now()

    print(f"Training time: {e-s}")
    return model
