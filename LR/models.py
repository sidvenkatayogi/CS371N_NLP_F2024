# models.py
import numpy as np
# import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sentiment_data import *
from utils import *

from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False):
        features = Counter()
        for word in sentence:
            word = word.lower()
            idx = self.indexer.add_and_get_index(word, add_to_indexer)

            features[idx] += 1 # frequency

        return features


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False):
        features = Counter()
        for i in range(len(sentence)-1):
            bigram = (sentence[i].lower(), sentence[i+1].lower()) # no start or stop tokens
            idx = self.indexer.add_and_get_index(bigram, add_to_indexer)
            features[idx] += 1 # frequency

        return features

    


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    # nltk.download('stopwords')
    sw = set(stopwords.words('english'))

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.nsentences = 0
        self.df = Counter() # how frequent words are present in all examples
        
    def corpuscts(self, exs: List[SentimentExample]): # computes df
        for ex in exs:
            self.nsentences += 1
            seen = set()
            for word in ex.words:
                if word not in self.sw: # discard stopwords
                    idx = self.indexer.add_and_get_index(word.lower(), True)
                    if idx not in seen:
                        self.df[idx] += 1 # first time seeing word in sentence, add it to df
                        seen.add(idx)

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False):
        # if adding to index, update df too
        if add_to_indexer:  # this should normally be False though, and corpuscts should be called outside this method
            self.corpuscts([sentence])
        wfreq = Counter() # counts frequency of words in sentence
        features = Counter() # tf-idf weighted features
        L = len(sentence)
        for word in sentence:
            if word not in self.sw: # discard stopwords
                word = word.lower()
                idx = self.indexer.add_and_get_index(word, False)
                if idx >= 0: # discard unseen words when validating
                    wfreq[idx] += 1 # frequency
                    tf = wfreq[idx] / L

                    if not add_to_indexer:
                        idf = np.log(self.nsentences / (self.df[idx] + 1)) # add one in case word has never been seen before
                        features[idx] = tf * idf

        return features


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor
    
    def predict(self, sentence: List[str]):
        
        f = self.feat_extractor.extract_features(sentence=sentence, add_to_indexer=False).items()
        dotp = 0
        for idx, value in f:
            dotp += (self.weights[idx]*value)

        y_pred = dotp
        if y_pred > 0:
            return 1
        return 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    
    # first pass adds words to indexer, sets vocab size for weights
    for ex in train_exs:
        feat_extractor.extract_features(sentence=ex.words, add_to_indexer=True)
    weights = np.zeros(len(feat_extractor.indexer))

    epochs = 10

    rng = np.random.default_rng(seed=42)
    # rng = np.random.default_rng()
    rng.shuffle(train_exs)

    a = 1

    for epoch in range(epochs):
        rng.shuffle(train_exs)

        # a = 1/(epoch+1)
        a *= 0.75
        
        for ex in train_exs:
            f = feat_extractor.extract_features(sentence=ex.words, add_to_indexer=False).items()
            
            dotp = 0
            for idx, value in f:
                if idx >= 0: # should be >= 0, but just in case
                    dotp += (weights[idx]*value)

            y_pred = dotp
            print(y_pred)
            
            if y_pred > 0:
                y_pred = 1
            else: 
                y_pred = 0

            if y_pred < ex.label: # y is 1 and y_pred is incorrectly 0
                for idx, value in f:
                    weights[idx] += a*value
            elif y_pred > ex.label: # y is 0 and y_pred is incorrectly 1
                for idx, value in f:
                    weights[idx] -= a*value
            # else y_pred is correct so no update

    return PerceptronClassifier(weights, feat_extractor)


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor
    
    def predict(self, sentence: List[str]):
        
        f = self.feat_extractor.extract_features(sentence=sentence, add_to_indexer=False).items()
        dotp = 0
        for idx, value in f:
            if idx >= 0: # unseen words discarded when validating
                dotp += (self.weights[idx]*value)

        y_prob = 1 / (1 + np.exp(-dotp))
        
        if y_prob > 0.5:
            return 1
        return 0


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, dev_exs = None) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """

    # first pass adds words to indexer, sets document counts, sets vocab size for weights
    if isinstance(feat_extractor, BetterFeatureExtractor):
        feat_extractor.corpuscts(train_exs)
    else:
        for ex in train_exs:
            feat_extractor.extract_features(sentence=ex.words, add_to_indexer=True)
            

    weights = np.zeros(len(feat_extractor.indexer))

    epochs = 3

    # rng = np.random.default_rng(seed=42)
    rng = np.random.default_rng()
    rng.shuffle(train_exs)

    a = 0.75

    # # only used if plotting accuracies while training
    # tda = []
    # dda = []

    for epoch in range(epochs):
        rng.shuffle(train_exs) # shuffling data every epoch

        a = 1/(epoch+1)
        # a *= 0.75

        for ex in train_exs:
            
            f = feat_extractor.extract_features(sentence=ex.words, add_to_indexer=False).items()

            dotp = 0
            for idx, value in f:
                if idx >= 0: # should be > 0, but just in case
                    dotp += (weights[idx]*value)

            y_prob = 1 / (1 + np.exp(-dotp))

            if ex.label == 1:
                for idx, value in f:
                    if idx >= 0:
                        weights[idx] += a*value*(1 - y_prob) # -gradient = f(x) * (1-y_prob)
            elif ex.label == 0:
                for idx, value in f:
                    if idx >= 0:
                        weights[idx] -= a*value*(y_prob) # -gradient = -f(x) * y_prob
    #     if dev_exs:
    #         nc = 0 # num correct
    #         tester = LogisticRegressionClassifier(weights, feat_extractor)
    #         for ex in train_exs:
    #             if tester.predict(ex.words) == ex.label:
    #                 nc += 1
    #         acc = nc / len(train_exs)
    #         tda.append(acc)

    #         nc = 0
    #         for ex in dev_exs:
    #             if tester.predict(ex.words) == ex.label:
    #                 nc += 1
    #         acc = nc / len(dev_exs)
    #         dda.append(acc)
            

    # if dev_exs:
    #     plt.plot(range(len(tda)), tda, marker='o', label= "training")
    #     plt.plot(range(len(dda)), dda, marker='o', label= "dev")
    #     plt.title(f"Model accuracy vs. training iteration, with step = {a}") # change {a} if step is dynamic
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Accuracy")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()

    return LogisticRegressionClassifier(weights, feat_extractor)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    # elif args.model == "LRplot":
    #     model = train_logistic_regression(train_exs, feat_extractor, dev_exs)

    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model