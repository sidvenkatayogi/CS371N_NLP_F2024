from models import UnigramFeatureExtractor, PerceptronClassifier, train_perceptron
from utils import Indexer
from sentiment_data import SentimentExample

def test_unigram_feature_extractor():
    indexer = Indexer()
    feat_extractor = UnigramFeatureExtractor(indexer)
    sentence = ["great", "movie", "great"]
    features = feat_extractor.extract_features(sentence, add_to_indexer=True)
    print("Features:", features)
    print("Indexer:", indexer)

def test_perceptron_training():
    train_data = [
        SentimentExample(["great", "movie"], 1),
        SentimentExample(["bad", "movie"], -1),
        SentimentExample(["not", "great"], -1),
    ]
    indexer = Indexer()
    feat_extractor = UnigramFeatureExtractor(indexer)
    classifier = train_perceptron(train_data, feat_extractor)

    # Test predictions
    print("Prediction for ['great', 'movie']:", classifier.predict(["great", "movie"]))
    print("Prediction for ['bad', 'movie']:", classifier.predict(["bad", "movie"]))
    print("Prediction for ['not', 'great']:", classifier.predict(["not", "great"]))

if __name__ == "__main__":
    test_unigram_feature_extractor()
    test_perceptron_training()