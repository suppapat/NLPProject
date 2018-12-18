import pickle
from pythainlp import word_tokenize, sentiment


class Display:

    def __init__(self):
        # self.dict_vectorizer = DictVectorizer(sparse=True)
        self.model = pickle.load(open("k-means_model.pickle", "rb"))
        self.dict_vectorizer = pickle.load(open("model_dict_vectorizer.pickle", "rb"))
        print(self.dict_vectorizer)
        print(self.model)

    def classify(self, text: str):
        """

        :param text:
        :return:
        """
        feature = self.make_feature(text)
        sparse_feature_matrix = self.dict_vectorizer.transform(feature)
        result = self.model.predict(sparse_feature_matrix)

        return result

    def make_feature(self, text):
        word_list = word_tokenize(text.strip(), engine='newmm')
        sentiment_str = sentiment(text.strip())
        if sentiment_str == 'neg':
            senti = 1
        elif sentiment_str == 'neutral':
            senti = 2
        elif sentiment_str == 'pos':
            senti = 3
        # text_dict = {'__len': len(text), '__word': len(word_list), '__sentiment': senti}
        text_dict = {'__sentiment': senti}
        for temp in word_list:
            if temp in text_dict:
                text_dict[str(temp)] += 1
            else:
                text_dict[str(temp)] = 1
        return text_dict

    def classify_sentiment_from_file(self, text_file_name):
        file = open(text_file_name, "r", encoding="utf8")
        for line in file.readlines():
            print(line.strip())
            print(self.classify_sentiment(line.strip()))

    def get_top_feature(self, label: int, top_k: int):
        res = self.model.__dict__
        top_feature_indices = res['cluster_centers_'].argsort()[:, -top_k - 1:-1]
        label0_top_features = [self.dict_vectorizer.get_feature_names()[x] for x in top_feature_indices[label]]
        label0_top_features.reverse()
        print(label0_top_features)

