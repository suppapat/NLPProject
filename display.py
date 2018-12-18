import pickle
from cleandata import process_word_to_feature

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def display_sticker(sticker_name):
    """
    Display image

    :param sticker_name: image file name
    :return:
    """
    img = mpimg.imread('sticker/' + sticker_name + '.PNG')
    imgplot = plt.imshow(img)


class Display:

    def __init__(self):
        self.model = pickle.load(open("k-means_model.pickle", "rb"))
        self.dict_vectorizer = pickle.load(open("model_dict_vectorizer.pickle", "rb"))
        print(self.dict_vectorizer)
        print(self.model)

    def classify(self, text: str):
        """
        Classify text to each group

        :param text: text
        :return:
        """
        feature = process_word_to_feature(text)
        sparse_feature_matrix = self.dict_vectorizer.transform(feature)
        result = self.model.predict(sparse_feature_matrix)

        return result

    def classify_sentiment_from_file(self, text_file_name):
        """
        Classify sentiment from text file

        :param text_file_name: name of text file
        :return:
        """
        file = open(text_file_name, "r", encoding="utf8")
        for line in file.readlines():
            print(line.strip())
            print(self.classify_sentiment(line.strip()))

    def get_top_feature(self, group: int, top_k: int):
        """
        Get top feature from each group

        :param group: group no.
        :param top_k: no of you want top feature
        :return:
        """
        res = self.model.__dict__
        top_feature_indices = res['cluster_centers_'].argsort()[:, -top_k - 1:-1]
        label0_top_features = [self.dict_vectorizer.get_feature_names()[x] for x in top_feature_indices[group]]
        label0_top_features.reverse()
        print(label0_top_features)

    def reply_sticker(self, text: str):
        """
        Use for replay sticker from text

        :param text: text
        :return:
        """
        result = self.classify(text)[0]
        if result == 5:
            display_sticker('normal')
        elif result == 3:
            display_sticker('let\'sgo')
        elif result == 4:
            display_sticker('what')
        else:
            display_sticker('cry')
