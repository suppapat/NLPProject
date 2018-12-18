import json
import pickle
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer


class MakeModel:

    def __init__(self, feature_file_name, number_of_cluster):
        """

        :param feature_file_name: name of sparse feature file in json format
        :param number_of_cluster: number of cluster group
        """
        self.model = KMeans(n_clusters=number_of_cluster)
        self.dict_vectorizer = DictVectorizer(sparse=True)
        self.train(feature_file_name)

    def train(self,feature_file_name):
        """
        Start training model

        :param feature_file_name: name of feature file
        :return:
        """
        file = open(feature_file_name, 'r', encoding="utf8")
        feature_list = json.loads(file.read())
        train_sparse_feature_matrix = self.dict_vectorizer.fit_transform(feature_list)
        self.model.fit(train_sparse_feature_matrix)
        pickle.dump(self.model, open("k-means_model.pickle", "wb"))
        pickle.dump(self.dict_vectorizer, open("model_dict_vectorizer.pickle", "wb"))
