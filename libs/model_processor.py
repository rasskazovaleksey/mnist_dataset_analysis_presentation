import json

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def read_params(path: str = "data/params.json") -> dict:
    with open(path) as json_file:
        data = json.load(json_file)

    return data


def create_default_model(name: str):
    if name == "LogisticRegression":
        return LogisticRegression()
    elif name == "KNeighborsClassifier":
        return KNeighborsClassifier()
    elif name == "GaussianNB":
        return GaussianNB()
    elif name == "DecisionTreeClassifier":
        return DecisionTreeClassifier()
    elif name == "RandomForestClassifier":
        return RandomForestClassifier()
    elif name == "KMeans":
        return KMeans()
    elif name == "AgglomerativeClustering":
        return AgglomerativeClustering()
    elif name == "SVC":
        return SVC()
    else:
        raise ValueError(f"'{name}' is not supported")
