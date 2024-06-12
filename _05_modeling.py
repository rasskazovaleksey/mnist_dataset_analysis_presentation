from libs.scaler import min_max_scaler
from libs.scaler import black_and_white_scaler
from libs.mnist_reader import load_mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from libs.model_processor import create_default_model
from pathlib import Path
import pandas as pd
from libs.model_processor import read_params

print("Scaling data")

X_train_mnist, Y_train_mnist = load_mnist(path="data/mnist", kind="train")
X_train_fashion, Y_train_fashion = load_mnist(path="data/fashion", kind="train")

mnist_black_and_white_no_scaling = black_and_white_scaler(X_train_mnist)
fashion_black_and_white_no_scaling = black_and_white_scaler(X_train_fashion)

datasets = {
    # mnist
    "mnist_original": X_train_mnist,
    "mnist_black_and_white_no_scaling": mnist_black_and_white_no_scaling,
    "mnist_original_89_attributes": PCA(n_components=89).fit_transform(X_train_mnist),
    "mnist_black_and_white_233_attributes": PCA(n_components=233).fit_transform(mnist_black_and_white_no_scaling),
    # fashion
    "fashion_original": X_train_fashion,
    "fashion_black_and_white_no_scaling": fashion_black_and_white_no_scaling,
    "fashion_original_89_attributes": PCA(n_components=89).fit_transform(X_train_fashion),
    "fashion_black_and_white_233_attributes": PCA(n_components=233).fit_transform(fashion_black_and_white_no_scaling),
}

def iterate_model(model_dict: dict, X: list, y: list, dataset_name: str):
    Path("output").mkdir(parents=True, exist_ok=True)
    model = create_default_model(model_dict["model"])
    result_file = f"output/{dataset_name}_{model_dict['model']}.csv"
    if Path(result_file).is_file():
        print(f"Skipping '{result_file}' since in alredy exist")
        return
    
    clf = GridSearchCV(estimator=model, param_grid=model_dict["params"], verbose=3, cv=[(slice(None), slice(None))])
    clf.fit(X, y)

    df = pd.DataFrame(clf.cv_results_)
    df.to_csv(result_file)

    return clf.cv_results_

models_params = read_params()
# for each dataset
for key in datasets:
    print(f"Processing dataset: {key}")
    # for each model in dict
    for model_dict in models_params:
        print(f"Processing model: {model_dict}")
        X = datasets[key]
        if key.startswith("mnist"):
            y = Y_train_mnist
        elif key.startswith("fashion"):
            y = Y_train_fashion
        else:
            raise KeyError
        iterate_model(model_dict, X=X, y=y, dataset_name=key)
