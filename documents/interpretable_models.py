from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_graphviz
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
import graphviz
from sklearn.metrics import (
    recall_score,
    precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
import logging
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm
import seaborn as sns
from sklearn import preprocessing
import math

LOGGER = logging.getLogger(__name__)


def initialize_logger(logger: logging = LOGGER):
    """
    Initializes a logger with the specified logging level.

    Parameters:
        logger (logging.Logger): The logger object to be initialized.
          Defaults to the global logger.

    Returns:
        None
    """

    logging.basicConfig(
        level=logging.CRITICAL, format="%(levelname)s:%(name)s: %(message)s"
    )
    logger.setLevel(logging.INFO)


class InterpretableModel:
    def __init__(self, model_class, language="english", **kwargs):
        self.model_class = model_class

        assert self.model_class in [
            "linear_regression",
            "decision_tree",
            "logistic_regression",
            "KNN_regression",
        ]

        if self.model_class == "linear_regression":
            self.model = LinearRegression(**kwargs)

        elif self.model_class == "decision_tree":
            self.model = DecisionTreeRegressor(**kwargs)

        elif self.model_class == "logistic_regression":
            self.model = LogisticRegression(**kwargs)

        elif self.model_class == "KNN_regression":
            assert "n_neighbors" in kwargs, "Must specify the numer of neighbors"
            self.model = KNeighborsRegressor(**kwargs)

        elif self.model_class == "support_vector_machines":
            self.model = SVC(**kwargs)

        else:
            self.model = None

        self.X_train = None
        self.y_train = None

        self.X_test = None
        self.y_test = None

        self.language = language
        self.map_languages = {
            "decision_tree": "Árbol de Decisión",
            "logistic_regression": "Regresión Logística",
            "linear_regression": "Regresión Lineal",
            "KNN_regression": "KNN",
            "support_vector_machines": "SVM",
        }

    def initiate_data(self, data, X_columns, y_column, y_label, standarize=True):
        assert y_column in data.columns, f"{y_column} column not found in data."
        assert y_label in data.columns, f"{y_label} column not found in data."
        assert "train_test" in data.columns, "train_test column not found in data."
        for x_c in X_columns:
            assert x_c in data.columns, f"{x_c} column not found in data."

        train_set = data[data["train_test"] == "train"]
        test_set = data[data["train_test"] == "test"]

        X_train = train_set[X_columns]
        y_train = train_set[y_column]
        y_label_train = train_set[y_label].apply(transform_labels)

        X_test = test_set[X_columns]
        y_test = test_set[y_column]
        y_label_test = test_set[y_label].apply(transform_labels)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_label_train = y_label_train
        self.y_label_test = y_label_test

        self.X_columns = X_columns
        self.predicted_label_name = "prediction_label_name"
        self.standarize = standarize

        if self.standarize:
            self.standard_scaler = preprocessing.StandardScaler().fit(self.X_train)
            if self.model_class != "logistic_regression":
                self.standard_scaler_y = preprocessing.StandardScaler().fit(
                    pd.DataFrame(self.y_train)
                )
            else:
                self.standard_scaler_y = None

        logging.info("Data initiated successfully.")

    def fit(self, weights=False):
        logging.info("Fitting model.")
        if self.standarize:
            X_train = self.standard_scaler.transform(self.X_train)
            if self.model_class != "logistic_regression":
                y_train = self.standard_scaler_y.transform(pd.DataFrame(self.y_train))
            else:
                y_train = self.y_train

        if weights:
            counts_ = self.y_label_train.value_counts()
            sample_weight = np.ones(self.X_train.shape[0])
            sample_weight = [
                (counts_.sum() - counts_[counts_.index == y].values[0])
                / counts_.sum()
                * w
                for w, y in zip(sample_weight, self.y_label_train)
            ]
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)

        logging.info("Model fit successfully.")

    def predict(self, verbose=False):
        if self.standarize:
            X_train = self.standard_scaler.transform(self.X_train)
            X_test = self.standard_scaler.transform(self.X_test)

        if self.model_class == "logistic_regression":
            self.y_pred_train = self.model.predict_proba(X_train)[:, 0]
            self.y_pred_test = self.model.predict_proba(X_test)[:, 0]
        else:
            self.y_pred_train = self.model.predict(X_train)
            self.y_pred_test = self.model.predict(X_test)

        if self.standarize and self.model_class != "logistic_regression":
            # Unsantadize
            self.y_pred_train = self.standard_scaler_y.inverse_transform(
                self.y_pred_train.reshape(-1, 1)
            )
            self.y_pred_test = self.standard_scaler_y.inverse_transform(
                self.y_pred_test.reshape(-1, 1)
            )

        logging.info("Model predicted successfully.")

    def predict_given_samples(self, X_samples, verbose=False):
        assert len(X_samples) == 1, "Implemented only for a single sample"

        if self.standarize:
            X_samples_s = self.standard_scaler.transform(X_samples)

        if self.model_class == "logistic_regression":
            y_pred = self.model.predict_proba(X_samples_s)[:, 0]
        else:
            y_pred = self.model.predict(X_samples_s)

        if self.standarize and self.model_class != "logistic_regression":
            # Unsantadize
            y_pred = self.standard_scaler_y.inverse_transform(y_pred.reshape(-1, 1))

        if verbose:
            for index_v, v in enumerate(self.X_columns):
                print(f"{v} estandarizado:", f"{X_samples_s[0][index_v]:.3g}")

            print("\nPredicción:", f"{y_pred[0][0]:.3g}")
            print("\n\n")
        return y_pred

    def label_samples(self, threshold, train_test="test"):
        if train_test == "train":
            y_pred_label = self.y_pred_train >= threshold
        elif train_test == "test":
            y_pred_label = self.y_pred_test >= threshold
        else:
            raise ValueError("train_test must be 'train' or 'test'")

        return y_pred_label

    def evaluate(self):
        # Create the ROC curve
        thresholds = np.arange(0, 1.01, 0.02).tolist()
        true_positive_rate = []
        false_positive_rate = []

        for threshold in thresholds:
            y_pred_label_test = self.label_samples(threshold)
            metrics = self.compute_metrics(self.y_label_test, y_pred_label_test)
            true_positive_rate.append((metrics["tp"] / (metrics["tp"] + metrics["fn"])))
            false_positive_rate.append(
                (metrics["fp"] / (metrics["fp"] + metrics["tn"]))
            )

        self.true_positive_rate = true_positive_rate
        self.false_positive_rate = false_positive_rate
        self.true_positive_rate.append(0)
        self.false_positive_rate.append(0)

        self.AUC_test = -np.trapz(self.true_positive_rate, self.false_positive_rate)

        true_positive_rate = []
        false_positive_rate = []
        for threshold in thresholds:
            y_pred_label_train = self.label_samples(threshold, train_test="train")
            metrics = self.compute_metrics(self.y_label_train, y_pred_label_train)
            true_positive_rate.append((metrics["tp"] / (metrics["tp"] + metrics["fn"])))
            false_positive_rate.append(
                (metrics["fp"] / (metrics["fp"] + metrics["tn"]))
            )

        values = [
            (x) ** 2 + (1 - y) ** 2
            for x, y in zip(false_positive_rate, true_positive_rate)
        ]

        self.best_threshold = thresholds[np.argmin(values)]
        self.best_false_positive_rate = false_positive_rate[np.argmin(values)]
        self.best_true_positive_rate = true_positive_rate[np.argmin(values)]

        true_positive_rate.append(0)
        false_positive_rate.append(0)
        self.true_positive_rate_train = true_positive_rate
        self.false_positive_rate_train = false_positive_rate
        self.AUC_train = -np.trapz(
            self.true_positive_rate_train, self.false_positive_rate_train
        )

        self.AUC = self.AUC_train

    def compute_metrics(self, value, prediction):
        metrics_results = {}
        # accuracy
        metrics_results["accuracy"] = accuracy_score(value, prediction)
        # precision
        metrics_results["precision"] = precision_score(value, prediction)
        # recall
        metrics_results["recall"] = recall_score(value, prediction)
        # f1
        metrics_results["f1"] = f1_score(value, prediction)

        # specificity
        tn, fp, fn, tp = confusion_matrix(value, prediction).ravel()
        metrics_results["tn"] = tn
        metrics_results["fp"] = fp
        metrics_results["fn"] = fn
        metrics_results["tp"] = tp

        return metrics_results

    def wirte_report(self, main_threshold=None, plot_ROC=True):
        if main_threshold is None:
            main_threshold = self.best_threshold

        y_pred_label_train = self.label_samples(main_threshold, train_test="train")
        metrics_train = self.compute_metrics(self.y_label_train, y_pred_label_train)

        y_pred_label_test = self.label_samples(main_threshold)
        metrics = self.compute_metrics(self.y_label_test, y_pred_label_test)

        print(f"Report on: {self.model_class} using main threshold as {main_threshold}")

        print("Train data: ---------")
        print(f"\tAccuracy: {metrics_train['accuracy']:.3f}")
        print(f"\tPrecision: {metrics_train['precision']:.3f}")
        print(f"\tRecall: {metrics_train['recall']:.3f}")
        print(f"\tF1: {metrics_train['f1']:.3f}")
        print(
            f"\tSpecificity: {metrics_train['tn']/(metrics_train['tn'] + metrics_train['fp']):.3f}"
        )
        print(f"\n\tAUC: {self.AUC:.3f}")

        print("Test data: ---------")
        print(f"\tAccuracy: {metrics['accuracy']:.3f}")
        print(f"\tPrecision: {metrics['precision']:.3f}")
        print(f"\tRecall: {metrics['recall']:.3f}")
        print(f"\tF1: {metrics['f1']:.3f}")
        print(f"\tSpecificity: {metrics['tn']/(metrics['tn'] + metrics['fp']):.3f}")
        print(f"\n\tAUC: {self.AUC:.3f}")

        if plot_ROC:
            fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
            tab10 = plt.get_cmap("tab10")
            # ax.plot(
            #    self.false_positive_rate,
            #    self.true_positive_rate,
            #    label="Test" if self.language == "english" else "Prueba",
            #    color=tab10(1),
            # )
            ax.scatter(
                self.best_false_positive_rate,
                self.best_true_positive_rate,
                marker="x",
                label=f"Valor óptimo de corte\n para el umbral = {self.best_threshold:.2f}",
                color="black",
                s=30,
                zorder=3,
            )
            ax.plot(
                self.false_positive_rate_train,
                self.true_positive_rate_train,
                label="Train" if self.language == "english" else "Entrenamiento",
                color=tab10(0),
                zorder=2,
            )
            # Plot straight line
            ax.plot([0, 1], [0, 1], "k--")

            if self.language == "english":
                ax.set(
                    title=f"ROC curve for {self.model_class} model with AUC Test = {self.AUC:.3f}"
                )
                ax.set(xlabel="False Positive Rate")
                ax.set(ylabel="True Positive Rate")
            else:
                ax.set(
                    title=f"Curva ROC para el modelo {self.map_languages[self.model_class]}"
                )
                ax.set(xlabel="Tasa de falsos positivos")
                ax.set(ylabel="Tasa de verdades positivos")

            ax.legend()

    def represent_decision_tree(self):
        assert self.model_class == "decision_tree", "Needed a decision tree for this."
        plt.figure(figsize=(35, 12))  # set plot size (denoted in inches)
        plot_tree(
            self.model,
            feature_names=self.X_columns,
            filled=True,
            fontsize=12,
            class_names=["Clase Control", "Clase Paciente"],
            impurity=False,
            proportion=True,
            rounded=True,
        )
        plt.show()


def transform_labels(label: str):
    if label == "Control":
        return 0
    elif label == "Leve":
        return 1
    elif label == "Intermedio":
        return 1
    elif label == "Grave":
        return 1
    elif label == "Low":
        return 1
    elif label == "Medium":
        return 1
    elif label == "High":
        return 1
    elif label == "Correct":
        return 0
    elif label == "Error":
        return 1
    else:
        print(label)
        raise ValueError


def transform_inputs_labels_to_dataframe(
    input_data: np.ndarray, input_labels: list, predictions: np.ndarray
):
    items = []
    for input, label, prediction in tqdm(
        zip(input_data, input_labels, predictions), total=len(input_labels)
    ):
        input = input.flatten()
        dict_items = {}
        for i in range(len(input)):
            dict_items[f"input_{i}"] = input[i]
        dict_items["label"] = transform_labels(label)
        dict_items["prediction"] = prediction[0]
        items.append(dict_items)

    return pd.DataFrame(items)


def compare_different_models(all_models):
    fig, ax = plt.subplots(1, 1)
    data_models = []
    for model in all_models:
        data_model = pd.DataFrame(
            {
                "False Positive Rate": model.false_positive_rate,
                "True Positive Rate": model.true_positive_rate,
            }
        )
        data_model["model"] = model.model_class
        data_models.append(data_model)
    data_models = pd.concat(data_models)
    sns.lineplot(
        x="False Positive Rate",
        y="True Positive Rate",
        hue="model",
        data=data_models,
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], "k--")


def write_interpretability_linear(coefs, columns, intercept, model):
    s = ""
    for index, values in enumerate(zip(coefs[0], columns)):
        coef, column = values
        if index < len(coefs[0]) - 1:
            s += f"({coef:.3g} * {column}) + "
        else:
            s += f"({coef:.3g} * {column})"

    s += f" + ({intercept[0]:.3g})"
    scale = f"{model.standard_scaler_y.scale_[0]:.3g}"
    mean_ = f"{model.standard_scaler_y.mean_[0]:.3g}"
    s = f"{scale} * ({s}) + {mean_}"
    print(s)


def distance_point_to_line(x1, y1, A=1, B=-1, C=0):
    numerator = abs(A * x1 + B * y1 + C)
    denominator = math.sqrt(A**2 + B**2)
    distance = numerator / denominator
    return distance
