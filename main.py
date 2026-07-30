"""Netflix stock closing price prediction using Random Forest, Polynomial Regression, and AdaBoost."""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")

DEFAULT_DATA_PATH = Path("data/NFLX.csv")
DEFAULT_OUTPUT_PATH = Path("data/NFLX_Final.csv")
DEFAULT_PLOTS_DIR = Path("imgs")

FEATURE_COLUMNS = ["Open", "High", "Low"]
TARGET_COLUMN = "Close"
EPOCH = pd.Timestamp("1970-01-01")

RANDOM_FOREST_PARAM_GRID: dict[str, Any] = {
    "n_estimators": [20, 50, 100, 500, 1000],
    "max_depth": np.arange(1, 15, 1),
    "min_samples_split": np.arange(2, 10, 2),
    "min_samples_leaf": np.arange(1, 15, 2),
    "bootstrap": [True, False],
    "random_state": [1],
}


@dataclass
class ModelMetrics:
    name: str
    mse: float
    mape: float
    train_score: float
    test_score: float
    elapsed_time: float


@dataclass
class SplitData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass
class PipelineResult:
    dataframe: pd.DataFrame
    metrics: dict[str, ModelMetrics]
    random_forest: RandomForestRegressor
    polynomial_features: PolynomialFeatures
    elastic_net: ElasticNet
    ada_boost: AdaBoostRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare regression models for Netflix stock closing prices.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Input CSV path (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path with predictions (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help=f"Directory for saved plots (default: {DEFAULT_PLOTS_DIR})",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plot files",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively after saving",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out fraction for testing (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=1,
        help="Random seed for reproducibility (default: 1)",
    )
    parser.add_argument(
        "--rf-iter",
        type=int,
        default=200,
        help="RandomizedSearchCV iterations for Random Forest (default: 200)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for Random Forest search (-1 uses all cores)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed hyperparameter search output",
    )
    return parser.parse_args()


def load_and_preprocess(path: Path) -> pd.DataFrame:
    """Load raw NFLX data and prepare features for modeling."""
    df = pd.read_csv(path)
    df.set_index("Date", inplace=True, drop=False)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = (df["Date"] - EPOCH) / pd.Timedelta(days=1)

    close_equals_adj = (df["Close"] == df["Adj Close"]).sum()
    print(f"Records where Close equals Adj Close: {close_equals_adj}/{len(df)}")

    df = df.drop(columns=["Adj Close", "Volume"])
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, SplitData]:
    features = df[FEATURE_COLUMNS]
    target = df[[TARGET_COLUMN]]

    x_train, x_test, y_train, y_test = train_test_split(
        features.values,
        target.values,
        test_size=test_size,
        random_state=random_state,
    )
    return df, SplitData(x_train, x_test, y_train, y_test)


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Matches the original notebook metric definition.
    return float(100 - round(np.mean(np.abs((y_true - y_pred) / y_true)), 4))


def build_metrics(
    name: str,
    model: Any,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    elapsed_time: float,
) -> ModelMetrics:
    return ModelMetrics(
        name=name,
        mse=round(metrics.mean_squared_error(y_test, y_pred), 4),
        mape=compute_mape(y_test, y_pred),
        train_score=round(model.score(x_train, y_train) * 100, 4),
        test_score=round(model.score(x_test, y_test) * 100, 4),
        elapsed_time=round(elapsed_time, 4),
    )


def print_metrics(result: ModelMetrics, time_label: str = "Total computational time") -> None:
    print(f"\n{result.name} evaluation metrics:")
    print(f"  MSE: {result.mse}")
    print(f"  MAPE: {result.mape}%")
    print(f"  Training score: {result.train_score}%")
    print(f"  Testing score: {result.test_score}%")
    print(f"  {time_label}: {result.elapsed_time}s")


def train_random_forest(
    split: SplitData,
    random_state: int,
    n_iter: int,
    n_jobs: int,
    verbose: bool,
) -> tuple[RandomForestRegressor, ModelMetrics]:
    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(),
        param_distributions=RANDOM_FOREST_PARAM_GRID,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=n_jobs,
        verbose=100 if verbose else 0,
        n_iter=n_iter,
        return_train_score=True,
        random_state=random_state,
    )

    start = time.time()
    search.fit(split.x_train, split.y_train.ravel())
    elapsed = time.time() - start

    model = search.best_estimator_
    model.fit(split.x_train, split.y_train.ravel())
    predictions = model.predict(split.x_test)

    if verbose:
        print("\nRandom Forest hyperparameter search results:")
        print(f"Best parameters: {search.best_params_}")
        print(f"Best MSE: {round(-search.best_score_, 2)}")
    else:
        print(f"\nRandom Forest best parameters: {search.best_params_}")

    metrics_result = build_metrics(
        "Random Forest",
        model,
        split.x_train,
        split.x_test,
        split.y_train,
        split.y_test,
        predictions,
        elapsed,
    )
    return model, metrics_result


def train_polynomial_regression(split: SplitData) -> tuple[PolynomialFeatures, ElasticNet, ModelMetrics]:
    elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)
    degree_scores: list[tuple[int, float]] = []

    print("\nPolynomial regression degree search:")
    start = time.time()

    for degree in range(1, 10):
        poly = PolynomialFeatures(degree=degree)
        x_train_poly = poly.fit_transform(split.x_train)
        x_test_poly = poly.transform(split.x_test)

        elastic_net.fit(x_train_poly, split.y_train)
        predictions = elastic_net.predict(x_test_poly)
        mse = round(metrics.mean_squared_error(split.y_test, predictions), 4)
        degree_scores.append((degree, mse))
        print(f"  degree={degree}\tmse={mse}")

    elapsed = time.time() - start
    best_degree = min(degree_scores, key=lambda item: item[1])[0]
    print(f"Best polynomial degree: {best_degree}")

    poly_features = PolynomialFeatures(degree=best_degree)
    model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    x_train_poly = poly_features.fit_transform(split.x_train)
    x_test_poly = poly_features.transform(split.x_test)
    model.fit(x_train_poly, split.y_train)
    predictions = model.predict(x_test_poly)

    metrics_result = build_metrics(
        "Polynomial Regression",
        model,
        x_train_poly,
        x_test_poly,
        split.y_train,
        split.y_test,
        predictions,
        elapsed,
    )
    return poly_features, model, metrics_result


def train_ada_boost(
    split: SplitData,
    base_estimator: RandomForestRegressor,
    random_state: int,
) -> tuple[AdaBoostRegressor, ModelMetrics]:
    model = AdaBoostRegressor(
        estimator=base_estimator,
        n_estimators=50,
        random_state=random_state,
    )

    start = time.time()
    model.fit(split.x_train, split.y_train)
    elapsed = time.time() - start

    predictions = model.predict(split.x_test)
    metrics_result = build_metrics(
        "AdaBoost",
        model,
        split.x_train,
        split.x_test,
        split.y_train,
        split.y_test,
        predictions,
        elapsed,
    )
    return model, metrics_result


def add_predictions(
    df: pd.DataFrame,
    random_forest: RandomForestRegressor,
    poly_features: PolynomialFeatures,
    elastic_net: ElasticNet,
    ada_boost: AdaBoostRegressor,
) -> pd.DataFrame:
    features = df[FEATURE_COLUMNS].values
    df = df.copy()
    df["RFR Close Predictions"] = random_forest.predict(features)
    df["Polynomial Close Predictions"] = elastic_net.predict(poly_features.transform(features))
    df["Ada Close Predictions"] = ada_boost.predict(features)
    return df


def compare_models(model_metrics: dict[str, ModelMetrics]) -> None:
    print("\nModel comparison:")

    for metric_name, lower_is_better in (("mse", True), ("mape", True), ("elapsed_time", True)):
        values = {name: getattr(stats, metric_name) for name, stats in model_metrics.items()}
        best = min(values, key=values.get) if lower_is_better else max(values, key=values.get)
        worst = max(values, key=values.get) if lower_is_better else min(values, key=values.get)
        label = metric_name.upper() if metric_name != "elapsed_time" else "Runtime"
        print(f"  Lowest {label}: {best} ({values[best]}{'%' if metric_name == 'mape' else 's' if metric_name == 'elapsed_time' else ''})")
        print(f"  Highest {label}: {worst} ({values[worst]}{'%' if metric_name == 'mape' else 's' if metric_name == 'elapsed_time' else ''})")


def _save_or_show(fig: plt.Figure, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def generate_plots(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    close_hist = plt.figure(figsize=(14, 5))
    plt.hist(df["Close"], color="dodgerblue")
    plt.xlabel("Close Prices in USD", weight="bold")
    plt.ylabel("Frequency", weight="bold")
    plt.title("\nClose Prices Frequency\n", weight="bold")
    plt.grid()
    _save_or_show(close_hist, plots_dir / "close_prices_frequency.png", show)

    close_series = plt.figure(figsize=(20, 7))
    df["Close"].plot(color="dodgerblue")
    plt.ylabel("Close Prices in USD", weight="bold")
    plt.xlabel("Date", weight="bold")
    plt.title("\nNetflix Stock Closing Prices Across 5 Years\n", weight="bold")
    plt.grid()
    _save_or_show(close_series, plots_dir / "nflx_stock_price.png", show)

    corr = df[[*FEATURE_COLUMNS, TARGET_COLUMN]].corr()
    heatmap = plt.figure()
    plt.imshow(corr, cmap="YlGnBu", interpolation="nearest", vmin=0.995)
    plt.colorbar()
    for row in range(len(corr)):
        for col in range(len(corr)):
            plt.annotate(
                f"{corr.values[row][col]:.4f}",
                xy=(col, row),
                ha="center",
                va="center",
                color="black",
            )
    plt.title("\nAttributes Heat Map\n", weight="bold")
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    _save_or_show(heatmap, plots_dir / "correlation_heatmap.png", show)

    for column, color, filename in (
        ("RFR Close Predictions", "dodgerblue", "random_forest_predictions.png"),
        ("Polynomial Close Predictions", "limegreen", "polynomial_predictions.png"),
        ("Ada Close Predictions", "gold", "ada_boost_predictions.png"),
    ):
        fig = plt.figure(figsize=(20, 7))
        df["Close"].plot(color="red", label="Close")
        df[column].plot(color=color, label=column)
        plt.xlabel("Date", weight="bold")
        plt.ylabel("Close Prices in USD", weight="bold")
        plt.title("\nNetflix Stock Closing Prices Across 5 Years\n", weight="bold")
        plt.legend()
        plt.grid()
        _save_or_show(fig, plots_dir / filename, show)

    comparison = plt.figure(figsize=(20, 7))
    df["Close"].plot(color="red", label="Close")
    df["Ada Close Predictions"].plot(color="gold", label="Ada Close Predictions")
    df["RFR Close Predictions"].plot(color="limegreen", label="RFR Close Predictions")
    df["Polynomial Close Predictions"].plot(color="dodgerblue", label="Polynomial Close Predictions")
    plt.xlabel("Date", weight="bold")
    plt.ylabel("Close Prices in USD", weight="bold")
    plt.title("\nPredicted Netflix Stock Closing Prices Across 5 Years\n", weight="bold")
    plt.legend()
    plt.grid()
    _save_or_show(comparison, plots_dir / "final_output.png", show)


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    print(f"Loading data from {args.input}")
    df = load_and_preprocess(args.input)
    print(f"Dataset shape after cleaning: {df.shape}")

    _, split = split_dataset(df, args.test_size, args.random_state)

    random_forest, rf_metrics = train_random_forest(
        split,
        random_state=args.random_state,
        n_iter=args.rf_iter,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )
    print_metrics(rf_metrics, "Total computational time (Random Forest search)")

    poly_features, elastic_net, poly_metrics = train_polynomial_regression(split)
    print_metrics(poly_metrics)

    ada_boost, ada_metrics = train_ada_boost(split, random_forest, args.random_state)
    print_metrics(ada_metrics)

    all_metrics = {
        "Random Forest": rf_metrics,
        "Polynomial": poly_metrics,
        "Ada boost": ada_metrics,
    }
    compare_models(all_metrics)

    df = add_predictions(df, random_forest, poly_features, elastic_net, ada_boost)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output)
    print(f"\nSaved predictions to {args.output}")

    if not args.no_plots:
        generate_plots(df, args.plots_dir, args.show_plots)
        print(f"Saved plots to {args.plots_dir}/")

    return PipelineResult(
        dataframe=df,
        metrics=all_metrics,
        random_forest=random_forest,
        polynomial_features=poly_features,
        elastic_net=elastic_net,
        ada_boost=ada_boost,
    )


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
