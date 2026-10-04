"""
District population modelling and forecasting.

Contains:
    - DistrictPopulation: holds a district's year/population series
    - Forecaster (ABC): abstract base class for forecasting models
    - LinearTrendForecaster, CAGRForecaster, FibonacciForecaster
    - evaluate_forecaster: train/test evaluation helper
    - classrooms_needed, additional_classrooms: planning arithmetic
    - bootstrap_prediction_interval: residual-resampling PI
    - plot_district: reusable subplot for one district

All logic lives here. Data creation, printing and figure assembly
belong in the notebooks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import mean, median, variance, stdev

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PART 1: Core Data Container
# ============================================================
class DistrictPopulation:
    """Stores a district's yearly population series (in thousands)."""

    def __init__(self, name: str, years: np.ndarray, populations: np.ndarray) -> None:
        years = np.asarray(years, dtype=float)
        populations = np.asarray(populations, dtype=float)
        if years.shape != populations.shape:
            raise ValueError("years and populations must have the same length")
        if np.any(populations < 0):
            raise ValueError("populations cannot be negative")
        if len(years) < 2:
            raise ValueError("need at least two years")
        self.name = name
        self.years = years
        self.populations = populations

    def __len__(self) -> int:
        return len(self.years)

    def __repr__(self) -> str:
        return f"DistrictPopulation(name={self.name!r}, n={len(self)})"

    def yoy_growth(self) -> np.ndarray:
        """Year-on-year growth rates as fractions."""
        return np.diff(self.populations) / self.populations[:-1]

    def cagr(self) -> float:
        """Compound annual growth rate over the whole series."""
        n = len(self.populations) - 1
        return float((self.populations[-1] / self.populations[0]) ** (1 / n) - 1)

    def stats_python(self) -> dict:
        """Descriptive stats using the `statistics` module (sample, ddof=1)."""
        return {
            "mean": mean(self.populations),
            "median": median(self.populations),
            "variance": variance(self.populations),
            "stdev": stdev(self.populations),
        }

    def stats_numpy(self) -> dict:
        """Descriptive stats using NumPy (population, ddof=0)."""
        return {
            "mean": float(np.mean(self.populations)),
            "median": float(np.median(self.populations)),
            "variance": float(np.var(self.populations)),
            "stdev": float(np.std(self.populations)),
        }


# ============================================================
# PART 2: Forecaster base class + 3 subclasses
# ============================================================
class Forecaster(ABC):
    """Abstract base class for all forecasting models."""

    def __init__(self) -> None:
        self._fitted: bool = False

    @abstractmethod
    def fit(self, years: np.ndarray, populations: np.ndarray) -> "Forecaster":
        """Learn the model parameters from training data."""

    @abstractmethod
    def predict(self, future_years: np.ndarray) -> np.ndarray:
        """Return predictions for the given future years."""

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before predicting."
            )


class LinearTrendForecaster(Forecaster):
    """Extends a straight best-fit line into the future."""

    def __init__(self) -> None:
        super().__init__()
        self._slope: float = 0.0
        self._intercept: float = 0.0

    def fit(self, years, populations):
        self._slope, self._intercept = np.polyfit(years, populations, 1)
        self._fitted = True
        return self

    def predict(self, future_years):
        self._check_fitted()
        return self._slope * np.asarray(future_years, dtype=float) + self._intercept


class CAGRForecaster(Forecaster):
    """Projects the last value forward at the training CAGR."""

    def __init__(self) -> None:
        super().__init__()
        self._rate: float = 0.0
        self._last_year: float = 0.0
        self._last_pop: float = 0.0

    def fit(self, years, populations):
        years = np.asarray(years, dtype=float)
        populations = np.asarray(populations, dtype=float)
        n = len(populations) - 1
        self._rate = float((populations[-1] / populations[0]) ** (1.0 / n) - 1.0)
        self._last_year = float(years[-1])
        self._last_pop = float(populations[-1])
        self._fitted = True
        return self

    def predict(self, future_years):
        self._check_fitted()
        steps = np.asarray(future_years, dtype=float) - self._last_year
        return self._last_pop * (1.0 + self._rate) ** steps


class FibonacciForecaster(Forecaster):
    """Multiplies the last value by successive Fibonacci ratios."""

    def __init__(self, n_terms: int = 30) -> None:
        super().__init__()
        self._fib: list[int] = [1, 1]
        for _ in range(n_terms):
            self._fib.append(self._fib[-1] + self._fib[-2])
        self._last_pop: float = 0.0

    def fit(self, years, populations):
        self._last_pop = float(np.asarray(populations, dtype=float)[-1])
        self._fitted = True
        return self

    def predict(self, future_years):
        self._check_fitted()
        horizon = len(np.asarray(future_years))
        ratios = [self._fib[i + 1] / self._fib[i] for i in range(horizon)]
        out = np.empty(horizon)
        value = self._last_pop
        for i, r in enumerate(ratios):
            value *= r
            out[i] = value
        return out

# ============================================================
# PART 3: Evaluation
# ============================================================
def evaluate_forecaster(
    model: Forecaster,
    years: np.ndarray,
    populations: np.ndarray,
    split_year: int = 2022,
) -> dict:
    """
    Fit on years < split_year and test on years >= split_year.

    Returns a dict with model, MAE, RMSE, MAPE, predictions,
    actuals and test_years.
    """
    years = np.asarray(years, dtype=float)
    populations = np.asarray(populations, dtype=float)
    train_mask = years < split_year
    test_mask = years >= split_year

    if train_mask.sum() < 2:
        raise ValueError("Need at least 2 training points.")
    if test_mask.sum() == 0:
        raise ValueError(f"No test points on or after {split_year}.")

    model.fit(years[train_mask], populations[train_mask])
    predictions = model.predict(years[test_mask])
    actuals = populations[test_mask]
    errors = actuals - predictions

    return {
        "model": type(model).__name__,
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MAPE": float(np.mean(np.abs(errors / actuals)) * 100.0),
        "predictions": predictions,
        "actuals": actuals,
        "test_years": years[test_mask],
    }

# ============================================================
# PART 4: Planning helpers
# ============================================================
SCHOOL_AGE_FRACTION: float = 0.18
PUPILS_PER_CLASSROOM: int = 53


def classrooms_needed(population_thousands: float) -> float:
    """Convert a population (in thousands) into classrooms required."""
    persons = population_thousands * 1000.0
    pupils = persons * SCHOOL_AGE_FRACTION
    return pupils / PUPILS_PER_CLASSROOM


def additional_classrooms(district: DistrictPopulation, best_forecast: dict) -> dict:
    """Compare the last known year's classrooms to the 2029 forecast."""
    pop_last = float(district.populations[-1])
    pop_2029 = float(best_forecast["forecast"][-1])
    cls_last = classrooms_needed(pop_last)
    cls_2029 = classrooms_needed(pop_2029)
    return {
        "district": district.name,
        "pop_2024": pop_last,
        "pop_2029": pop_2029,
        "classrooms_2024": cls_last,
        "classrooms_2029": cls_2029,
        "additional": cls_2029 - cls_last,
    }

# ============================================================
# PART 5: Forecasting helper + bootstrap
# ============================================================
def forecast_with_best(
    district: DistrictPopulation,
    model_factories: list,
    split_year: int = 2022,
    horizon_years: np.ndarray | None = None,
) -> dict:
    """
    Picked the best model by MAPE on the hold-out period, refit it on
    ALL data, then forecasted the horizon (default 2025-2029).
    """
    if horizon_years is None:
        horizon_years = np.array([2025, 2026, 2027, 2028, 2029])

    results = [
        evaluate_forecaster(f(), district.years, district.populations, split_year)
        for f in model_factories
    ]
    best = min(results, key=lambda r: r["MAPE"])

    winner = next(f() for f in model_factories if type(f()).__name__ == best["model"])
    winner.fit(district.years, district.populations)
    forecast = winner.predict(horizon_years)

    return {
        "district": district.name,
        "model": best["model"],
        "MAPE_holdout": best["MAPE"],
        "forecast_years": horizon_years,
        "forecast": forecast,
        "var_actual": float(np.var(district.populations)),
        "var_forecast": float(np.var(forecast)),
    }


def bootstrap_prediction_interval(
    model: Forecaster,
    years: np.ndarray,
    populations: np.ndarray,
    horizon_years: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap residual resampling to build a 95% prediction band."""
    rng = np.random.default_rng(seed)
    years = np.asarray(years, dtype=float)
    populations = np.asarray(populations, dtype=float)

    model.fit(years, populations)
    fitted = model.predict(years)
    residuals = populations - fitted

    boot_forecasts = np.empty((n_boot, len(horizon_years)))
    for i in range(n_boot):
        sampled = rng.choice(residuals, size=len(residuals), replace=True)
        pseudo = fitted + sampled
        m = type(model)()
        m.fit(years, pseudo)
        boot_forecasts[i] = m.predict(horizon_years)

    return (
        np.percentile(boot_forecasts, 2.5, axis=0),
        np.percentile(boot_forecasts, 97.5, axis=0),
    )


# ============================================================
# PART 6: Reusable plot helper
# ============================================================
def plot_district(
    ax,
    district: DistrictPopulation,
    model_factories: list,
    split_year: int = 2022,
    horizon_years: np.ndarray | None = None,
) -> None:
    """Draw one subplot: actual, fitted, forecast and 95% PI band."""
    if horizon_years is None:
        horizon_years = np.array([2025, 2026, 2027, 2028, 2029])

    results = [
        evaluate_forecaster(f(), district.years, district.populations, split_year)
        for f in model_factories
    ]
    best = min(results, key=lambda r: r["MAPE"])

    winner = next(f() for f in model_factories if type(f()).__name__ == best["model"])
    winner.fit(district.years, district.populations)

    fitted = winner.predict(district.years)
    forecast = winner.predict(horizon_years)
    lower, upper = bootstrap_prediction_interval(
        winner, district.years, district.populations, horizon_years, n_boot=1000
    )

    ax.plot(district.years, district.populations, "o-", label="Actual", color="black")
    ax.plot(district.years, fitted, "--", label=f"Fitted ({best['model']})", color="tab:blue")
    ax.plot(horizon_years, forecast, "s-", label="Forecast", color="tab:red")
    ax.fill_between(horizon_years, lower, upper, color="tab:red", alpha=0.2,
                    label="95% prediction interval")
    ax.axvline(split_year - 0.5, color="gray", linestyle=":",
               label="Train / Test split")
    ax.axvspan(district.years[0] - 0.5, split_year - 0.5, color="green", alpha=0.05)
    ax.axvspan(split_year - 0.5, horizon_years[-1] + 0.5, color="orange", alpha=0.05)

    ax.set_title(f"{district.name} — winner: {best['model']} (MAPE {best['MAPE']:.2f}%)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Population (thousands)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)