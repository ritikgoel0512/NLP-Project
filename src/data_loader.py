"""Data loading utilities for the WikiNews NLP project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


SELECTED_CATEGORIES = [
    "Politics and conflicts",
    "Economy and business",
    "Science and technology",
    "Sports",
]


def load_jsonl(path: str | Path) -> pd.DataFrame:
    """Load a JSONL file into a pandas DataFrame."""
    records = []

    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    dataframe = pd.DataFrame(records)

    return dataframe


def join_text_sentences(text_value: object) -> str:
    """Convert WikiNews sentence lists into a single article string."""
    if isinstance(text_value, list):
        return " ".join(str(sentence).strip() for sentence in text_value)

    if pd.isna(text_value):
        return ""

    return str(text_value).strip()


def prepare_wikinews_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and normalize core WikiNews fields."""
    result = dataframe.copy()

    result["text"] = result["text"].apply(join_text_sentences)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")

    result["word_count"] = result["text"].str.split().str.len()
    result["sentence_count"] = result["text"].str.count(r"[.!?]+")
    result["category_count"] = result["categories"].apply(
        lambda value: len(value) if isinstance(value, list) else 0
    )

    return result


def filter_by_language(
    dataframe: pd.DataFrame,
    language: str = "en",
) -> pd.DataFrame:
    """Filter WikiNews articles by language."""
    return dataframe.loc[dataframe["lang"] == language].copy()


def filter_by_categories(
    dataframe: pd.DataFrame,
    categories: Iterable[str],
) -> pd.DataFrame:
    """Keep articles containing at least one selected category."""
    category_set = set(categories)

    mask = dataframe["categories"].apply(
        lambda values: bool(
            category_set.intersection(values)
            if isinstance(values, list)
            else set()
        )
    )

    return dataframe.loc[mask].copy()


def assign_primary_category(
    dataframe: pd.DataFrame,
    categories: Iterable[str],
) -> pd.DataFrame:
    """Assign one selected category to each article.

    If an article belongs to multiple selected categories,
    the first category in the supplied category order is used.
    """
    category_order = list(categories)

    result = dataframe.copy()

    def get_primary(values: object) -> str | None:
        if not isinstance(values, list):
            return None

        for category in category_order:
            if category in values:
                return category

        return None

    result["primary_category"] = result["categories"].apply(get_primary)

    return result


def count_selected_categories(
    dataframe: pd.DataFrame,
    categories: Iterable[str],
) -> pd.Series:
    """Count how many articles belong to each selected category."""
    counts = {}

    for category in categories:
        counts[category] = dataframe["categories"].apply(
            lambda values: (
                category in values
                if isinstance(values, list)
                else False
            )
        ).sum()

    return pd.Series(counts, name="article_count")


def select_analysis_sample(
    dataframe: pd.DataFrame,
    articles_per_category: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Select a reproducible sample for detailed NLP analysis."""
    samples = []

    for category in SELECTED_CATEGORIES:
        category_frame = dataframe.loc[
            dataframe["primary_category"] == category
        ]

        if len(category_frame) < articles_per_category:
            raise ValueError(
                f"Not enough articles for category: {category}"
            )

        sample = category_frame.sample(
            n=articles_per_category,
            random_state=random_state,
        )

        samples.append(sample)

    result = pd.concat(samples, ignore_index=True)

    return result
def filter_exclusive_categories(
    dataframe: pd.DataFrame,
    categories: Iterable[str],
) -> pd.DataFrame:
    """Keep articles belonging to exactly one selected category."""
    category_list = list(categories)

    def matched_categories(values: object) -> list[str]:
        if not isinstance(values, list):
            return []

        return [
            category
            for category in category_list
            if category in values
        ]

    result = dataframe.copy()

    result["matched_categories"] = result["categories"].apply(
        matched_categories
    )

    result = result.loc[
        result["matched_categories"].str.len() == 1
    ].copy()

    result["primary_category"] = result[
        "matched_categories"
    ].str[0]

    return result


def filter_analysis_candidates(
    dataframe: pd.DataFrame,
    min_words: int = 100,
    max_words: int = 3000,
) -> pd.DataFrame:
    """Filter articles suitable for detailed NLP analysis."""
    mask = (
        dataframe["date"].notna()
        & dataframe["text"].ne("")
        & dataframe["word_count"].between(
            min_words,
            max_words,
        )
    )

    return dataframe.loc[mask].copy()
