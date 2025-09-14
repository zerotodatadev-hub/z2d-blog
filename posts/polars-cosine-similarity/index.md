<!--
.. title: Cosine Similarity in Polars: From Naive to Vectorized
.. slug: polars-cosine-similarity
.. date: 2025-09-11 10:00:00 UTC
.. tags: python, polars, embeddings, performance, vectorization, data-engineering
.. category: Data Engineering
.. link:
.. description: Calculating cosine similarity between consecutive row embeddings in Polars — starting naïve, then making it pretty and fast with vectorized arrays.
.. type: text
-->

## Why I Wrote This

I needed to compute cosine similarity between word embeddings of consecutive rows in a large dataset. The goal was simple: detect how much the embedding at row i resembles the embedding at row i+1. I started with a quick, naïve version to validate correctness and then refactored to a vectorized, production‑ready approach in Polars.

Mindset I follow: make it work → make it pretty → make it fast. Don’t pre-optimize; ship something correct first, then iterate.

## The Setup

- The input is a Polars `DataFrame` with an `embedding` column.
- Each `embedding` is a fixed‑length list of floats (e.g., 384‑d or 768‑d).
- We want a new column `cosine_similarity` comparing each row’s embedding with the next row’s embedding.

For clarity, I show two implementations:

1) a naïve, `map_elements`/Python approach for a proof of concept
2) a vectorized, array‑based approach that is much faster on large data

## Naïve First: Make It Work

This version uses `pl.struct` to package multiple columns per row and `map_elements` to call a small Python function. It’s straightforward and great for getting the logic right.

```python
import polars as pl
import numpy as np


def _compute_cosine(vec_1: list[float], vec_2: list[float]) -> float:
    # handle cases where a vector is missing (eg. last entry because of shift)
    if vec_1 is None or vec_2 is None:
        return None

    # cosine similarity calculation with numpy
    v1 = np.array(vec_1)
    v2 = np.array(vec_2)
    cosine = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    return cosine


def calculate_cosine_similarity_naive(df: pl.DataFrame) -> pl.DataFrame:
    # shift the embedding before calculation of the vector similarity
    # with the next row element
    df = df.with_columns([pl.col("embedding").shift(-1).alias("next_embedding")])

    df = df.with_columns(
        [
            # select required columns as struct. Allows dict-like usage in lambda
            pl.struct(("embedding", "next_embedding"))
            .map_elements(  # iteration over entries (comparable to pandas .apply() )
                # parametrize the function for calculation of the cosine similarity
                lambda s: _compute_cosine(s["embedding"], s["next_embedding"]),
                return_dtype=pl.Float32,  # specify the return
                returns_scalar=True,
            )
            .alias("cosine_similarity")
        ]
    )
    return df
```

Why this first?

- It’s easy to read and verify.
- `pl.struct` cleanly passes multiple row fields into a single function.
- Perfect for a small sample to check the math and edge cases (e.g., last row → `None`).

Trade-offs:

- It iterates in Python space, so it’s slower on large datasets.
- It won’t fully leverage Polars’ vectorized execution engine.

## Make It Pretty and Fast: Vectorized Arrays

Once the logic looked good, I refactored to a vectorized implementation. The key is to convert list embeddings into fixed-size array dtype and then use elementwise arithmetic plus `arr.sum()` to compute dot products and norms efficiently.

```python
import polars as pl
import numpy as np


def calculate_cosine_similarity_optimized(
    df: pl.DataFrame,
) -> pl.DataFrame:
    # Determine the embedding length from the first row
    embedding_length = len(df.head(1).select(pl.col("embedding")).item())

    similarity = (
        df.with_columns(
            # shift the embedding before calculation of the vector similarity
            # with the next row element
            [
                pl.col("embedding").alias("current_embedding"),
                pl.col("embedding").shift(-1).alias("next_embedding"),
            ]
        )
        .with_columns(
            [
                # convert both embedding columns to fixed-size arrays for vectorized ops
                pl.col("current_embedding")
                .list.to_array(embedding_length)
                .alias("current_array"),
                pl.col("next_embedding")
                .list.to_array(embedding_length)
                .alias("next_array"),
            ]
        )
        .with_columns(
            [
                # dot product via elementwise multiply + sum across the array
                (pl.col("current_array") * pl.col("next_array"))
                .arr.sum()
                .alias("dot_product"),

                # current norm: sqrt(sum(current_array * current_array))
                (pl.col("current_array") * pl.col("current_array"))
                .arr.sum()
                .sqrt()
                .alias("current_norm"),

                # next norm: sqrt(sum(next_array * next_array))
                (pl.col("next_array") * pl.col("next_array"))
                .arr.sum()
                .sqrt()
                .alias("next_norm"),
            ]
        )
        .with_columns(
            [
                # cosine similarity = dot / (||a|| * ||b||)
                (
                    pl.col("dot_product")
                    / (pl.col("current_norm") * pl.col("next_norm"))
                ).alias("cosine_similarity")
            ]
        )
    )
    return similarity
```

Why this is faster:

- Breaking the computation into simple vector expressions lets Polars optimize execution.
- Converting lists to `array` dtype enables true elementwise arithmetic and `arr.sum()` reductions.
- Everything runs inside Polars’ engine, minimizing Python overhead.

## Small Example

Here’s a minimal example you can run to see both versions side-by-side:

```python
import polars as pl

df = pl.DataFrame(
    {
        "id": [1, 2, 3],
        "embedding": [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
        ],
    }
)

naive = calculate_cosine_similarity_naive(df)
optimized = calculate_cosine_similarity_optimized(df)

print(naive.select(["id", "cosine_similarity"]))
print(optimized.select(["id", "cosine_similarity"]))
```

Both versions produce a `cosine_similarity` column. The last row will typically be `null` because there is no “next” row after the final one.

## Key Ideas to Reuse

- Breaking steps down: dot product and norms are separate, simple expressions. This makes the intent clear and enables Polars to optimize effectively.
- `pl.array` via `list.to_array(n)`: converting lists to fixed-size arrays unlocks elementwise ops and fast reductions like `.arr.sum()`.
- `pl.struct(...)`: the easiest way to pass multiple row fields into a single `map_elements` function in the naïve version.
- `shift(-1)`: aligns each row with the next row for pairwise comparisons across consecutive entries.

Edge cases and correctness:

- Ensure all embeddings have the same length before calling `list.to_array(n)`.
- If your last row’s `next_embedding` is null, the final `cosine_similarity` should be null as well — that’s expected.
- If you have missing embeddings earlier in the column, decide whether to fill, drop, or carry forward before computing similarity.

## Lessons Learned / Takeaways

- Make it work → make it pretty → make it fast. Start naïve to validate logic and edge cases, then vectorize.
- Prefer vectorized Polars operations over row-wise Python loops for performance.
- Convert list embeddings to fixed-size arrays to use elementwise arithmetic and `.arr.sum()`.
- Use `pl.struct` when you need to feed multiple columns into a single row-wise function.
- Keep computations explicit and composable; it’s easier to reason about, debug, and optimize.

