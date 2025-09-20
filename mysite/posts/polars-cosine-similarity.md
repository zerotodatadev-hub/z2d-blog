<!--
.. title: Polars Arrays: Fast Cosine Similarity for Adjacent Embeddings
.. slug: polars-cosine-similarity
.. date: 2025-09-20 12:18:00 UTC
.. tags: python, polars, tutorial, embeddings, performance, vectorization, data-science, beginner, intermediate
.. category: Data Science
.. link:
.. description: Calculating cosine similarity between consecutive row embeddings in Polars (Python Tutorial) — starting naive, then making it pretty and fast with vectorized arrays.
.. type: text
.. has_math: true
-->
# Polars Arrays: Fast Cosine Similarity for Adjacent Embeddings

Text streams can shift quickly: a news feed may move from one story to the next, or a tweet stream may switch topics in seconds. Embedding models such as [OpenAI's recent](https://platform.openai.com/docs/guides/embeddings#use-cases) `'text-embedding-3-large'` can convert the nuances of the text into meaningful vector representations. Comparing embeddings of adjacent entries helps flag these shifts — a drop in similarity is often the simplest signal of change.

> **TL;DR**  
>
> Cosine similarity between consecutive embeddings leveraging [Polars](https://docs.pola.rs/user-guide/getting-started/)-native syntax (Python Tutorial).  
>
> - Start with a simple, row-by-row `map_elements` approach (*make it work*).  
> - Switch to `pl.Array` and vectorized operations for big speedups (*make it fast*).  
> - In benchmarks, the Polars-native method runs ~17× faster than the naive Python version.  

## Exemplary dataset

The embedding process converts text into vectors, which we can store in a single Polars column. Depending on the model, vector size can range from 100 to several thousand values.

> **Setup**  
>
> Code tested with:  
>
> - *Polars 1.33.1*
> - *NumPy 2.3.3*
>
> Later versions may have small API differences.

Let's look at a simpler `pl.DataFrame` where each embedding is a list with 3 values. After the embedding process, the object will look something like:

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
print(df)
```

By default, Polars casts the data type of the `'embedding'` column to a list of floats (`pl.List[pl.Float64]`). We can inspect the types after printing the DataFrame.

```output
shape: (3, 2)
┌─────┬─────────────────┐
│ id  ┆ embedding       │
│ --- ┆ ---             │
│ i64 ┆ list[f64]       │
╞═════╪═════════════════╡
│ 1   ┆ [1.0, 0.0, 0.0] │
│ 2   ┆ [0.5, 0.5, 0.0] │
│ 3   ┆ [0.0, 1.0, 0.0] │
└─────┴─────────────────┘
```

The vector representation encodes semantics and enables mathematical analysis. The [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity), for example, is commonly used to measure similarity between vectors:

$$
\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) =
\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \,\|\mathbf{b}\|} =
\frac{\sum_{i=1}^n a_i b_i}{\sqrt{\sum_{i=1}^n a_i^2} \sqrt{\sum_{i=1}^n b_i^2}}
$$

Applied to our use case, embeddings of two semantically similar texts will likely score higher than those of distinct texts. By calculating the cosine similarities for all adjacent embeddings (1->2, 2->3, ...) we can easily spot sudden drops of the scores. These drops pinpoint semantic shifts, detectable with change-point methods.

In the following, we work on the adjacent cosine similarity calculation following the  **make it work -> make it pretty -> make it fast** principle. Where we focus on providing a proof of concept before diving into optimizations using Polars-native syntax:

1. naive and simple: calculate the cosine similarity row-by-row using `map_elements` by applying a custom function.
2. optimized and vectorized: array‑based approach leveraging polars native syntax

## Naive First: Make It Work

> **When to use this?**
>
> There can be several situations in which `map_elements` and a custom function is useful
>
> - It is a great start because it is simple and readable. Using a custom function will work even when the logic is not writable in pure Polars syntax.
> - The custom function is already available and you can directly use it
> - Performance is not a bottleneck and readability/simplicity counts

Looking again at the data structure, we see that we need to get access to the embedding vectors from two rows at the same time. For the first similarity calculation for example embeddings from row 1 and row 2:

```output
shape: (3, 2)
┌─────┬─────────────────┐
│ id  ┆ embedding       │
│ --- ┆ ---             │
│ i64 ┆ list[f64]       │
╞═════╪═════════════════╡
│ 1   ┆ [1.0, 0.0, 0.0] │
│ 2   ┆ [0.5, 0.5, 0.0] │
│ 3   ┆ [0.0, 1.0, 0.0] │
└─────┴─────────────────┘
```

The easiest way for us is to create a second column `'_next'` that contains the embedding of the following row. `.shift(-1)` moves the embedding up by one row, and `.with_columns([...])` adds all the columns declared inside. Giving an `alias(...)` prevents shadowing of the existing `'embedding'` column in this case.

```python
df = df.with_columns([pl.col("embedding").shift(-1).alias("_next")])
print(df)
```

This trick allows us to conveniently access both embeddings within a single row. Further note that the last row, receives a `null` entry due to a missing successor.

```output
shape: (3, 3)
┌─────┬─────────────────┬─────────────────┐
│ id  ┆ embedding       ┆ _next           │
│ --- ┆ ---             ┆ ---             │
│ i64 ┆ list[f64]       ┆ list[f64]       │
╞═════╪═════════════════╪═════════════════╡
│ 1   ┆ [1.0, 0.0, 0.0] ┆ [0.5, 0.5, 0.0] │
│ 2   ┆ [0.5, 0.5, 0.0] ┆ [0.0, 1.0, 0.0] │
│ 3   ┆ [0.0, 1.0, 0.0] ┆ null            │
└─────┴─────────────────┴─────────────────┘
```

Now we can iterate over individual rows using `map_elements()` without accessing multiple rows at once. We use the two vectors to call the custom similarity `_cosine_py(vec1, vec2)` function, which is defined later.

```python
df = df.with_columns(
    [
        # select required columns as struct. Allows dict-like usage in lambda
        pl.struct(("embedding", "_next"))
        .map_elements(  # iteration over entries (comparable to pandas .apply() )
            # parameterize the function for calculation of the cosine similarity
            lambda s: _cosine_py(s["embedding"], s["_next"]),
            return_dtype=pl.Float32,  # specify the return
            returns_scalar=True,
        )
        .alias("cosine_similarity")
    ]
)
```

`map_elements()` expects a function that is called for every row. We need to take care of the function arguments. Selecting both vector columns, using `pl.struct()` is useful to for explicit and dict-like retrieval of the vectors. We simply wrap it inside a `lambda` to parameterize the cosine function call.

With the steps above in mind, we can compose the `cosine_adjacent_naive(df)` function that orchestrates the similarity calculation based on the DataFrame.

```python
import polars as pl
import numpy as np

def _cosine_py(vec_1: list[float] | None, vec_2: list[float] | None) -> float | None:
    # handle cases where a vector is missing (eg. last entry because of shift)
    if vec_1 is None or vec_2 is None:
        return None

    # cosine similarity calculation with numpy
    v1 = np.array(vec_1, dtype=float)
    v2 = np.array(vec_2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    # escape division error
    if n1 == 0 or n2 == 0:
        return None

    return float(np.dot(v1, v2) / (n1 * n2))


def cosine_adjacent_naive(df: pl.DataFrame) -> pl.DataFrame:
    # shift the embedding before calculation of the vector similarity
    # with the next row element
    df = df.with_columns([pl.col("embedding").shift(-1).alias("_next")])

    df = df.with_columns(
        [
            # select required columns as struct. Allows dict-like usage in lambda
            pl.struct(("embedding", "_next"))
            .map_elements(  # iteration over entries (comparable to pandas .apply() )
                # parameterize the function for calculation of the cosine similarity
                lambda s: _cosine_py(s["embedding"], s["_next"]),
                return_dtype=pl.Float32,  # specify the return
                returns_scalar=True,
            )
            .alias("cosine_similarity")
        ]
    # cleanup of temporary series 
    ).drop("_next")
    return df
```

Let's construct an example to prove our concept of the change point detection with following DataFrame.

```python
df = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6],
        "embedding": [
            [1.0, 0.8, 0.0],
            [0.7, 0.9, 0.0],
            [0.9, 0.7, 0.0],
            [0.1, 0.2, 0.9],  # NOTE change
            [0.0, 0.1, 0.7],
            [0.1, 0.1, 0.7],
        ],
    }
)

naive = cosine_adjacent_naive(df)

print(naive.select(["id", "cosine_similarity"]))
```

Inspecting `'cosine_similarity'`, the drop from 0.97 to 0.21 at row 3 stands out. It was calculated for the pair 3->4. We can detect it visually already and could further think of algorithms to programmatically mark areas of interest.

The `null` in the last output row is due to the missing successor and doesn't need to concern us.

```output
shape: (6, 2)
┌─────┬───────────────────┐
│ id  ┆ cosine_similarity │
│ --- ┆ ---               │
│ i64 ┆ f32               │
╞═════╪═══════════════════╡
│ 1   ┆ 0.972511          │
│ 2   ┆ 0.969231          │
│ 3   ┆ 0.217524          │
│ 4   ┆ 0.991241          │
│ 5   ┆ 0.990148          │
│ 6   ┆ null              │
└─────┴───────────────────┘
```

Great! We proved that the concept works and have written an algorithm to calculate the similarity scores for us. We know the pros and cons: the logic is readable and maintainable, but slow due to Python iteration.

After asking a few questions:

- Is the code fast enough for my purposes already?
- How often do I need to run the processing?
- Can I rewrite the logic optimized in Polars before the deadline?

We can either finish up our solution, or we can move ahead to optimize the runtime.

## Make It Pretty and Fast: Vectorized Arrays

> **When to use this?**
>
> The vector calculation is based on the data type `pl.Array` and the `expr.arr` attribute.
>
> - The [series must be convertible to `pl.Array`](https://docs.pola.rs/user-guide/expressions/lists-and-arrays/#the-data-type-array). Requires homogeneous type and uniform length in all fields.
> - Performance and/or memory consumption counts

To unlock the performance and memory potential of Polars, we need to take care of the embedding column's data type. As stated initially the list of floats is cast to `pl.List[pl.Float64]` by default. Unfortunately Polars doesn't infer the `pl.Array` type automatically, which can be used for homogeneous types of fixed-size containers. And this is exactly what applies to our embeddings. Since all vectors were generated by the same model and processing, the embeddings qualify perfectly as arrays.

We can cast from list to array using the expression `expr.list.to_array(len)`. Therefore we extract the length of a single embedding vector. And convert the data type to array.

Before the implementation, let’s understand the required [array expressions](https://docs.pola.rs/api/python/stable/reference/expressions/array.html) using another similar DataFrame.

```python
mf = pl.DataFrame(
    {
        "id": [1, 2],
        "embedding": [
            [1.0, 2.0, 3.0],
            [10.0, 10.0, 10.0],
        ],
    }
)

# retrieve the length of the first vector for conversion
embedding_length = mf.head(1).select(pl.col("embedding")).item().len()

# cast the list to array type
mf = mf.with_columns([pl.col("embedding").list.to_array(embedding_length)])

# explore linear algebra using pl.Array
res = mf.with_columns(
    [
        # element-wise multiplication v1^2 , v2^2 , v3^2
        (pl.col("embedding") * pl.col("embedding")).alias("element_wise"),

        # same as above + summing up the vector's elements -> dot product
        (pl.col("embedding") * pl.col("embedding")).arr.sum().alias("dot"),
        
    ]
).with_columns(
    [
        # the second with_columns allows to reference precomputed and aliased columns
        pl.col('dot').sqrt().alias("norm"),
    ]
)
print(res)
```

The resulting DataFrame shows how to perform common vector calculations (such as element-wise multiplications) or array aggregations (such as the dot product).

```output
shape: (2, 5)
┌─────┬────────────────────┬───────────────────────┬───────┬───────────┐
│ id  ┆ embedding          ┆ element_wise          ┆ dot   ┆ norm      │
│ --- ┆ ---                ┆ ---                   ┆ ---   ┆ ---       │
│ i64 ┆ array[f64, 3]      ┆ array[f64, 3]         ┆ f64   ┆ f64       │
╞═════╪════════════════════╪═══════════════════════╪═══════╪═══════════╡
│ 1   ┆ [1.0, 2.0, 3.0]    ┆ [1.0, 4.0, 9.0]       ┆ 14.0  ┆ 3.741657  │
│ 2   ┆ [10.0, 10.0, 10.0] ┆ [100.0, 100.0, 100.0] ┆ 300.0 ┆ 17.320508 │
└─────┴────────────────────┴───────────────────────┴───────┴───────────┘
```

With the array syntax in mind, we can compose a clean polars-native expression to calculate the similarity scores. Let us look at the function `cosine_adjacent_vectorized(df)`. It is composed of multiple steps, where each step has a distinct purpose inside the corresponding `with_columns(...)` block:

1. row shift making both vectors available
2. unlocking array potential via a cast `expr.list.to_array()`
3. precomputation of formula variables using linear algebra through `expr.arr` syntax
4. cosine similarity calculation

```python
import polars as pl

def cosine_adjacent_vectorized(
    df: pl.DataFrame, out_dtype: pl.DataType = pl.Float32
) -> pl.DataFrame:
    embedding_length = df.head(1).select(pl.col("embedding")).item().len()

    similarity = (
        df.with_columns(
            # 1) row shift
            # shift the embedding before calculation of the similarity
            # with the next row element
            # columns can be accessed in the next with_columns block
            [
                pl.col("embedding").alias("_current_em"),
                pl.col("embedding").shift(-1).alias("_next_em"),
            ]
        )
        .with_columns(
            # 2) unlocking array potential via a cast `expr.list.to_array()`
            # convert both embedding columns to array this allows vectorized operations
            # in the the following with_columns section
            [
                pl.col("_current_em")
                .list.to_array(embedding_length)
                .alias("_current_em"),
                pl.col("_next_em").list.to_array(embedding_length).alias("_next_em"),
            ]
        )
        .with_columns(

            # 3) precomputation of formula variables 
            [
                # # dot product
                # element-wise calculation
                (pl.col("_current_em") * pl.col("_next_em"))
                # cast to array & sum up the elements
                .arr.sum()
                .alias("_dot"),

                # #  current norm 
                # element-wise calculation
                (pl.col("_current_em") * pl.col("_current_em"))
                .arr.sum()
                # eventually compute the square root
                .sqrt()
                .alias("_current_norm"),

                # # next norm
                # analog to above
                (pl.col("_next_em") * pl.col("_next_em"))
                .arr.sum()
                .sqrt()
                .alias("_next_norm"),
            ]
        )
        .with_columns(

            # 4) cosine similarity calculation
            [
                # plug in the precomputed values
                (pl.col("_dot") / (pl.col("_current_norm") * pl.col("_next_norm")))
                .cast(out_dtype)
                .alias("cosine_similarity")
            ]
        )
    # cleanup of temporary series 
    ).drop("_next_em", "_next_norm", "_current_em", "_current_norm", "_dot") 
    return similarity
```

Now it's time to put the function into action and compare it to `cosine_adjacent_naive()`:

```python
import polars as pl

df = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6],
        "embedding": [
            [1.0, 0.8, 0.0],
            [0.7, 0.9, 0.0],
            [0.9, 0.7, 0.0],
            [0.1, 0.2, 0.9],  # NOTE
            [0.0, 0.1, 0.7],
            [0.1, 0.1, 0.7],
        ],
    }
)

opt = cosine_adjacent_vectorized(df)

print(opt.select(["id", "cosine_similarity"]))
```

Comparing the vectorized output block with the naive version above, we see identical similarity scores. And can conclude that both functions produce the same results.

```output
shape: (6, 2)
┌─────┬───────────────────┐
│ id  ┆ cosine_similarity │
│ --- ┆ ---               │
│ i64 ┆ f32               │
╞═════╪═══════════════════╡
│ 1   ┆ 0.972511          │
│ 2   ┆ 0.969231          │
│ 3   ┆ 0.217524          │
│ 4   ┆ 0.991241          │
│ 5   ┆ 0.990148          │
│ 6   ┆ null              │
└─────┴───────────────────┘
```

Great again - the Polars-native function works! It uses the efficient `pl.Array` data type and uses `expr.arr` to perform matrix calculations. So it should be faster, right?

Let’s benchmark whether shifting computation from Python to Polars’ Rust engine performs better.

## Benchmarking

In a direct comparison, the `cosine_adjacent_vectorized(df)` clearly outperforms `cosine_adjacent_naive(df)`. It was benchmarked on my laptop computer with a dataset of 1,000,000 entries and embeddings of length 500, for a total of 3 runs.

```output
Preparing DataFrame with 1,000,000 rows and embeddings of length 500.
Data preparation took 8.98s

Running benchmark (3 run(s) per method)
vectorized | avg=2.81s best=2.04s worst=3.50s rows/s=355,696
     naive | avg=48.11s best=46.84s worst=49.97s rows/s=20,785
Benchmark complete.
```

The Polars-native approach executes ~17 times faster on average. Depending on the size of the datasets you are processing, the gains pay off: 1 second vs 17 seconds, 1 minute vs 17 minutes, 1 hour vs 17 hours.

## Key Ideas to Reuse

Beyond the speedup, this case shows general problem-solving strategies and highlights Polars’ array toolbox.

- **Make it work → make it pretty → make it fast**: Create value early on with the approach you can realize the fastest. Improve if you need to.
- **prefer vectorized over iterations**: cast data to `pl.Array` for vector and matrix operations for homogeneous and fixed size data
- Keep computations **explicit and composable**: it’s easier to reason about, debug, and optimize.
