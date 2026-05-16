# Shape Preflight

Catch array shape errors before running expensive NumPy code.

This is a small proof of concept for running one NumPy-style program in two modes:

1. preflight: cheap metadata arrays record shape facts;
2. execute: real NumPy arrays run the same computation.

The matrix example is ordinary NumPy-style code:

```python
def matmul_program(x, w1, w2, *, iters=100, np=np):
    h = x
    for _ in range(iters):
        h = np.maximum(h @ w1, 0)
    return h @ w2
```

In preflight mode, `MetaArray.__matmul__` intercepts `h @ w1`, and the `PreflightNP` proxy handles `np.maximum(...)`. No numerical arrays are allocated for that path.

## Try It

```bash
uv run shape_preflight.py vector foo bar baz
uv run shape_preflight.py vector --case empty
uv run shape_preflight.py matmul --run preflight --case ok --iters 3
uv run shape_preflight.py matmul --run preflight --case bad-final-inner --iters 500
uv run shape_preflight.py matmul --run execute --case bad-final-inner --iters 500
```

The vector demo shows runtime length names and simple dimension arithmetic. `cli_args` represents the simulated command-line arguments. The tool models `foo bar baz` as a metadata vector with dtype `String` and symbolic length `argc = 3`, printed as `Vec<String, argc=3>`. With the default `--more-len 2`, `concat0` computes `argc + m = 5`, printed as `(argc+m)=5`.

Excerpt:

```
parse:   nums: Vec<Int, argc=3>
concat0: concat0(nums,more): Vec<Int, (argc+m)=5>
tail:    tail(nums): Vec<Int, (argc-1)=2>
filter:  filter(nums): Vec<Int, ?filter(nums)>

preflight status: ok
```

The empty case makes `argc=0`, so `head(nums)` fails before any real computation:

```
violations:
  [4] head: requires argc=0 > 0

preflight status: failed
```

The matrix failure is caught after tracing the structural path:

```
violations:
  [1003] matmul: inner dimension requires d=2048 == k=1024

preflight status: failed
```

Example timings on my machine with `--iters 500`:

```
PREFLIGHT model-only: error elapsed=0.001406s
EXECUTE model-only: error elapsed=14.181211s
```

## Code Map

- [`vector_program`](shape_preflight.py#L11): demo for symbolic vector lengths.
- [`matmul_program`](shape_preflight.py#L18): user code written in normal NumPy style.
- [`MetaArray.__matmul__`](shape_preflight.py#L75): proxy hook for `@`.
- [`PreflightNP`](shape_preflight.py#L141): metadata-only NumPy-ish proxy.
- [`matmul_input_specs`](shape_preflight.py#L227): shared matrix shape setup for preflight and execute.

## Status

This is not a dependent type system, a static type checker, or a proof of shape correctness. It is symbolic-lite abstract execution: dimensions have names and runtime values, and compatibility is checked using available concrete values.

Future work could replace the stringy `Dim` layer and direct equality checks with symbolic constraints plus an SMT solver.
