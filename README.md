# Shape Preflight

Catch array shape problems before expensive NumPy execution.

Shape Preflight runs normal NumPy-style code once on metadata arrays instead of
real arrays. The metadata run records shape flow and reports requirements that
operations impose, such as `matmul` inner dimensions matching.

This is trace-time shape analysis: Python executes, but over symbolic metadata
values rather than numerical arrays.

## Example

```python
def matmul_program(x, w1, w2, *, iters=100, np=np):
    h = x
    for _ in range(iters):
        h = np.maximum(h @ w1, 0)
    return h @ w2
```

Given:

```text
x:  Mat<float32, n, d>
w1: Mat<float32, d, d>
w2: Mat<float32, k, out>
```

the final `h @ w2` requires `d == k`. If the input specs do not guarantee that,
preflight reports an unresolved obligation before real arrays are allocated.

## Try It

```bash
uv run shape_preflight.py matmul --w2-rows d --iters 3
uv run shape_preflight.py matmul --w2-rows k --iters 500
uv run shape_preflight.py matmul --w2-rows k --d 64 --k 32 --iters 3

uv run shape_preflight.py vector foo bar baz
uv run shape_preflight.py vector --case empty
```

Underconstrained input specs:

```text
unresolved obligations:
  [1003] matmul: requires d == k

preflight report:
  potential shape failure at matmul node [1003]
    required: d == k
    example failing assignment: d = 64, k = 32
```

Concrete contradiction:

```text
violations:
  [9] matmul: inner dimension requires d=64 == k=32
```

## Why

On large inputs, preflight can find the shape problem before NumPy does the
expensive loop:

```bash
uv run shape_preflight.py matmul --preset large --w2-rows k --iters 500
uv run shape_preflight.py matmul --preset large --run execute --w2-rows k --iters 500
```

Example timings:

```text
PREFLIGHT model-only: ok elapsed=0.001915s
EXECUTE model-only: error elapsed=14.181211s
```

## DOT

```bash
uv run shape_preflight.py matmul --w2-rows k --iters 3 --dot-file shape.dot
```

## Files

- `shape_preflight.py`: CLI and demo programs
- `preflight_core.py`: metadata arrays, dimensions, DAG, and NumPy-like backend
