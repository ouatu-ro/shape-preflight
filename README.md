# Shape Preflight

Catch array shape errors before running expensive NumPy code.

This is a small proof of concept for running one NumPy-style program in two modes:

1. preflight: cheap metadata arrays record shape facts in a typed preflight DAG;
2. execute: real NumPy arrays run the same computation.

The matrix example is ordinary NumPy-style code:

```python
def matmul_program(x, w1, w2, *, iters=100, np=np):
    h = x
    for _ in range(iters):
        h = np.maximum(h @ w1, 0)
    return h @ w2
```

In preflight mode, `MetaArray.__matmul__` intercepts `h @ w1`, and the `PreflightNP` proxy handles `np.maximum(...)`. No numerical arrays are allocated for that path; operations, dependencies, output metadata, violations, and unresolved obligations are recorded as structured DAG nodes.

## Try It

```bash
uv run shape_preflight.py vector foo bar baz
uv run shape_preflight.py vector --case empty
uv run shape_preflight.py matmul --run preflight --case ok --iters 3
uv run shape_preflight.py matmul --run preflight --case bad-final-inner --iters 500
uv run shape_preflight.py matmul --run execute --case bad-final-inner --iters 500
```

The matrix demo defaults to a small preset, so execute mode is safe to try. Use `--preset large` when you specifically want the expensive-array version of the demo.

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
preflight DAG: 1004 nodes, showing last 8
  [1003] matmul(1002, 2)

violations:
  [1003] matmul: inner dimension requires d=64 == k=32

preflight status: failed
```

## DOT Export

Preflight runs can emit a Graphviz DOT representation of the typed preflight DAG:

```bash
uv run shape_preflight.py matmul --run preflight --case bad-final-inner --iters 3 --dot-file shape.dot
dot -Tsvg shape.dot -o shape.svg
```

Use `--dot` to print DOT to stdout. The DOT contains operation nodes, input edges, output shape metadata, and any violations or unresolved obligations. Use small traces such as `--iters 3` for readable diagrams.

Example timings on my machine with `--preset large --iters 500`:

```
PREFLIGHT model-only: error elapsed=0.001406s
EXECUTE model-only: error elapsed=14.181211s
```

## Code Map

- [`shape_preflight.py`](shape_preflight.py): user programs, demo runners, and CLI.
- [`preflight_core.py`](preflight_core.py): metadata arrays, dimension logic, preflight DAG, and proxy backend.
- [`vector_program`](shape_preflight.py#L15): demo for symbolic vector lengths.
- [`matmul_program`](shape_preflight.py#L23): user code written in normal NumPy style.
- [`broadcast_shapes`](preflight_core.py#L58): minimal NumPy-style broadcasting for metadata arrays.
- [`MetaArray.__matmul__`](preflight_core.py#L97): proxy hook for `@`.
- [`PreflightNode`](preflight_core.py#L132) and [`PreflightGraph`](preflight_core.py#L143): structured preflight DAG.
- [`PreflightGraph.to_dot`](preflight_core.py#L237): DOT exporter.
- [`PreflightNP`](preflight_core.py#L272): metadata-only NumPy-ish proxy.
- [`matmul_input_specs`](shape_preflight.py#L53): shared matrix shape setup for preflight and execute.

## Status

This is not a dependent type system, a static type checker, or a proof of shape correctness. It is symbolic-lite abstract execution over the executed path: dimensions have names and runtime values, known contradictions become violations, and unknown-but-required relationships become unresolved obligations.

Current limitations: `head` returns the scalar head as a dtype string rather than a scalar metadata value, and `maximum` handles broadcasting shape flow but not full NumPy ufunc dtype promotion or value semantics. Future work could replace the stringy `Dim` layer with symbolic constraints plus an SMT solver.
