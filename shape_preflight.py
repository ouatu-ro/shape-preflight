from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from preflight_core import Dim, MetaArray, PreflightGraph, PreflightNP, concrete_shape

# ---------------------------------------------------------------------
# User programs: ordinary program logic, written once.
# ---------------------------------------------------------------------

def vector_program(cli_args, more, *, np):
    nums = np.parse_ints(cli_args)
    both = np.concat0(nums, more)
    x, tail = np.head(nums)
    kept = np.filter(nums)
    return nums, both, x, tail, kept

def matmul_program(x, w1, w2, *, iters=100, np:Any=np):
    h = x
    for _ in range(iters):
        h = np.maximum(h @ w1, 0)
    return h @ w2


# ---------------------------------------------------------------------
# Demo runners: construct inputs, run programs, print reports.
# ---------------------------------------------------------------------

MATMUL_PRESETS = {
    "small": (64, 64, 32),
    "large": (4096, 2048, 1024),
}

def reset_preflight():
    PreflightNP.graph = PreflightGraph([], [], [])

def resolved_matmul_dims(args):
    n, d, k = MATMUL_PRESETS[args.preset]
    return (
        args.n if args.n is not None else n,
        args.d if args.d is not None else d,
        args.k if args.k is not None else k,
    )

def matmul_input_specs(args):
    n_value, d_value, k_value = resolved_matmul_dims(args)
    n, d, k = Dim("n", n_value), Dim("d", d_value), Dim("k", k_value)
    w2_rows = k if args.case == "bad-final-inner" else d
    return (
        ("x", (n, d)),
        ("w1", (d, d)),
        ("w2", (w2_rows, k)),
    )

def run_vector_demo(args):
    reset_preflight()
    item_values = [] if args.case == "empty" else args.items
    cli_args = PreflightNP.input("cli_args", (Dim("argc", len(item_values)),), "String")
    more = PreflightNP.input("more", (Dim("m", args.m),), "Int")
    print(f"runtime argc: {cli_args.shape[0].value}")
    try:
        nums, both, x, tail, kept = vector_program(cli_args, more, np=PreflightNP)
        print("input: ", cli_args)
        print("parse: ", nums)
        print("concat0:", both)
        print("head:  ", x)
        print("tail:  ", tail)
        print("filter:", kept)
    except TypeError as e:
        print(f"error:  TypeError: {e}")
    finally:
        PreflightNP.graph.report()
        PreflightNP.graph.status()
        emit_dot(args)

def matmul_meta_inputs(args):
    return tuple(MetaArray(shape, "float32", name) for name, shape in matmul_input_specs(args))


def materialize_numpy(meta):
    return np.random.randn(*concrete_shape(meta.shape)).astype(np.float32)

def timed(label, fn):
    start = time.perf_counter()
    try:
        out = fn()
    except Exception as e:
        print(f"{label}: error elapsed={time.perf_counter() - start:.6f}s")
        print(f"{type(e).__name__}: {e}")
        return None
    print(f"{label}: ok elapsed={time.perf_counter() - start:.6f}s")
    return out

def run_matmul_preflight_demo(args):
    reset_preflight()
    x, w1, w2 = (PreflightNP.input(name, shape) for name, shape in matmul_input_specs(args))
    print(f"preflight inputs: {x}; {w1}; {w2}; iters={args.iters}")
    out = timed("PREFLIGHT model-only", lambda: matmul_program(x, w1, w2, np=PreflightNP, iters=args.iters))
    PreflightNP.graph.report()
    PreflightNP.graph.status()
    if out is not None:
        print("planned output:", out)
    emit_dot(args)

def run_matmul_execute_demo(args):
    x, w1, w2 = matmul_meta_inputs(args)
    print("allocating real NumPy arrays outside timer...")
    xr, w1r, w2r = materialize_numpy(x), materialize_numpy(w1), materialize_numpy(w2)
    print(f"real inputs: x={xr.shape} w1={w1r.shape} w2={w2r.shape} iters={args.iters}")
    out = timed("EXECUTE model-only", lambda: matmul_program(xr, w1r, w2r, iters=args.iters))
    if out is not None:
        print("real output:", out.shape, out.dtype)
    if args.dot or args.dot_file:
        print("DOT export is only available for preflight runs.")

def run_matmul_demo(args):
    if args.run == "preflight":
        return run_matmul_preflight_demo(args)
    return run_matmul_execute_demo(args)

def emit_dot(args):
    dot = None
    if args.dot_file:
        dot = PreflightNP.graph.to_dot()
        with open(args.dot_file, "w", encoding="utf-8") as f:
            f.write(dot)
            f.write("\n")
        print(f"\nwrote DOT: {args.dot_file}")
    if args.dot:
        if dot is None:
            dot = PreflightNP.graph.to_dot()
        print("\nDOT:")
        print(dot)

# ---------------------------------------------------------------------
# CLI: parse command-line args and dispatch to demo runners.
# ---------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Run tiny shape-preflight demos over metadata arrays and NumPy arrays.",
        allow_abbrev=False,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    vector = sub.add_parser(
        "vector",
        help="demo symbolic vector lengths through parse, concat0, head, and filter",
        allow_abbrev=False,
    )
    vector.add_argument("items", nargs="*", help="strings parsed into nums; argc = number of items")
    vector.add_argument(
        "--case",
        choices=("ok", "empty"),
        default="ok",
        help="demo scenario: ok uses the given items; empty forces argc=0 so head(nums) fails",
    )
    vector.add_argument(
        "--more-len",
        dest="m",
        type=int,
        default=2,
        help="length m of the second vector used by concat0(nums, more)",
    )
    add_dot_args(vector)
    matmul = sub.add_parser(
        "matmul",
        help="demo early shape failure for repeated matrix multiplication",
        allow_abbrev=False,
    )
    matmul.add_argument(
        "--run",
        choices=("preflight", "execute"),
        default="preflight",
        help="preflight runs over metadata only; execute allocates real NumPy arrays",
    )
    matmul.add_argument(
        "--case",
        choices=("ok", "bad-final-inner"),
        default="ok",
        help="bad-final-inner makes w2 rows = k instead of d, so final h @ w2 violates d == k",
    )
    matmul.add_argument(
        "--preset",
        choices=tuple(MATMUL_PRESETS),
        default="small",
        help="dimension preset; small is safe for casual execute runs, large shows the expensive-work motivation",
    )
    matmul.add_argument("--n", type=int, help="override batch/input rows: x has shape Mat<float32, n, d>")
    matmul.add_argument("--d", type=int, help="override hidden width: x columns, w1 rows/cols, and valid w2 rows")
    matmul.add_argument(
        "--k",
        type=int,
        help="override output width: valid w2 has shape Mat<float32, d, k>; bad case uses Mat<float32, k, k>",
    )
    matmul.add_argument(
        "--iters",
        type=int,
        default=100,
        help="number of repeated h = maximum(h @ w1, 0) steps before final h @ w2",
    )
    add_dot_args(matmul)
    return parser

def add_dot_args(parser):
    parser.add_argument("--dot", action="store_true", help="print Graphviz DOT for the preflight DAG")
    parser.add_argument("--dot-file", help="write Graphviz DOT for the preflight DAG to PATH", metavar="PATH")

def main():
    parser = build_parser()
    args = parser.parse_args()
    runners = {"vector": run_vector_demo, "matmul": run_matmul_demo}
    runners[args.cmd](args)

if __name__ == "__main__":
    main()
