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
    "small": {"n": 64, "d": 64, "k": 32, "out": 16},
    "large": {"n": 4096, "d": 2048, "k": 1024, "out": 512},
}

def reset_preflight():
    PreflightNP.graph = PreflightGraph([], [], [])

def resolved_matmul_dims(args) -> dict[str, int]:
    preset = MATMUL_PRESETS[args.preset]
    return {
        "n": args.n if args.n is not None else preset["n"],
        "d": args.d if args.d is not None else preset["d"],
        "k": args.k if args.k is not None else preset["k"],
        "out": args.out if args.out is not None else preset["out"],
    }

def matmul_dim(args, name: str, *, concrete: bool) -> Dim:
    values = resolved_matmul_dims(args)
    value = values[name] if concrete or getattr(args, name) is not None else None
    return Dim(name, value)

def matmul_input_specs(args, *, concrete: bool = False):
    n = matmul_dim(args, "n", concrete=concrete)
    d = matmul_dim(args, "d", concrete=concrete)
    k = matmul_dim(args, "k", concrete=concrete)
    out = matmul_dim(args, "out", concrete=concrete)
    w2_rows = d if args.w2_rows == "d" else k
    return (
        ("x", (n, d)),
        ("w1", (d, d)),
        ("w2", (w2_rows, out)),
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
    return tuple(MetaArray(shape, "float32", name) for name, shape in matmul_input_specs(args, concrete=True))


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
    print(f"input shape specs: {x}; {w1}; {w2}; iters={args.iters}")
    out = timed("PREFLIGHT model-only", lambda: matmul_program(x, w1, w2, np=PreflightNP, iters=args.iters))
    PreflightNP.graph.report()
    print_matmul_preflight_report(args)
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

def print_matmul_preflight_report(args):
    matmul_obligations = [o for o in PreflightNP.graph.obligations if o.op == "matmul"]
    if not matmul_obligations:
        return

    print("\npreflight report:")
    values = resolved_matmul_dims(args)
    for obligation in matmul_obligations:
        node = PreflightNP.graph.nodes[obligation.node]
        lhs = node.attrs.get("contract_lhs")
        rhs = node.attrs.get("contract_rhs")
        if isinstance(lhs, Dim) and isinstance(rhs, Dim):
            print(f"  potential shape failure at matmul node [{obligation.node}]")
            print(f"    required: {lhs.expr} == {rhs.expr}")
            print("    reason: input specs do not guarantee that relation")
            if lhs.expr in values and rhs.expr in values and values[lhs.expr] != values[rhs.expr]:
                print(f"    example failing assignment: {lhs.expr} = {values[lhs.expr]}, {rhs.expr} = {values[rhs.expr]}")
        else:
            print(f"  potential shape failure at matmul node [{obligation.node}]: {obligation.message}")

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
        help="trace matrix code over symbolic input specs",
        allow_abbrev=False,
    )
    matmul.add_argument(
        "--run",
        choices=("preflight", "execute"),
        default="preflight",
        help="preflight runs over metadata only; execute allocates real NumPy arrays",
    )
    matmul.add_argument(
        "--w2-rows",
        choices=("d", "k"),
        default="d",
        help="symbol used for w2 rows; k leaves final h @ w2 underconstrained because it requires d == k",
    )
    matmul.add_argument(
        "--preset",
        choices=tuple(MATMUL_PRESETS),
        default="small",
        help="concrete dimension preset for execute runs and counterexample values",
    )
    matmul.add_argument("--n", type=int, help="assign concrete value for input rows n")
    matmul.add_argument("--d", type=int, help="assign concrete value for hidden width d")
    matmul.add_argument(
        "--k",
        type=int,
        help="assign concrete value for alternate w2 row symbol k",
    )
    matmul.add_argument(
        "--out",
        type=int,
        help="assign concrete value for output width out",
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
