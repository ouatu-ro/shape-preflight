from __future__ import annotations
from dataclasses import dataclass
import argparse
import time
from typing import Any
import numpy as np

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
# SDK sketch: symbolic-lite dimensions and metadata arrays.
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Dim:
    expr: str
    value: int | None = None
    def __add__(self, other):
        other = dim(other)
        value = self.value + other.value if self.value is not None and other.value is not None else None
        return Dim(f"({self.expr}+{other.expr})", value)
    def __sub__(self, other):
        other = dim(other)
        value = self.value - other.value if self.value is not None and other.value is not None else None
        if value is not None and value < 0:
            raise ValueError(f"negative dimension: {self} - {other} = {value}")
        return Dim(f"({self.expr}-{other.expr})", value)
    def __repr__(self):
        if self.value is None:
            return self.expr
        if self.expr == str(self.value):
            return str(self.value)
        return f"{self.expr}={self.value}"

def dim(x):
    if isinstance(x, Dim):
        return x
    if isinstance(x, int):
        return Dim(str(x), x)
    raise TypeError(f"not a dimension: {x!r}")

def runtime_compatible(a, b):
    if a.value is not None and b.value is not None:
        return a.value == b.value
    return a.expr == b.expr

def concrete_shape(shape):
    vals = []
    for d in shape:
        if d.value is None:
            raise ValueError(f"cannot materialize symbolic dimension {d}")
        vals.append(d.value)
    return tuple(vals)

@dataclass(frozen=True)
class MetaArray:
    shape: tuple[Dim, ...]
    dtype: str = "float32"
    name: str = "?"
    node: int = -1
    def __matmul__(self, other):
        return PreflightNP.matmul(self, other)
    def __rmatmul__(self, other):
        return PreflightNP.matmul(other, self)
    @property
    def ndim(self):
        return len(self.shape)
    def __repr__(self):
        if self.ndim == 0:
            return f"{self.name}: {self.dtype}"
        if self.ndim == 1:
            return f"{self.name}: Vec<{self.dtype}, {self.shape[0]}>"
        if self.ndim == 2:
            return f"{self.name}: Mat<{self.dtype}, {self.shape[0]}, {self.shape[1]}>"
        return f"{self.name}: Array<{self.dtype}, shape={self.shape}>"

@dataclass(frozen=True)
class Violation:
    node: int
    op: str
    message: str

@dataclass(frozen=True)
class Obligation:
    node: int
    op: str
    message: str

@dataclass
class Graph:
    nodes: list[str]
    violations: list[Violation]
    obligations: list[Obligation]
    def add(self, text: str) -> int:
        node = len(self.nodes)
        self.nodes.append(text)
        return node
    def violate(self, node: int, op: str, message: str):
        self.violations.append(Violation(node, op, message))
    def require(self, node: int, op: str, message: str):
        self.obligations.append(Obligation(node, op, message))
    def report(self, last=8):
        shown = min(last, len(self.nodes))
        print(f"\nshape graph: {len(self.nodes)} nodes, showing last {shown}")
        for i, text in list(enumerate(self.nodes))[-last:]:
            print(f"  [{i}] {text}")
        if self.obligations:
            print("\nunresolved obligations:")
            for o in self.obligations:
                print(f"  [{o.node}] {o.op}: {o.message}")
        if self.violations:
            print("\nviolations:")
            for v in self.violations:
                print(f"  [{v.node}] {v.op}: {v.message}")
    def status(self):
        if self.violations:
            print("\npreflight status: failed")
        elif self.obligations:
            print("\npreflight status: unresolved obligations")
        else:
            print("\npreflight status: ok")

# ---------------------------------------------------------------------
# SDK sketch: metadata-only NumPy-ish backend.
# ---------------------------------------------------------------------

class PreflightNP:
    graph = Graph([], [], [])
    @staticmethod
    def input(name, shape, dtype="float32"):
        arr = MetaArray(shape, dtype, name)
        node = PreflightNP.graph.add(f"input {arr}")
        return MetaArray(shape, dtype, name, node)
    @staticmethod
    def matmul(a, b):
        node = PreflightNP.graph.add(f"matmul({a.name}, {b.name})")
        if a.ndim != 2 or b.ndim != 2:
            PreflightNP.graph.violate(node, "matmul", f"expected matrices, got {a} and {b}")
            raise TypeError(f"matmul expects matrices, got {a} and {b}")
        n, m = a.shape
        m2, k = b.shape
        if not runtime_compatible(m, m2):
            msg = f"inner dimension requires {m} == {m2}"
            PreflightNP.graph.violate(node, "matmul", msg)
            raise TypeError(f"bad matmul: {a.shape} @ {b.shape}; {msg}")
        out = MetaArray((n, k), a.dtype, f"matmul_{node}", node)
        PreflightNP.graph.nodes[node] += f" -> {out}"
        return out
    @staticmethod
    def maximum(a, b):
        # Demo-only: scalar maximum, no NumPy broadcasting model.
        node = PreflightNP.graph.add(f"maximum({a.name}, {b})")
        out = MetaArray(a.shape, a.dtype, f"relu_{node}", node)
        PreflightNP.graph.nodes[node] += f" -> {out}"
        return out
    @staticmethod
    def parse_ints(strings):
        node = PreflightNP.graph.add(f"parse_ints({strings.name})")
        if strings.ndim != 1 or strings.dtype != "String":
            PreflightNP.graph.violate(node, "parse_ints", f"expected Vec<String, n>, got {strings}")
            raise TypeError(f"parse_ints expects Vec<String, n>, got {strings}")
        out = MetaArray(strings.shape, "Int", "nums", node)
        PreflightNP.graph.nodes[node] += f" -> {out}"
        return out
    @staticmethod
    def concat0(xs, ys):
        node = PreflightNP.graph.add(f"concat0({xs.name}, {ys.name})")
        if xs.ndim != ys.ndim:
            PreflightNP.graph.violate(node, "concat0", f"rank mismatch: {xs} vs {ys}")
            raise TypeError(f"concat0 rank mismatch: {xs} vs {ys}")
        if xs.dtype != ys.dtype:
            PreflightNP.graph.violate(node, "concat0", f"dtype mismatch: {xs.dtype} vs {ys.dtype}")
            raise TypeError(f"concat0 dtype mismatch: {xs.dtype} vs {ys.dtype}")
        for i, (a, b) in enumerate(zip(xs.shape[1:], ys.shape[1:]), start=1):
            if not runtime_compatible(a, b):
                PreflightNP.graph.violate(node, "concat0", f"axis {i} requires {a} == {b}")
                raise TypeError(f"concat0 mismatch at axis {i}: {a} != {b}")
        out = MetaArray((xs.shape[0] + ys.shape[0], *xs.shape[1:]), xs.dtype, f"concat0({xs.name},{ys.name})", node)
        PreflightNP.graph.nodes[node] += f" -> {out}"
        return out
    @staticmethod
    def head(xs):
        node = PreflightNP.graph.add(f"head({xs.name})")
        if xs.ndim != 1:
            PreflightNP.graph.violate(node, "head", f"expected Vec<T, n>, got {xs}")
            raise TypeError(f"head expects Vec<T, n>, got {xs}")
        n = xs.shape[0]
        if n.value == 0:
            PreflightNP.graph.violate(node, "head", f"requires {n} > 0")
            raise TypeError(f"head requires nonempty Vec, got {xs}")
        if n.value is None:
            PreflightNP.graph.require(node, "head", f"requires {n} > 0")
        tail = MetaArray((n - 1,), xs.dtype, f"tail({xs.name})", node)
        PreflightNP.graph.nodes[node] += f" -> ({xs.dtype}, {tail})"
        return xs.dtype, tail
    @staticmethod
    def filter(xs):
        node = PreflightNP.graph.add(f"filter({xs.name})")
        if xs.ndim != 1:
            PreflightNP.graph.violate(node, "filter", f"expected Vec<T, n>, got {xs}")
            raise TypeError(f"filter expects Vec<T, n>, got {xs}")
        out = MetaArray((Dim(f"?filter({xs.name})"),), xs.dtype, f"filter({xs.name})", node)
        PreflightNP.graph.nodes[node] += f" -> {out}"
        return out

# ---------------------------------------------------------------------
# Demo runners: construct inputs, run programs, print reports.
# ---------------------------------------------------------------------

def reset_preflight():
    PreflightNP.graph = Graph([], [], [])

def matmul_input_specs(args):
    n, d, k = Dim("n", args.n), Dim("d", args.d), Dim("k", args.k)
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

def run_matmul_execute_demo(args):
    x, w1, w2 = matmul_meta_inputs(args)
    print("allocating real NumPy arrays outside timer...")
    xr, w1r, w2r = materialize_numpy(x), materialize_numpy(w1), materialize_numpy(w2)
    print(f"real inputs: x={xr.shape} w1={w1r.shape} w2={w2r.shape} iters={args.iters}")
    out = timed("EXECUTE model-only", lambda: matmul_program(xr, w1r, w2r, iters=args.iters))
    if out is not None:
        print("real output:", out.shape, out.dtype)

def run_matmul_demo(args):
    if args.run == "preflight":
        return run_matmul_preflight_demo(args)
    return run_matmul_execute_demo(args)

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
    matmul.add_argument("--n", type=int, default=4096, help="batch/input rows: x has shape Mat<float32, n, d>")
    matmul.add_argument("--d", type=int, default=2048, help="hidden width: x columns, w1 rows/cols, and valid w2 rows")
    matmul.add_argument(
        "--k",
        type=int,
        default=1024,
        help="output width: valid w2 has shape Mat<float32, d, k>; bad case uses Mat<float32, k, k>",
    )
    matmul.add_argument(
        "--iters",
        type=int,
        default=100,
        help="number of repeated h = maximum(h @ w1, 0) steps before final h @ w2",
    )
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    runners = {"vector": run_vector_demo, "matmul": run_matmul_demo}
    runners[args.cmd](args)

if __name__ == "__main__":
    main()
