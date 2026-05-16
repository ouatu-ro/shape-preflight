from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

# Dimensions and shape utilities

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

Shape: TypeAlias = tuple[Dim, ...]
BroadcastResult: TypeAlias = tuple[Shape | None, list[str], str | None]

def dim(x) -> Dim:
    if isinstance(x, Dim):
        return x
    if isinstance(x, int):
        return Dim(str(x), x)
    raise TypeError(f"not a dimension: {x!r}")

class DimRelation(Enum):
    PROVEN = "proven"
    DISPROVEN = "disproven"
    UNKNOWN = "unknown"

def dim_relation(a: Dim, b: Dim) -> DimRelation:
    if a.value is not None and b.value is not None:
        return DimRelation.PROVEN if a.value == b.value else DimRelation.DISPROVEN
    if a.expr == b.expr:
        return DimRelation.PROVEN
    return DimRelation.UNKNOWN

def dim_is_one(x: Dim) -> bool:
    return x.value == 1

def broadcast_shapes(lhs: Shape, rhs: Shape) -> BroadcastResult:
    out: list[Dim] = []
    obligations: list[str] = []
    max_rank = max(len(lhs), len(rhs))
    for offset in range(1, max_rank + 1):
        a = lhs[-offset] if offset <= len(lhs) else None
        b = rhs[-offset] if offset <= len(rhs) else None
        if a is None:
            assert b is not None
            out.append(b)
        elif b is None or dim_is_one(b):
            out.append(a)
        elif dim_is_one(a):
            out.append(b)
        else:
            relation = dim_relation(a, b)
            if relation is DimRelation.DISPROVEN:
                return None, obligations, f"broadcast axis -{offset} requires {a} == {b} or one dimension is 1"
            out.append(a)
            if relation is DimRelation.UNKNOWN:
                obligations.append(f"broadcast axis -{offset} requires {a} == {b} or one dimension is 1")
    return tuple(reversed(out)), obligations, None

def concrete_shape(shape: Shape) -> tuple[int, ...]:
    vals: list[int] = []
    for d in shape:
        if d.value is None:
            raise ValueError(f"cannot materialize symbolic dimension {d}")
        vals.append(d.value)
    return tuple(vals)

# Metadata arrays

@dataclass(frozen=True)
class MetaArray:
    shape: Shape
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

# Preflight DAG

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

@dataclass(frozen=True)
class PreflightNode:
    id: int
    op: str
    inputs: tuple[int, ...]
    output_name: str
    output_shape: Shape
    output_dtype: str
    attrs: dict[str, object] = field(default_factory=dict)

@dataclass
class PreflightGraph:
    nodes: list[PreflightNode]
    violations: list[Violation]
    obligations: list[Obligation]

    def add_node(
        self,
        op: str,
        inputs: tuple[MetaArray, ...],
        output_name: str,
        output_shape: Shape,
        output_dtype: str,
        attrs: dict[str, object] | None = None,
    ) -> MetaArray:
        node_id = len(self.nodes)
        output = MetaArray(output_shape, output_dtype, output_name, node_id)
        self.nodes.append(
            PreflightNode(
                id=node_id,
                op=op,
                inputs=tuple(x.node for x in inputs if x.node >= 0),
                output_name=output.name,
                output_shape=output.shape,
                output_dtype=output.dtype,
                attrs=dict(attrs or {}),
            )
        )
        return output

    def add_input(self, name: str, shape: Shape, dtype: str = "float32") -> MetaArray:
        return self.add_node(
            op="input",
            inputs=(),
            output_name=name,
            output_shape=shape,
            output_dtype=dtype,
            attrs={"name": name},
        )

    def add_failed_op(
        self,
        op: str,
        inputs: tuple[MetaArray, ...],
        attrs: dict[str, object] | None = None,
    ) -> int:
        node_id = len(self.nodes)
        self.nodes.append(
            PreflightNode(
                id=node_id,
                op=op,
                inputs=tuple(x.node for x in inputs if x.node >= 0),
                output_name="",
                output_shape=(),
                output_dtype="",
                attrs=dict(attrs or {}),
            )
        )
        return node_id

    def violate(self, node: int, op: str, message: str):
        self.violations.append(Violation(node, op, message))
    def require(self, node: int, op: str, message: str):
        self.obligations.append(Obligation(node, op, message))
    def format_node(self, node: PreflightNode) -> str:
        args = ", ".join(str(i) for i in node.inputs)
        if node.output_name:
            output = MetaArray(node.output_shape, node.output_dtype, node.output_name, node.id)
            return f"{node.op}({args}) -> {output}"
        return f"{node.op}({args})"

    def report(self, last=8):
        shown = min(last, len(self.nodes))
        print(f"\npreflight DAG: {len(self.nodes)} nodes, showing last {shown}")
        for node in self.nodes[-last:]:
            print(f"  [{node.id}] {self.format_node(node)}")
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

    def to_dot(self) -> str:
        violations_by_node: dict[int, list[Violation]] = {}
        for violation in self.violations:
            violations_by_node.setdefault(violation.node, []).append(violation)
        obligations_by_node: dict[int, list[Obligation]] = {}
        for obligation in self.obligations:
            obligations_by_node.setdefault(obligation.node, []).append(obligation)

        lines = [
            "digraph preflight_dag {",
            "  rankdir=LR;",
            "  node [shape=box];",
            "",
        ]
        for node in self.nodes:
            label = f"{node.id}: {node.op}"
            if node.output_name:
                output = MetaArray(node.output_shape, node.output_dtype, node.output_name, node.id)
                label += f"\n{output}"
            for violation in violations_by_node.get(node.id, []):
                label += f"\nVIOLATION: {violation.message}"
            for obligation in obligations_by_node.get(node.id, []):
                label += f"\nOBLIGATION: {obligation.message}"
            lines.append(f'  n{node.id} [label="{dot_escape(label)}"];')
        for node in self.nodes:
            for input_id in node.inputs:
                lines.append(f"  n{input_id} -> n{node.id};")
        lines.append("}")
        return "\n".join(lines)

def dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

# NumPy-ish metadata backend

class PreflightNP:
    graph = PreflightGraph([], [], [])

    @staticmethod
    def input(name, shape, dtype="float32"):
        return PreflightNP.graph.add_input(name, shape, dtype)
    @staticmethod
    def matmul(a, b):
        attrs = {"lhs_shape": a.shape, "rhs_shape": b.shape}
        if a.ndim != 2 or b.ndim != 2:
            node = PreflightNP.graph.add_failed_op("matmul", (a, b), attrs)
            PreflightNP.graph.violate(node, "matmul", f"expected matrices, got {a} and {b}")
            raise TypeError(f"matmul expects matrices, got {a} and {b}")
        n, m = a.shape
        m2, k = b.shape
        attrs |= {
            "contract_lhs": m,
            "contract_rhs": m2,
        }
        relation = dim_relation(m, m2)
        if relation is DimRelation.DISPROVEN:
            node = PreflightNP.graph.add_failed_op("matmul", (a, b), attrs)
            msg = f"inner dimension requires {m} == {m2}"
            PreflightNP.graph.violate(node, "matmul", msg)
            raise TypeError(f"bad matmul: {a.shape} @ {b.shape}; {msg}")
        out = PreflightNP.graph.add_node(
            op="matmul",
            inputs=(a, b),
            output_name=f"matmul_{len(PreflightNP.graph.nodes)}",
            output_shape=(n, k),
            output_dtype=a.dtype,
            attrs=attrs,
        )
        if relation is DimRelation.UNKNOWN:
            PreflightNP.graph.require(out.node, "matmul", f"requires {m} == {m2}")
        return out
    @staticmethod
    def maximum(a, b):
        if not isinstance(a, MetaArray) and isinstance(b, MetaArray):
            return PreflightNP.maximum(b, a)
        if not isinstance(a, MetaArray):
            raise TypeError(f"maximum expects at least one metadata array, got {a!r} and {b!r}")

        inputs = (a, b) if isinstance(b, MetaArray) else (a,)
        attrs = {"rhs_shape": b.shape, "rhs_dtype": b.dtype} if isinstance(b, MetaArray) else {"rhs_scalar": b}
        output_shape: Shape = a.shape
        obligations: list[str] = []
        if isinstance(b, MetaArray):
            broadcast_shape, obligations, violation = broadcast_shapes(a.shape, b.shape)
            if violation is not None:
                node = PreflightNP.graph.add_failed_op("maximum", inputs, attrs)
                PreflightNP.graph.violate(node, "maximum", violation)
                raise TypeError(f"maximum broadcast mismatch: {violation}")
            if broadcast_shape is None:
                raise AssertionError("broadcast_shapes returned no shape without a violation")
            output_shape = broadcast_shape

        out = PreflightNP.graph.add_node(
            op="maximum",
            inputs=inputs,
            output_name=f"relu_{len(PreflightNP.graph.nodes)}",
            output_shape=output_shape,
            output_dtype=a.dtype,
            attrs=attrs,
        )
        for obligation in obligations:
            PreflightNP.graph.require(out.node, "maximum", obligation)
        return out
    @staticmethod
    def parse_ints(strings):
        if strings.ndim != 1 or strings.dtype != "String":
            node = PreflightNP.graph.add_failed_op("parse_ints", (strings,))
            PreflightNP.graph.violate(node, "parse_ints", f"expected Vec<String, n>, got {strings}")
            raise TypeError(f"parse_ints expects Vec<String, n>, got {strings}")
        return PreflightNP.graph.add_node(
            op="parse_ints",
            inputs=(strings,),
            output_name="nums",
            output_shape=strings.shape,
            output_dtype="Int",
        )
    @staticmethod
    def concat0(xs, ys):
        if xs.ndim != ys.ndim:
            node = PreflightNP.graph.add_failed_op("concat0", (xs, ys))
            PreflightNP.graph.violate(node, "concat0", f"rank mismatch: {xs} vs {ys}")
            raise TypeError(f"concat0 rank mismatch: {xs} vs {ys}")
        if xs.dtype != ys.dtype:
            node = PreflightNP.graph.add_failed_op("concat0", (xs, ys))
            PreflightNP.graph.violate(node, "concat0", f"dtype mismatch: {xs.dtype} vs {ys.dtype}")
            raise TypeError(f"concat0 dtype mismatch: {xs.dtype} vs {ys.dtype}")
        for i, (a, b) in enumerate(zip(xs.shape[1:], ys.shape[1:]), start=1):
            relation = dim_relation(a, b)
            if relation is DimRelation.DISPROVEN:
                node = PreflightNP.graph.add_failed_op("concat0", (xs, ys), {"axis": i})
                PreflightNP.graph.violate(node, "concat0", f"axis {i} requires {a} == {b}")
                raise TypeError(f"concat0 mismatch at axis {i}: {a} != {b}")
        out = PreflightNP.graph.add_node(
            op="concat0",
            inputs=(xs, ys),
            output_name=f"concat0({xs.name},{ys.name})",
            output_shape=(xs.shape[0] + ys.shape[0], *xs.shape[1:]),
            output_dtype=xs.dtype,
        )
        for i, (a, b) in enumerate(zip(xs.shape[1:], ys.shape[1:]), start=1):
            if dim_relation(a, b) is DimRelation.UNKNOWN:
                PreflightNP.graph.require(out.node, "concat0", f"axis {i} requires {a} == {b}")
        return out
    @staticmethod
    def head(xs):
        if xs.ndim != 1:
            node = PreflightNP.graph.add_failed_op("head", (xs,))
            PreflightNP.graph.violate(node, "head", f"expected Vec<T, n>, got {xs}")
            raise TypeError(f"head expects Vec<T, n>, got {xs}")
        n = xs.shape[0]
        if n.value == 0:
            node = PreflightNP.graph.add_failed_op("head", (xs,), {"head_dtype": xs.dtype})
            PreflightNP.graph.violate(node, "head", f"requires {n} > 0")
            raise TypeError(f"head requires nonempty Vec, got {xs}")
        tail = PreflightNP.graph.add_node(
            op="head",
            inputs=(xs,),
            output_name=f"tail({xs.name})",
            output_shape=(n - 1,),
            output_dtype=xs.dtype,
            attrs={"head_dtype": xs.dtype},
        )
        if n.value is None:
            PreflightNP.graph.require(tail.node, "head", f"requires {n} > 0")
        return xs.dtype, tail
    @staticmethod
    def filter(xs):
        if xs.ndim != 1:
            node = PreflightNP.graph.add_failed_op("filter", (xs,))
            PreflightNP.graph.violate(node, "filter", f"expected Vec<T, n>, got {xs}")
            raise TypeError(f"filter expects Vec<T, n>, got {xs}")
        return PreflightNP.graph.add_node(
            op="filter",
            inputs=(xs,),
            output_name=f"filter({xs.name})",
            output_shape=(Dim(f"?filter({xs.name})"),),
            output_dtype=xs.dtype,
        )
