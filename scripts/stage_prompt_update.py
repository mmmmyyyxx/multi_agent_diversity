"""One-time AST refactor of the prompt update method into explicit stages."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "multi_dataset_diverse_rl" / "optimization" / "prompt_update_controller.py"
STAGES = (
    ("CandidateGenerationStage", 0, 29),
    ("CheapPrescreenStage", 29, 51),
    ("CandidateEvaluationStage", 51, 61),
    ("CandidateClassificationAndRefillStage", 61, 67),
    ("ArchiveSelectionStage", 67, 80),
    ("CandidateEventStage", 80, 81),
    ("UpdateSummaryStage", 81, 90),
)


def collect_outer_locals(function: ast.AsyncFunctionDef) -> set[str]:
    names = {arg.arg for arg in function.args.args if arg.arg != "self"}

    class Collector(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            names.add(node.name)

        def visit_AsyncFunctionDef(self, node):
            names.add(node.name)

        def visit_Lambda(self, node):
            return

        def visit_ClassDef(self, node):
            names.add(node.name)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                names.add(node.id)

    collector = Collector()
    for statement in function.body:
        collector.visit(statement)
    return names


def nested_locals(node) -> set[str]:
    names = {arg.arg for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]}
    if node.args.vararg: names.add(node.args.vararg.arg)
    if node.args.kwarg: names.add(node.args.kwarg.arg)

    class Collector(ast.NodeVisitor):
        def visit_FunctionDef(self, child):
            names.add(child.name)
        visit_AsyncFunctionDef = visit_FunctionDef
        def visit_Lambda(self, child): return
        def visit_ClassDef(self, child): names.add(child.name)
        def visit_Name(self, child):
            if isinstance(child.ctx, ast.Store): names.add(child.id)

    collector = Collector()
    for statement in node.body:
        collector.visit(statement)
    return names


class ContextTransformer(ast.NodeTransformer):
    def __init__(self, outer: set[str]):
        self.outer = outer
        self.local_scopes: list[set[str]] = []

    def visit_Name(self, node):
        if node.id == "self":
            return ast.copy_location(ast.Name(id="system", ctx=node.ctx), node)
        if node.id in self.outer and not any(node.id in scope for scope in self.local_scopes):
            return ast.copy_location(
                ast.Attribute(value=ast.Name(id="context", ctx=ast.Load()), attr=node.id, ctx=node.ctx), node,
            )
        return node

    def _visit_nested(self, node):
        self.local_scopes.append(nested_locals(node))
        node = self.generic_visit(node)
        self.local_scopes.pop()
        assignment = ast.Assign(
            targets=[ast.Attribute(value=ast.Name(id="context", ctx=ast.Load()), attr=node.name, ctx=ast.Store())],
            value=ast.Name(id=node.name, ctx=ast.Load()),
        )
        return [node, assignment]

    def visit_FunctionDef(self, node): return self._visit_nested(node)
    def visit_AsyncFunctionDef(self, node): return self._visit_nested(node)


def main() -> None:
    source = PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PromptUpdateMixin")
    function = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "update_prompt_with_beam")
    outer = collect_outer_locals(function)
    stage_classes = []
    for name, start, stop in STAGES:
        body = function.body[start:stop]
        transformed = []
        transformer = ContextTransformer(outer)
        for statement in body:
            result = transformer.visit(ast.fix_missing_locations(statement))
            transformed.extend(result if isinstance(result, list) else [result])
        stage_classes.append(ast.ClassDef(
            name=name, bases=[], keywords=[], decorator_list=[],
            body=[ast.AsyncFunctionDef(
                name="run",
                args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="system"), ast.arg(arg="context")], kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=transformed or [ast.Pass()], decorator_list=[ast.Name(id="staticmethod", ctx=ast.Load())],
            )],
        ))
    wrapper = ast.AsyncFunctionDef(
        name="update_prompt_with_beam",
        args=function.args,
        body=[
            ast.Assign(
                targets=[ast.Name(id="context", ctx=ast.Store())],
                value=ast.Call(func=ast.Name(id="PromptUpdateContext", ctx=ast.Load()), args=[], keywords=[
                    ast.keyword(arg=name, value=ast.Name(id=name, ctx=ast.Load()))
                    for name in ("agent_id", "overlap_diagnosis", "eval_batch", "step_id", "epoch_id")
                ]),
            ),
            *[
                ast.Expr(value=ast.Await(value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr="run", ctx=ast.Load()),
                    args=[ast.Name(id="self", ctx=ast.Load()), ast.Name(id="context", ctx=ast.Load())], keywords=[],
                ))) for name, _, _ in STAGES
            ],
            ast.Return(value=ast.Tuple(elts=[
                ast.Call(func=ast.Name(id="bool", ctx=ast.Load()), args=[ast.Attribute(value=ast.Name(id="context", ctx=ast.Load()), attr="changed", ctx=ast.Load())], keywords=[]),
                ast.Attribute(value=ast.Name(id="context", ctx=ast.Load()), attr="summary", ctx=ast.Load()),
            ], ctx=ast.Load())),
        ], decorator_list=[], returns=function.returns,
    )
    cls.body = [wrapper]
    context_class = ast.ClassDef(
        name="PromptUpdateContext", bases=[], keywords=[],
        decorator_list=[ast.Name(id="dataclass", ctx=ast.Load())],
        body=[
            ast.AnnAssign(target=ast.Name(id=name, ctx=ast.Store()), annotation=annotation, value=None, simple=1)
            for name, annotation in (
                ("agent_id", ast.Name(id="int", ctx=ast.Load())),
                ("overlap_diagnosis", ast.Subscript(value=ast.Name(id="Dict", ctx=ast.Load()), slice=ast.Tuple(elts=[ast.Name(id="str", ctx=ast.Load()), ast.Name(id="Any", ctx=ast.Load())], ctx=ast.Load()), ctx=ast.Load())),
                ("eval_batch", ast.Subscript(value=ast.Name(id="List", ctx=ast.Load()), slice=ast.Subscript(value=ast.Name(id="Dict", ctx=ast.Load()), slice=ast.Tuple(elts=[ast.Name(id="str", ctx=ast.Load()), ast.Name(id="str", ctx=ast.Load())], ctx=ast.Load()), ctx=ast.Load()), ctx=ast.Load())),
                ("step_id", ast.Name(id="int", ctx=ast.Load())),
                ("epoch_id", ast.Name(id="int", ctx=ast.Load())),
            )
        ],
    )
    module = ast.Module(body=[
        ast.Expr(value=ast.Constant(value="Typed prompt-update pipeline stages.")),
        ast.ImportFrom(module="dataclasses", names=[ast.alias(name="dataclass")], level=0),
        ast.ImportFrom(module="system_shared", names=[ast.alias(name="*")], level=2),
        context_class, *stage_classes, cls,
    ], type_ignores=[])
    PATH.write_text(ast.unparse(ast.fix_missing_locations(module)) + "\n", encoding="utf-8")
    print("staged prompt update orchestration")


if __name__ == "__main__":
    main()
