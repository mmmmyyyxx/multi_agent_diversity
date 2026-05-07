from .config import Config, build_parser

__all__ = ["Config", "build_parser", "TextualGradientRLSystem"]


def __getattr__(name: str):
    if name == "TextualGradientRLSystem":
        from .system import TextualGradientRLSystem

        return TextualGradientRLSystem
    raise AttributeError(name)
