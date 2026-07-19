"""One-time extractor for checkpoint persistence from the legacy CLI."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "multi_dataset_diverse_rl" / "cli.py"
OUT = ROOT / "multi_dataset_diverse_rl" / "persistence" / "checkpoint.py"


def main() -> None:
    lines = CLI.read_text(encoding="utf-8").splitlines(keepends=True)
    first = lines[378:530]
    second = lines[532:1121]
    OUT.write_text(
        '"""Canonical checkpoint serialization and compatibility validation."""\n\n'
        "import hashlib\nimport json\nimport os\nimport random\nimport time\nimport uuid\n\n"
        "import numpy as np\n\n"
        "from ..config import Config\n"
        "from ..utils import canonical_aggregation_mode\n\n\n"
        + "".join(first + second),
        encoding="utf-8",
    )
    replacement = [
        "from .persistence.checkpoint import (\n",
        "    BEHAVIOR_CONFIG_FIELDS, CHECKPOINT_VERSION, abort_incompatible_checkpoint,\n",
        "    build_training_checkpoint, checkpoint_behavior_config, checkpoint_behavior_config_fingerprint,\n",
        "    checkpoint_compatible, checkpoint_config_signature, checkpoint_incompatibility_reasons,\n",
        "    checkpoint_path, clear_training_checkpoint, read_json_file, restore_cost_summary,\n",
        "    restore_system_state, write_json_atomic, write_training_checkpoint,\n",
        ")\n\n",
    ]
    CLI.write_text("".join(lines[:378] + replacement + lines[530:532] + lines[1121:]), encoding="utf-8")
    print("extracted checkpoint persistence")


if __name__ == "__main__":
    main()
