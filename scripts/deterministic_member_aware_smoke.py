from __future__ import annotations

from deterministic_member_aware_system_smoke import main as system_main
from deterministic_member_objective_unit_smoke import main as unit_main


def main() -> None:
    unit_main()
    system_main()


if __name__ == "__main__":
    main()
