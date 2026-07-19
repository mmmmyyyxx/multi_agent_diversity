from enum import Enum


class CandidateType(str, Enum):
    TASK_REPAIR = "task_specific_repair"
    MECHANISM_ALTERNATIVE = "mechanism_alternative"


class ArchiveBucket(str, Enum):
    SAFE = "safe"
    PROBATION = "probation"
    CATASTROPHIC = "catastrophic"


class ParentSource(str, Enum):
    ACTIVE = "active"
    SAFE_NICHE = "safe_niche"
    PROBATION_NICHE = "probation_niche"
