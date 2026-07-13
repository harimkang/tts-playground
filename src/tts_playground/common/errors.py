from dataclasses import dataclass

from tts_playground.common.validation import ValidationIssue


@dataclass(slots=True)
class ServiceError(Exception):
    status_code: int
    code: str
    message: str
    backend_failure: bool = False
    details: tuple[ValidationIssue, ...] | None = None

    def __str__(self) -> str:
        return self.message
