import re
from dataclasses import dataclass
from typing import Final, Literal, cast

ValidationReason = Literal["required", "invalid", "duplicate", "unknown"]
MAX_VALIDATION_ISSUES: Final = 16
_FIELD_PATTERN: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_REASONS: Final = frozenset({"required", "invalid", "duplicate", "unknown"})


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    field: str
    reason: ValidationReason


def sanitize_field_identifier(value: object) -> str:
    return (
        value
        if isinstance(value, str) and _FIELD_PATTERN.fullmatch(value)
        else "unknown"
    )


def parse_validation_issues(value: object) -> tuple[ValidationIssue, ...] | None:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_VALIDATION_ISSUES:
        return None

    issues: list[ValidationIssue] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"field", "reason"}:
            return None
        field = item["field"]
        reason = item["reason"]
        if (
            not isinstance(field, str)
            or _FIELD_PATTERN.fullmatch(field) is None
            or not isinstance(reason, str)
            or reason not in _REASONS
        ):
            return None
        issues.append(ValidationIssue(field, cast(ValidationReason, reason)))
    return tuple(issues)


def validation_issues_to_public(
    issues: tuple[ValidationIssue, ...],
) -> list[dict[str, str]]:
    return [{"field": issue.field, "reason": issue.reason} for issue in issues]
