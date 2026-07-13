from tts_playground.common.validation import (
    ValidationIssue,
    parse_validation_issues,
    sanitize_field_identifier,
    validation_issues_to_public,
)


def test_field_identifier_keeps_only_bounded_ascii_names() -> None:
    assert sanitize_field_identifier("audio") == "audio"
    assert sanitize_field_identifier("a" * 64) == "a" * 64
    for unsafe in ("", "1audio", "a-b", "한글", "a" * 65, None, 7):
        assert sanitize_field_identifier(unsafe) == "unknown"


def test_untrusted_issue_parser_is_all_or_nothing_and_bounded() -> None:
    assert parse_validation_issues(
        [{"field": "audio", "reason": "required"}]
    ) == (ValidationIssue("audio", "required"),)
    assert parse_validation_issues(
        [{"field": "audio", "reason": "private-reason"}]
    ) is None
    assert parse_validation_issues(
        [{"field": "audio", "reason": "required"}] * 17
    ) is None


def test_untrusted_issue_parser_rejects_malformed_payloads() -> None:
    malformed: tuple[object, ...] = (
        [],
        {"field": "audio", "reason": "required"},
        [{"field": "audio", "reason": "required", "private": "value"}],
        [{"field": "audio"}],
        [{"reason": "required"}],
        [{"field": "audio", "reason": 7}],
        [{"field": "a-b", "reason": "required"}],
        [{"field": "한글", "reason": "required"}],
        [{"field": "a" * 65, "reason": "required"}],
        ["not-an-issue"],
    )

    for value in malformed:
        assert parse_validation_issues(value) is None


def test_validation_issues_serialize_only_public_keys() -> None:
    issues = (
        ValidationIssue("audio", "required"),
        ValidationIssue("language", "invalid"),
    )

    assert validation_issues_to_public(issues) == [
        {"field": "audio", "reason": "required"},
        {"field": "language", "reason": "invalid"},
    ]
