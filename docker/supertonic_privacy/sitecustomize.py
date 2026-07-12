import logging
from typing import Any

_SAFE_ERROR_MESSAGE = "Supertonic request failed."


class _PrivacyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            record.msg = _SAFE_ERROR_MESSAGE
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return True


_original_record_factory = logging.getLogRecordFactory()
_privacy_filter = _PrivacyFilter()


def _privacy_record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    record = _original_record_factory(*args, **kwargs)
    _privacy_filter.filter(record)
    return record


logging.setLogRecordFactory(_privacy_record_factory)
