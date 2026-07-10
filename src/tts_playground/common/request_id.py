import re
from uuid import uuid4

REQUEST_ID_PATTERN = re.compile(r"req_[A-Za-z0-9_-]{8,80}")


def resolve_request_id(candidate: str | None) -> str:
    if candidate is not None and REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate
    return f"req_{uuid4().hex}"
