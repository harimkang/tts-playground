from dataclasses import dataclass


@dataclass(slots=True)
class ServiceError(Exception):
    status_code: int
    code: str
    message: str
    backend_failure: bool = False

    def __str__(self) -> str:
        return self.message
