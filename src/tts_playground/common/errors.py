from dataclasses import dataclass


@dataclass(slots=True)
class ServiceError(Exception):
    status_code: int
    code: str
    message: str

    def __str__(self) -> str:
        return self.message
