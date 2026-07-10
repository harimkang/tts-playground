from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Histogram


@dataclass(slots=True)
class GatewayMetrics:
    registry: CollectorRegistry
    requests_total: Counter
    request_seconds: Histogram
    audio_seconds: Histogram
    backend_errors_total: Counter

    @classmethod
    def create(cls) -> "GatewayMetrics":
        registry = CollectorRegistry()
        return cls(
            registry=registry,
            requests_total=Counter(
                "tts_gateway_requests_total",
                "Gateway synthesis requests.",
                ("model", "status"),
                registry=registry,
            ),
            request_seconds=Histogram(
                "tts_gateway_request_seconds",
                "Gateway synthesis request duration in seconds.",
                ("model",),
                registry=registry,
            ),
            audio_seconds=Histogram(
                "tts_gateway_audio_seconds",
                "Generated audio duration in seconds.",
                ("model",),
                registry=registry,
            ),
            backend_errors_total=Counter(
                "tts_gateway_backend_errors_total",
                "Backend errors observed by the gateway.",
                ("backend", "code"),
                registry=registry,
            ),
        )

    def record_request(
        self,
        *,
        model: str,
        status: int,
        elapsed_seconds: float,
        audio_seconds: float | None,
    ) -> None:
        self.requests_total.labels(model=model, status=str(status)).inc()
        self.request_seconds.labels(model=model).observe(elapsed_seconds)
        if audio_seconds is not None:
            self.audio_seconds.labels(model=model).observe(audio_seconds)

    def record_backend_error(self, *, backend: str, code: str) -> None:
        self.backend_errors_total.labels(backend=backend, code=code).inc()
