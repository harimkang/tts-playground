from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from tts_playground.common.models import ModelID
from tts_playground.qwen.manager import WorkerState, WorkerStatus

_QWEN_MODELS = (
    ModelID.QWEN_CUSTOM,
    ModelID.QWEN_DESIGN,
    ModelID.QWEN_CLONE,
)


@dataclass(slots=True)
class QwenMetrics:
    registry: CollectorRegistry
    requests_total: Counter
    errors_total: Counter
    request_seconds: Histogram
    audio_seconds: Histogram
    queue_depth: Gauge
    state: Gauge
    loaded_model: Gauge
    free_vram_bytes: Gauge

    @classmethod
    def create(cls) -> "QwenMetrics":
        registry = CollectorRegistry()
        metrics = cls(
            registry=registry,
            requests_total=Counter(
                "tts_qwen_requests_total",
                "Qwen synthesis requests.",
                ("model", "status"),
                registry=registry,
            ),
            errors_total=Counter(
                "tts_qwen_errors_total",
                "Qwen worker errors.",
                ("code",),
                registry=registry,
            ),
            request_seconds=Histogram(
                "tts_qwen_request_seconds",
                "Qwen synthesis request duration in seconds.",
                ("model",),
                registry=registry,
            ),
            audio_seconds=Histogram(
                "tts_qwen_audio_seconds",
                "Generated Qwen audio duration in seconds.",
                ("model",),
                registry=registry,
            ),
            queue_depth=Gauge(
                "tts_qwen_queue_depth",
                "Current number of queued Qwen requests.",
                registry=registry,
            ),
            state=Gauge(
                "tts_qwen_state",
                "Current Qwen worker state.",
                ("state",),
                registry=registry,
            ),
            loaded_model=Gauge(
                "tts_qwen_loaded_model",
                "Whether a Qwen model is currently loaded.",
                ("model",),
                registry=registry,
            ),
            free_vram_bytes=Gauge(
                "tts_qwen_free_vram_bytes",
                "Last observed free CUDA memory in bytes.",
                registry=registry,
            ),
        )
        metrics.observe_status(
            WorkerStatus(
                state=WorkerState.UNLOADED,
                loaded_model=None,
                queue_depth=0,
                last_error_code=None,
            )
        )
        return metrics

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

    def record_error(self, code: str) -> None:
        self.errors_total.labels(code=code).inc()

    def observe_status(self, status: WorkerStatus) -> None:
        self.queue_depth.set(status.queue_depth)
        for state in WorkerState:
            self.state.labels(state=state.value).set(float(state == status.state))
        for model in _QWEN_MODELS:
            self.loaded_model.labels(model=model.value).set(
                float(model == status.loaded_model)
            )

    def observe_free_vram(self, free_bytes: int) -> None:
        self.free_vram_bytes.set(free_bytes)
