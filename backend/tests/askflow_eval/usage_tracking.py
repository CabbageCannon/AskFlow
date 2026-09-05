"""Low-intrusion API usage instrumentation for AskFlow evaluation."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from threading import Lock
from typing import Any, Awaitable, Callable, Iterator
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler

try:
    from .pricing import PricingSnapshot
except ImportError:
    from pricing import PricingSnapshot


UNKNOWN_MODEL = "unknown"
UNKNOWN_STAGE = "unknown"

LANGGRAPH_STAGE_MAP = {
    "clarify_with_user": "clarification",
    "write_research_brief": "research_brief",
    "research_supervisor": "supervisor",
    "supervisor": "supervisor",
    "researcher": "researcher",
    "researcher_tools": "researcher",
    "compress_research": "compression",
    "evidence_verifier": "evidence_verification",
    "final_report_generation": "final_report",
}


@dataclass(frozen=True)
class NormalizedLLMUsage:
    """Provider-independent token usage for one completed LLM request."""

    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    usage_present: bool = True


@dataclass
class UsageAggregate:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    missing_usage_count: int = 0
    error_calls: int = 0

    def add(
        self,
        usage: NormalizedLLMUsage,
        *,
        error: bool = False,
    ) -> None:
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cached_input_tokens += usage.cached_input_tokens
        if not usage.usage_present:
            self.missing_usage_count += 1
        if error:
            self.error_calls += 1


@dataclass(frozen=True)
class _LLMRunContext:
    model: str
    stage: str


@dataclass(frozen=True)
class _RecordedLLMEvent:
    usage: NormalizedLLMUsage
    stage: str
    error: bool = False


_CURRENT_TRACKER: ContextVar["EvalUsageTracker | None"] = ContextVar(
    "askflow_eval_usage_tracker",
    default=None,
)

_CURRENT_STAGE_OVERRIDE: ContextVar[str | None] = ContextVar(
    "askflow_eval_usage_stage",
    default=None,
)


def set_current_usage_tracker(
    tracker: "EvalUsageTracker",
) -> Token:
    return _CURRENT_TRACKER.set(tracker)


def reset_current_usage_tracker(token: Token) -> None:
    _CURRENT_TRACKER.reset(token)


@contextmanager
def usage_stage(stage: str) -> Iterator[None]:
    """Temporarily override LLM stage attribution in the current async context."""

    token = _CURRENT_STAGE_OVERRIDE.set(stage)
    try:
        yield
    finally:
        _CURRENT_STAGE_OVERRIDE.reset(token)


def current_usage_tracker() -> "EvalUsageTracker | None":
    return _CURRENT_TRACKER.get()


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _extract_cached_tokens(usage: dict[str, Any]) -> int:
    details = usage.get("input_token_details")
    if isinstance(details, dict):
        for key in (
            "cache_read",
            "cached_tokens",
            "cache_hit_tokens",
        ):
            value = _coerce_int(details.get(key))
            if value is not None:
                return value

    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        value = _coerce_int(prompt_details.get("cached_tokens"))
        if value is not None:
            return value

    for key in (
        "cached_input_tokens",
        "cache_read_input_tokens",
        "cached_tokens",
    ):
        value = _coerce_int(usage.get(key))
        if value is not None:
            return value

    return 0


def _normalize_usage_dict(
    usage: Any,
    *,
    model: str,
) -> NormalizedLLMUsage | None:
    if not isinstance(usage, dict):
        return None

    input_tokens = _coerce_int(
        usage.get("input_tokens", usage.get("prompt_tokens"))
    )
    output_tokens = _coerce_int(
        usage.get("output_tokens", usage.get("completion_tokens"))
    )

    if input_tokens is None or output_tokens is None:
        return None

    return NormalizedLLMUsage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=min(
            input_tokens,
            _extract_cached_tokens(usage),
        ),
        usage_present=True,
    )


def _iter_generation_messages(response: Any):
    generations = getattr(response, "generations", None)
    if not isinstance(generations, (list, tuple)):
        return

    for generation_group in generations:
        if not isinstance(generation_group, (list, tuple)):
            generation_group = [generation_group]

        for generation in generation_group:
            message = getattr(generation, "message", None)
            if message is not None:
                yield message


def _model_from_message(message: Any) -> str | None:
    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        for key in ("model_name", "model", "model_id"):
            value = response_metadata.get(key)
            if value:
                return str(value)

    usage_metadata = getattr(message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        for key in ("model_name", "model", "model_id"):
            value = usage_metadata.get(key)
            if value:
                return str(value)

    return None


def normalize_llm_result(
    response: Any,
    *,
    fallback_model: str,
) -> NormalizedLLMUsage:
    """Normalize LangChain/provider usage metadata without assuming one schema."""

    resolved_model = fallback_model or UNKNOWN_MODEL

    for message in _iter_generation_messages(response):
        message_model = _model_from_message(message)
        if message_model:
            resolved_model = message_model

        usage_metadata = getattr(message, "usage_metadata", None)
        normalized = _normalize_usage_dict(
            usage_metadata,
            model=resolved_model,
        )
        if normalized is not None:
            return normalized

        response_metadata = getattr(message, "response_metadata", None)
        if isinstance(response_metadata, dict):
            for key in ("token_usage", "usage"):
                normalized = _normalize_usage_dict(
                    response_metadata.get(key),
                    model=resolved_model,
                )
                if normalized is not None:
                    return normalized

    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        for model_key in ("model_name", "model", "model_id"):
            if llm_output.get(model_key):
                resolved_model = str(llm_output[model_key])
                break

        for key in ("token_usage", "usage"):
            normalized = _normalize_usage_dict(
                llm_output.get(key),
                model=resolved_model,
            )
            if normalized is not None:
                return normalized

    return NormalizedLLMUsage(
        model=resolved_model,
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        usage_present=False,
    )


def _extract_start_model(
    serialized: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> str:
    metadata = metadata or {}

    for key in ("ls_model_name", "model_name", "model"):
        value = metadata.get(key)
        if value:
            return str(value)

    invocation_params = kwargs.get("invocation_params")
    if isinstance(invocation_params, dict):
        for key in ("model", "model_name"):
            value = invocation_params.get(key)
            if value:
                return str(value)

    if isinstance(serialized, dict):
        kwargs_payload = serialized.get("kwargs")
        if isinstance(kwargs_payload, dict):
            for key in ("model", "model_name"):
                value = kwargs_payload.get(key)
                if value:
                    return str(value)

    return UNKNOWN_MODEL


def _extract_stage(metadata: dict[str, Any] | None) -> str:
    override = _CURRENT_STAGE_OVERRIDE.get()
    if override:
        return override

    metadata = metadata or {}
    node = metadata.get("langgraph_node")
    if node:
        return LANGGRAPH_STAGE_MAP.get(str(node), str(node))

    return UNKNOWN_STAGE


class EvalUsageTracker:
    """Thread-safe per-eval-task API usage accumulator."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._totals = UsageAggregate()
        self._by_model: dict[str, UsageAggregate] = defaultdict(UsageAggregate)
        self._by_stage: dict[str, UsageAggregate] = defaultdict(UsageAggregate)
        self._llm_events: list[_RecordedLLMEvent] = []
        self._search_requests: dict[str, int] = defaultdict(int)
        self._search_error_requests: dict[str, int] = defaultdict(int)

    def record_llm(
        self,
        usage: NormalizedLLMUsage,
        *,
        stage: str,
        error: bool = False,
    ) -> None:
        model = usage.model or UNKNOWN_MODEL
        stage = stage or UNKNOWN_STAGE
        normalized = NormalizedLLMUsage(
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            usage_present=usage.usage_present,
        )

        with self._lock:
            self._totals.add(normalized, error=error)
            self._by_model[model].add(normalized, error=error)
            self._by_stage[stage].add(normalized, error=error)
            self._llm_events.append(
                _RecordedLLMEvent(
                    usage=normalized,
                    stage=stage,
                    error=error,
                )
            )

    def record_external_search(
        self,
        *,
        provider: str,
        error: bool = False,
    ) -> None:
        with self._lock:
            self._search_requests[provider] += 1
            if error:
                self._search_error_requests[provider] += 1

    @staticmethod
    def _aggregate_to_dict(
        aggregate: UsageAggregate,
    ) -> dict[str, Any]:
        return {
            "calls": aggregate.calls,
            "input_tokens": aggregate.input_tokens,
            "output_tokens": aggregate.output_tokens,
            "cached_input_tokens": aggregate.cached_input_tokens,
            "missing_usage_count": aggregate.missing_usage_count,
            "error_calls": aggregate.error_calls,
        }

    @staticmethod
    def _price_event(
        event: _RecordedLLMEvent,
        pricing: PricingSnapshot,
    ) -> float | None:
        if not event.usage.usage_present:
            return None

        return pricing.estimate_model_cost(
            model_name=event.usage.model,
            input_tokens=event.usage.input_tokens,
            output_tokens=event.usage.output_tokens,
            cached_input_tokens=event.usage.cached_input_tokens,
        )

    def snapshot(
        self,
        pricing: PricingSnapshot,
    ) -> dict[str, Any]:
        with self._lock:
            totals = UsageAggregate(**vars(self._totals))
            by_model = {
                model: UsageAggregate(**vars(aggregate))
                for model, aggregate in self._by_model.items()
            }
            by_stage = {
                stage: UsageAggregate(**vars(aggregate))
                for stage, aggregate in self._by_stage.items()
            }
            llm_events = list(self._llm_events)
            search_requests = dict(self._search_requests)
            search_error_requests = dict(self._search_error_requests)

        model_costs: dict[str, float] = defaultdict(float)
        stage_costs: dict[str, float] = defaultdict(float)
        model_cost_complete: dict[str, bool] = {
            model: aggregate.missing_usage_count == 0
            for model, aggregate in by_model.items()
        }
        stage_cost_complete: dict[str, bool] = {
            stage: aggregate.missing_usage_count == 0
            for stage, aggregate in by_stage.items()
        }

        unpriced_models: set[str] = set()
        known_llm_cost = 0.0

        for event in llm_events:
            model = event.usage.model or UNKNOWN_MODEL
            stage = event.stage or UNKNOWN_STAGE
            cost = self._price_event(event, pricing)

            if cost is None:
                model_cost_complete[model] = False
                stage_cost_complete[stage] = False
                if pricing.resolve_model(model) is None:
                    unpriced_models.add(model)
                continue

            model_costs[model] += cost
            stage_costs[stage] += cost
            known_llm_cost += cost

        usage_by_model: dict[str, Any] = {}
        for model, aggregate in sorted(by_model.items()):
            item = self._aggregate_to_dict(aggregate)
            item["cost"] = (
                round(model_costs[model], 8)
                if model_cost_complete.get(model, False)
                else None
            )
            usage_by_model[model] = item

        usage_by_stage: dict[str, Any] = {}
        for stage, aggregate in sorted(by_stage.items()):
            item = self._aggregate_to_dict(aggregate)
            item["cost"] = (
                round(stage_costs[stage], 8)
                if stage_cost_complete.get(stage, False)
                else None
            )
            usage_by_stage[stage] = item

        external_search_requests = sum(search_requests.values())
        known_search_cost = 0.0
        unpriced_search_providers: list[str] = []

        for provider, request_count in search_requests.items():
            provider_cost = pricing.estimate_search_cost(
                provider=provider,
                requests=request_count,
            )
            if provider_cost is None:
                unpriced_search_providers.append(provider)
            else:
                known_search_cost += provider_cost

        llm_cost_complete = (
            totals.missing_usage_count == 0
            and all(model_cost_complete.values())
            and not unpriced_models
        )
        search_cost_complete = not unpriced_search_providers
        cost_complete = llm_cost_complete and search_cost_complete

        known_api_cost = known_llm_cost + known_search_cost

        return {
            "llm_calls": totals.calls,
            "input_tokens": totals.input_tokens,
            "output_tokens": totals.output_tokens,
            "cached_input_tokens": totals.cached_input_tokens,
            "missing_usage_count": totals.missing_usage_count,
            "llm_error_calls": totals.error_calls,
            "llm_cost": (
                round(known_llm_cost, 8)
                if llm_cost_complete
                else None
            ),
            "known_llm_cost": round(known_llm_cost, 8),
            "external_search_requests": external_search_requests,
            "external_search_requests_by_provider": search_requests,
            "search_error_requests_by_provider": search_error_requests,
            "search_cost": (
                round(known_search_cost, 8)
                if search_cost_complete
                else None
            ),
            "known_search_cost": round(known_search_cost, 8),
            "total_api_cost": (
                round(known_api_cost, 8)
                if cost_complete
                else None
            ),
            "known_api_cost": round(known_api_cost, 8),
            "cost_complete": cost_complete,
            "pricing_currency": pricing.currency,
            "pricing_snapshot": pricing.effective_date,
            "pricing_basis": pricing.pricing_basis,
            "unpriced_models": sorted(unpriced_models),
            "unpriced_search_providers": sorted(unpriced_search_providers),
            "usage_by_model": usage_by_model,
            "usage_by_stage": usage_by_stage,
        }


class EvalUsageCallbackHandler(BaseCallbackHandler):
    """Observe each real chat-model attempt without changing model behavior."""

    raise_error = False
    run_inline = True

    def __init__(self, tracker: EvalUsageTracker) -> None:
        super().__init__()
        self._tracker = tracker
        self._lock = Lock()
        self._runs: dict[UUID, _LLMRunContext] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            context = _LLMRunContext(
                model=_extract_start_model(
                    serialized,
                    metadata,
                    kwargs,
                ),
                stage=_extract_stage(metadata),
            )
            with self._lock:
                self._runs[run_id] = context
        except Exception:
            return

    def _pop_context(self, run_id: UUID) -> _LLMRunContext:
        with self._lock:
            return self._runs.pop(
                run_id,
                _LLMRunContext(
                    model=UNKNOWN_MODEL,
                    stage=UNKNOWN_STAGE,
                ),
            )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            context = self._pop_context(run_id)
            usage = normalize_llm_result(
                response,
                fallback_model=context.model,
            )
            self._tracker.record_llm(
                usage,
                stage=context.stage,
            )
        except Exception:
            return

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            context = self._pop_context(run_id)
            self._tracker.record_llm(
                NormalizedLLMUsage(
                    model=context.model,
                    input_tokens=0,
                    output_tokens=0,
                    usage_present=False,
                ),
                stage=context.stage,
                error=True,
            )
        except Exception:
            return


def wrap_async_external_search(
    original: Callable[..., Awaitable[Any]],
    *,
    provider: str,
) -> Callable[..., Awaitable[Any]]:
    """Count each actual SDK search call, including infrastructure retries."""

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        tracker = current_usage_tracker()
        try:
            result = await original(*args, **kwargs)
        except BaseException:
            if tracker is not None:
                tracker.record_external_search(
                    provider=provider,
                    error=True,
                )
            raise
        else:
            if tracker is not None:
                tracker.record_external_search(
                    provider=provider,
                    error=False,
                )
            return result

    return wrapped


async def run_in_stage(
    stage: str,
    awaitable_factory: Callable[[], Awaitable[Any]],
) -> Any:
    """Run one async operation with a stage override inherited by child tasks."""

    with usage_stage(stage):
        return await awaitable_factory()
