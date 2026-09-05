"""Pricing snapshot helpers for AskFlow evaluation cost accounting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelPriceTier:
    """One single-request pricing tier in the snapshot currency."""

    max_input_tokens: int | None
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None


@dataclass(frozen=True)
class ModelPrice:
    """Per-request list pricing for one model in the snapshot currency."""

    canonical_name: str
    tiers: tuple[ModelPriceTier, ...]


@dataclass(frozen=True)
class PricingSnapshot:
    """Immutable reference pricing used to estimate benchmark serving cost."""

    effective_date: str
    currency: str
    pricing_basis: str
    models: dict[str, dict[str, Any]]
    search: dict[str, dict[str, Any]]
    _aliases: dict[str, str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PricingSnapshot":
        models = dict(data.get("models", {}))
        aliases: dict[str, str] = {}

        for canonical_name, raw in models.items():
            names = [canonical_name, *raw.get("aliases", [])]
            for name in names:
                aliases[str(name).strip().lower()] = canonical_name

        return cls(
            effective_date=str(data["effective_date"]),
            currency=str(data["currency"]),
            pricing_basis=str(data.get("pricing_basis", "reference")),
            models=models,
            search=dict(data.get("search", {})),
            _aliases=aliases,
        )

    @staticmethod
    def _parse_tier(raw: dict[str, Any]) -> ModelPriceTier:
        max_input_tokens = raw.get("max_input_tokens")
        return ModelPriceTier(
            max_input_tokens=(
                int(max_input_tokens)
                if max_input_tokens is not None
                else None
            ),
            input_per_million=float(raw["input_per_million"]),
            output_per_million=float(raw["output_per_million"]),
            cached_input_per_million=(
                float(raw["cached_input_per_million"])
                if raw.get("cached_input_per_million") is not None
                else None
            ),
        )

    def resolve_model(self, model_name: str | None) -> ModelPrice | None:
        if not model_name:
            return None

        normalized = str(model_name).strip().lower()
        canonical = self._aliases.get(normalized)

        if canonical is None and ":" in normalized:
            canonical = self._aliases.get(normalized.split(":", 1)[1])

        if canonical is None:
            return None

        raw = self.models[canonical]

        raw_tiers = raw.get("tiers")
        if isinstance(raw_tiers, list) and raw_tiers:
            tiers = tuple(
                self._parse_tier(tier)
                for tier in raw_tiers
            )
        else:
            # Backward-compatible flat-rate schema: one unbounded tier.
            tiers = (
                ModelPriceTier(
                    max_input_tokens=None,
                    input_per_million=float(raw["input_per_million"]),
                    output_per_million=float(raw["output_per_million"]),
                    cached_input_per_million=(
                        float(raw["cached_input_per_million"])
                        if raw.get("cached_input_per_million") is not None
                        else None
                    ),
                ),
            )

        return ModelPrice(
            canonical_name=canonical,
            tiers=tiers,
        )

    @staticmethod
    def _select_tier(
        price: ModelPrice,
        *,
        input_tokens: int,
    ) -> ModelPriceTier | None:
        input_tokens = max(0, int(input_tokens))

        for tier in price.tiers:
            if (
                tier.max_input_tokens is None
                or input_tokens <= tier.max_input_tokens
            ):
                return tier

        return None

    def estimate_model_cost(
        self,
        *,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float | None:
        """Estimate one real provider request.

        Tier selection is based on *this request's* input token count. Do not
        call this with per-model benchmark totals for tiered models.
        """

        price = self.resolve_model(model_name)
        if price is None:
            return None

        input_tokens = max(0, int(input_tokens))
        output_tokens = max(0, int(output_tokens))
        cached_input_tokens = min(
            input_tokens,
            max(0, int(cached_input_tokens)),
        )

        tier = self._select_tier(
            price,
            input_tokens=input_tokens,
        )
        if tier is None:
            return None

        uncached_input_tokens = input_tokens - cached_input_tokens

        input_cost = (
            uncached_input_tokens
            / 1_000_000
            * tier.input_per_million
        )

        if cached_input_tokens:
            cached_rate = (
                tier.cached_input_per_million
                if tier.cached_input_per_million is not None
                else tier.input_per_million
            )
            input_cost += (
                cached_input_tokens
                / 1_000_000
                * cached_rate
            )

        output_cost = (
            output_tokens
            / 1_000_000
            * tier.output_per_million
        )

        return input_cost + output_cost

    def estimate_search_cost(
        self,
        *,
        provider: str,
        requests: int,
    ) -> float | None:
        raw = self.search.get(provider)
        if raw is None:
            return None

        cost_per_request = raw.get("cost_per_request")
        if cost_per_request is None:
            return None

        return max(0, int(requests)) * float(cost_per_request)


def load_pricing_snapshot(path: Path | None = None) -> PricingSnapshot:
    """Load the checked-in pricing snapshot used by the eval harness."""

    snapshot_path = path or Path(__file__).with_name("pricing.json")
    with snapshot_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return PricingSnapshot.from_dict(data)
