"""Utilities for running Markov Chain Monte Carlo over SQL query candidates."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol


class ProposalError(RuntimeError):
    """Raised when a proposal distribution fails to generate a candidate."""


class Proposal(Protocol):
    """Protocol describing proposal distributions used by :class:`MCMCSampler`."""

    def propose(self, state: "MCMCState", rng: random.Random) -> "MCMCState":
        """Return a new state sampled from the proposal distribution."""

    def log_transition_probability(
        self, from_state: "MCMCState", to_state: "MCMCState"
    ) -> float:
        """Return the log transition probability for asymmetric proposals.

        The default implementation should return ``0.0`` to indicate that the
        proposal distribution is symmetric and therefore the ratio cancels out
        in the Metropolis-Hastings acceptance rule.
        """


@dataclass
class MCMCState:
    """Container describing a single state in the Markov chain."""

    query: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy_with(self, *, query: Optional[str] = None, score: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> "MCMCState":
        """Return a shallow copy of the state with optional overrides."""

        return MCMCState(
            query=query if query is not None else self.query,
            score=score if score is not None else self.score,
            metadata=dict(self.metadata if metadata is None else metadata),
        )


@dataclass
class MCMCConfig:
    """Configuration for controlling the behaviour of :class:`MCMCSampler`."""

    iterations: int
    burn_in: int = 0
    thinning: int = 1
    temperature: float = 1.0
    max_failed_proposals: Optional[int] = 100
    random_seed: Optional[int] = None
    store_trace: bool = True

    def validate(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if self.thinning <= 0:
            raise ValueError("thinning must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.max_failed_proposals is not None and self.max_failed_proposals < 0:
            raise ValueError("max_failed_proposals must be non-negative")


@dataclass
class MCMCResult:
    """Summary statistics for a completed Markov chain."""

    accepted_states: List[MCMCState]
    rejected_states: int
    acceptance_rate: float
    best_state: MCMCState

    def trace(self) -> Iterable[MCMCState]:
        """Return an iterator over the accepted states."""

        return iter(self.accepted_states)


class MCMCSampler:
    """Implementation of a generic Metropolis-Hastings sampler for SQL queries."""

    def __init__(
        self,
        proposal: Proposal,
        evaluator: "QueryEvaluatorProtocol",
        config: MCMCConfig,
    ) -> None:
        config.validate()
        self._proposal = proposal
        self._evaluator = evaluator
        self._config = config
        self._rng = random.Random(config.random_seed)

    def _acceptance_probability(
        self,
        current_state: MCMCState,
        candidate_state: MCMCState,
    ) -> float:
        """Compute the Metropolis-Hastings acceptance probability."""

        if math.isinf(candidate_state.score) and candidate_state.score < 0:
            return 0.0

        log_prob_ratio = (candidate_state.score - current_state.score) / self._config.temperature

        try:
            reverse_log_prob = self._proposal.log_transition_probability(candidate_state, current_state)
            forward_log_prob = self._proposal.log_transition_probability(current_state, candidate_state)
        except AttributeError:
            reverse_log_prob = forward_log_prob = 0.0

        log_prob_ratio += reverse_log_prob - forward_log_prob
        if log_prob_ratio >= 0:
            return 1.0
        if math.isinf(log_prob_ratio):
            return 0.0
        return math.exp(log_prob_ratio)

    def _draw_candidate(self, current_state: MCMCState) -> MCMCState:
        try:
            candidate = self._proposal.propose(current_state, self._rng)
        except ProposalError:
            raise
        except Exception as exc:  # pragma: no cover - defensive path
            raise ProposalError(f"Proposal distribution failed: {exc!r}") from exc
        if not isinstance(candidate, MCMCState):
            raise ProposalError("Proposal must return an MCMCState instance")
        return candidate

    def run(self, initial_query: str, *, metadata: Optional[Dict[str, Any]] = None) -> MCMCResult:
        """Execute the sampler starting from ``initial_query``.

        Parameters
        ----------
        initial_query:
            The SQL query that seeds the Markov chain. The score for the query
            is computed using the supplied evaluator.
        metadata:
            Optional metadata dictionary that will be attached to the initial
            state and copied to future states. Mutations may update this
            dictionary.
        """

        base_metadata = dict(metadata or {})
        initial_score = self._evaluator.evaluate(initial_query, base_metadata)
        current_state = MCMCState(initial_query, initial_score, base_metadata)

        accepted_states: List[MCMCState] = []
        rejected_states = 0
        best_state = current_state
        failed_proposals = 0
        accepted_moves = 0

        for step in range(self._config.iterations):
            try:
                candidate_state = self._draw_candidate(current_state)
            except ProposalError:
                failed_proposals += 1
                if (
                    self._config.max_failed_proposals is not None
                    and failed_proposals > self._config.max_failed_proposals
                ):
                    raise ProposalError(
                        "Maximum number of failed proposals exceeded; check proposal configuration"
                    )
                continue

            candidate_state = candidate_state.copy_with(
                score=self._evaluator.evaluate(candidate_state.query, candidate_state.metadata)
            )

            acceptance_probability = self._acceptance_probability(current_state, candidate_state)
            if self._rng.random() < acceptance_probability:
                current_state = candidate_state
                accepted_moves += 1
                if current_state.score > best_state.score:
                    best_state = current_state
                if self._config.store_trace and step >= self._config.burn_in and (
                    (step - self._config.burn_in) % self._config.thinning == 0
                ):
                    accepted_states.append(current_state)
            else:
                rejected_states += 1

        total_moves = accepted_moves + rejected_states
        acceptance_rate = accepted_moves / max(1, total_moves)

        return MCMCResult(
            accepted_states=accepted_states,
            rejected_states=rejected_states,
            acceptance_rate=acceptance_rate,
            best_state=best_state,
        )


class QueryEvaluatorProtocol(Protocol):
    """Protocol capturing the behaviour expected from a query evaluator."""

    def evaluate(self, query: str, metadata: Dict[str, Any]) -> float:
        ...
