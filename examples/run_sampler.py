"""Example script showcasing AlphaSQL MCMC and database connectivity."""
from __future__ import annotations

import argparse
import random
from typing import Dict, List, Sequence, Tuple

from alphasql import (
    DatabaseConnector,
    DatabaseType,
    MCMCConfig,
    MCMCSampler,
    QueryEvaluator,
    SQLMutationProposal,
)


# ---------------------------------------------------------------------------
# Demo utilities
# ---------------------------------------------------------------------------
class DummyCursor:
    def __init__(self, rng: random.Random, table: str) -> None:
        self._rng = rng
        self._table = table
        self._rows: List[Tuple[float]] = []

    def execute(self, query: str) -> None:  # pragma: no cover - used interactively
        base = abs(hash((query, self._table))) % 100
        self._rows = [(self._rng.random() + base / 100.0,) for _ in range(self._rng.randint(5, 20))]

    def fetchall(self) -> List[Tuple[float]]:
        return list(self._rows)

    def fetchmany(self, size: int) -> List[Tuple[float]]:
        return list(self._rows[:size])

    def close(self) -> None:
        self._rows.clear()


class DummyConnection:
    """In-memory connection used when no database backend is available."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def cursor(self) -> DummyCursor:
        return DummyCursor(self._rng, "dummy_table")

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Application logic
# ---------------------------------------------------------------------------

def build_proposal() -> SQLMutationProposal:
    """Create a proposal distribution for demo purposes."""

    tables = {
        "orders": ("order_id", "customer_id", "total", "created_at"),
        "customers": ("customer_id", "first_name", "last_name", "country"),
    }
    filter_bank = {
        "orders": ("total > 100", "total < 1000", "status = 'SHIPPED'"),
        "customers": ("country = 'US'", "country = 'GB'"),
    }
    orderings = {
        "orders": ("created_at DESC", "total DESC"),
        "customers": ("last_name ASC",),
    }
    candidate_limits = (10, 50, 100)
    return SQLMutationProposal(
        tables,
        filter_bank=filter_bank,
        orderings=orderings,
        candidate_limits=candidate_limits,
    )


def default_score(rows: Sequence[Tuple[float]], metadata: Dict[str, object]) -> float:
    """Score queries by the inverse of the variance of the first column."""

    if not rows:
        return float("-inf")
    values = [row[0] for row in rows]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance == 0:
        return float("inf")
    return 1.0 / variance


def build_evaluator(args: argparse.Namespace) -> QueryEvaluator:
    if args.dry_run:
        connection = DummyConnection(seed=args.seed)
    else:
        backend = DatabaseType(args.backend)
        try:
            connector = DatabaseConnector.from_environment(backend)
            connection = connector.connect()
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise SystemExit(
                f"Failed to import database driver for backend '{backend.value}': {exc}"
            ) from exc
    return QueryEvaluator(connection, scorer=default_score, fetch_all=True, autocommit=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=[db.value for db in DatabaseType], default=DatabaseType.POSTGRES.value)
    parser.add_argument("--iterations", type=int, default=200, help="Number of MCMC iterations to run")
    parser.add_argument("--burn-in", type=int, default=20, help="Number of initial samples to discard")
    parser.add_argument("--seed", type=int, default=13, help="Random seed used by the sampler")
    parser.add_argument("--dry-run", action="store_true", help="Use an in-memory evaluator instead of a database")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    proposal = build_proposal()
    evaluator = build_evaluator(args)

    try:
        config = MCMCConfig(
            iterations=args.iterations,
            burn_in=args.burn_in,
            thinning=1,
            random_seed=args.seed,
        )

        sampler = MCMCSampler(proposal, evaluator, config)

        initial_query = "SELECT order_id FROM orders"
        result = sampler.run(initial_query, metadata={})

        print("Best query:")
        print(result.best_state.query)
        print(f"Score: {result.best_state.score:.4f}")
        print(f"Accepted samples: {len(result.accepted_states)}")
        print(f"Acceptance rate: {result.acceptance_rate:.2%}")
    finally:
        evaluator.close()


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
