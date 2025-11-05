"""Evaluation utilities for scoring SQL queries during MCMC."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple


@dataclass
class ExecutionResult:
    """Container describing the outcome of executing a SQL query."""

    query: str
    rows: Sequence[Tuple[Any, ...]]
    score: float
    metadata: Dict[str, Any]

class QueryEvaluator:
    """Execute SQL queries against a database connection and score the result."""

    def __init__(
        self,
        connection,
        scorer: Optional[Callable[[Sequence[Tuple[Any, ...]], Dict[str, Any]], float]] = None,
        *,
        fetch_all: bool = True,
        fetch_size: int = 1000,
        error_score: float = float("-inf"),
        autocommit: bool = False,
    ) -> None:
        self._connection = connection
        self._scorer = scorer or (lambda rows, metadata: float(len(rows)))
        self._fetch_all = fetch_all
        self._fetch_size = fetch_size
        self._error_score = error_score
        self._autocommit = autocommit

    def execute(self, query: str, metadata: Dict[str, Any]) -> ExecutionResult:
        cursor = self._connection.cursor()
        try:
            cursor.execute(query)
            if self._autocommit:
                try:
                    self._connection.commit()
                except AttributeError:
                    pass
            if self._fetch_all:
                rows = cursor.fetchall()
            else:
                rows = cursor.fetchmany(self._fetch_size)
            score = float(self._scorer(rows, metadata))
            return ExecutionResult(query=query, rows=tuple(rows), score=score, metadata=dict(metadata))
        except Exception:
            return ExecutionResult(query=query, rows=(), score=self._error_score, metadata=dict(metadata))
        finally:
            try:
                cursor.close()
            except Exception:
                pass

    def evaluate(self, query: str, metadata: Dict[str, Any]) -> float:
        result = self.execute(query, metadata)
        return result.score

    def close(self) -> None:
        """Close the underlying database connection if it exposes ``close``."""

        close = getattr(self._connection, "close", None)
        if callable(close):
            close()
