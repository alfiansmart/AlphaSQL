"""AlphaSQL: Azure OpenAI implementation with planning, MCMC search and DB connectors.

This module exposes the :class:`AlphaSQLEngine` capable of translating natural
language questions into SQL using Azure's ``o3-mini`` reasoning model. In
addition to the base planning mode, an optional Metropolis-Hastings Monte Carlo
chain can explore alternative SQL drafts. Utilities for connecting to Postgres,
Snowflake and StarRocks databases are also provided.

Example
-------
>>> from alphasql import AlphaSQLEngine
>>> engine = AlphaSQLEngine()
>>> sql = engine.generate_sql(
...     "How many users registered in 2023?",
...     "table users(id, name, registered_at)",
...     planning=True,
...     mcmc_steps=10,
... )
>>> print(sql)
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # ``openai`` is required at runtime but optional for static analysis
    from openai import AzureOpenAI
except Exception:  # pragma: no cover - handled at runtime
    AzureOpenAI = None  # type: ignore


class DatabaseConnector:
    """Lightweight protocol for database connectors."""

    def execute(self, sql: str) -> Iterable[Tuple[Any, ...]]:  # pragma: no cover - runtime API
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - runtime API
        raise NotImplementedError

    def __enter__(self) -> "DatabaseConnector":  # pragma: no cover - convenience
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - convenience
        self.close()


class PostgresConnector(DatabaseConnector):
    """PostgreSQL connector using :mod:`psycopg` or :mod:`psycopg2`.

    Connection information is read from ``POSTGRES_DSN`` when not provided
    explicitly.
    """

    def __init__(self, dsn: Optional[str] = None, **connect_kwargs: Any) -> None:
        dsn = dsn or os.getenv("POSTGRES_DSN")
        if not dsn and not connect_kwargs:
            raise ValueError(
                "PostgresConnector requires a DSN or keyword arguments such as host, dbname, user."
            )

        try:
            try:
                import psycopg

                self._conn = psycopg.connect(dsn=dsn, **connect_kwargs)
            except ImportError:  # pragma: no cover - fallback import
                import psycopg2

                self._conn = psycopg2.connect(dsn=dsn, **connect_kwargs)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - runtime error propagation
            raise RuntimeError("Failed to connect to PostgreSQL") from exc

        self._conn.autocommit = True

    def execute(self, sql: str) -> Iterable[Tuple[Any, ...]]:
        with self._conn.cursor() as cur:
            cur.execute(sql)
            try:
                rows = cur.fetchall()
            except Exception:  # pragma: no cover - statements without result sets
                rows = []
        return rows

    def close(self) -> None:
        self._conn.close()


class SnowflakeConnector(DatabaseConnector):
    """Snowflake connector using :mod:`snowflake.connector`.

    Credentials are taken from the corresponding ``SNOWFLAKE_*`` environment
    variables unless provided explicitly.
    """

    def __init__(
        self,
        *,
        user: Optional[str] = None,
        password: Optional[str] = None,
        account: Optional[str] = None,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        try:
            import snowflake.connector
        except Exception as exc:  # pragma: no cover - runtime import guard
            raise ImportError(
                "snowflake-connector-python is required for SnowflakeConnector"
            ) from exc

        params = {
            "user": user or os.getenv("SNOWFLAKE_USER"),
            "password": password or os.getenv("SNOWFLAKE_PASSWORD"),
            "account": account or os.getenv("SNOWFLAKE_ACCOUNT"),
            "warehouse": warehouse or os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": database or os.getenv("SNOWFLAKE_DATABASE"),
            "schema": schema or os.getenv("SNOWFLAKE_SCHEMA"),
        }
        if role or os.getenv("SNOWFLAKE_ROLE"):
            params["role"] = role or os.getenv("SNOWFLAKE_ROLE")

        if not params["user"] or not params["password"] or not params["account"]:
            raise ValueError(
                "SnowflakeConnector requires user, password and account information."
            )

        try:
            self._conn = snowflake.connector.connect(**params)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Failed to connect to Snowflake") from exc

    def execute(self, sql: str) -> Iterable[Tuple[Any, ...]]:
        cur = self._conn.cursor()
        try:
            cur.execute(sql)
            try:
                rows = cur.fetchall()
            except Exception:  # pragma: no cover
                rows = []
        finally:
            cur.close()
        return rows

    def close(self) -> None:
        self._conn.close()


class StarRocksConnector(DatabaseConnector):
    """StarRocks connector using the MySQL wire protocol.

    The implementation prefers :mod:`starrocksdb`, but falls back to
    :mod:`mysql.connector` which is commonly available.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        host = host or os.getenv("STARROCKS_HOST")
        port = port or (int(os.getenv("STARROCKS_PORT", "0")) or None)
        user = user or os.getenv("STARROCKS_USER")
        password = password or os.getenv("STARROCKS_PASSWORD")
        database = database or os.getenv("STARROCKS_DATABASE")

        if not host or not user or password is None:
            raise ValueError(
                "StarRocksConnector requires host, user and password information."
            )

        try:
            try:
                import starrocksdb  # type: ignore

                self._conn = starrocksdb.connect(
                    host=host,
                    port=port or 9030,
                    user=user,
                    password=password,
                    database=database,
                )
            except ImportError:  # pragma: no cover - optional dependency fallback
                import mysql.connector

                self._conn = mysql.connector.connect(
                    host=host,
                    port=port or 9030,
                    user=user,
                    password=password,
                    database=database,
                )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Failed to connect to StarRocks") from exc

    def execute(self, sql: str) -> Iterable[Tuple[Any, ...]]:
        cur = self._conn.cursor()
        try:
            cur.execute(sql)
            try:
                rows = cur.fetchall()
            except Exception:  # pragma: no cover
                rows = []
        finally:
            cur.close()
        return rows

    def close(self) -> None:
        self._conn.close()


@dataclass
class AlphaSQLEngine:
    """Interface for the AlphaSQL system with optional MCMC search.

    Parameters
    ----------
    api_key:
        Azure OpenAI API key. If ``None``, the value is read from the
        ``AZURE_OPENAI_KEY`` environment variable.
    endpoint:
        Azure OpenAI endpoint. If ``None``, the value is read from the
        ``AZURE_OPENAI_ENDPOINT`` environment variable.
    api_version:
        API version to use. Defaults to ``2024-08-01-preview`` which supports the
        ``o3`` reasoning models.
    deployment:
        Deployment or model name. Defaults to ``"o3-mini"``.
    temperature:
        Sampling temperature passed to the model. Defaults to ``0`` for
        deterministic outputs.
    beta:
        Exploration parameter used when converting quality scores to acceptance
        probabilities in the Metropolis-Hastings procedure. Higher values favour
        high-scoring proposals.
    """

    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    api_version: str = "2024-08-01-preview"
    deployment: str = "o3-mini"
    temperature: float = 0.0
    beta: float = 5.0

    def __post_init__(self) -> None:
        if AzureOpenAI is None:  # pragma: no cover - import guard
            raise ImportError(
                "The 'openai' package is required. Install it with 'pip install openai'."
            )

        key = self.api_key or os.getenv("AZURE_OPENAI_KEY")
        endpoint = self.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not key or not endpoint:
            raise ValueError(
                "Azure OpenAI credentials are missing. Set 'AZURE_OPENAI_KEY' and 'AZURE_OPENAI_ENDPOINT'."
            )

        self._client = AzureOpenAI(
            api_key=key,
            azure_endpoint=endpoint,
            api_version=self.api_version,
        )

    # ------------------------------------------------------------------
    # Model invocation helpers
    # ------------------------------------------------------------------
    def _complete(self, prompt: str) -> str:
        response = self._client.responses.create(
            model=self.deployment,
            input=[{"role": "user", "content": prompt}],
            reasoning={"effort": "medium"},
            temperature=self.temperature,
        )
        return response.output_text.strip()

    # ------------------------------------------------------------------
    # Planning utilities
    # ------------------------------------------------------------------
    def plan(self, question: str, schema: str) -> str:
        """Generate a reasoning plan for the query."""

        prompt = (
            "You are a data expert planning how to answer a question with SQL.\n"
            f"Question: {question}\n"
            f"Schema: {schema}\n"
            "Provide an ordered list of high level steps that lead to the answer."
        )
        return self._complete(prompt)

    def _base_sql_prompt(
        self,
        question: str,
        schema: str,
        plan_text: Optional[str] = None,
    ) -> str:
        parts = [
            "You are an expert SQL generator.",
            f"Question: {question}",
            f"Schema: {schema}",
        ]
        if plan_text:
            parts.append(f"Plan: {plan_text}")
        parts.append(
            "Generate only the SQL query that answers the question. Do not add explanations."
        )
        return "\n".join(parts)

    def nl_to_sql(self, question: str, schema: str, planning: bool = True) -> str:
        """Convert a natural language question to a SQL query."""

        plan_text = self.plan(question, schema) if planning else None
        return self._complete(self._base_sql_prompt(question, schema, plan_text))

    # ------------------------------------------------------------------
    # MCMC search utilities
    # ------------------------------------------------------------------
    def _score_sql(self, question: str, schema: str, sql: str) -> float:
        """Score a SQL candidate using the LLM on a 0-1 scale."""

        prompt = (
            "You are an expert SQL reviewer.\n"
            "Rate how well the following SQL answers the user's question given the schema.\n"
            "Respond with a floating point score between 0 and 1 inclusive.\n"
            f"Question: {question}\n"
            f"Schema: {schema}\n"
            f"SQL: {sql}\n"
            "Score:"
        )
        raw = self._complete(prompt)
        try:
            return max(0.0, min(1.0, float(raw.split()[0])))
        except Exception:  # pragma: no cover - parsing fallback
            return 0.0

    def _propose_sql(
        self,
        question: str,
        schema: str,
        current_sql: str,
        plan_text: Optional[str],
    ) -> str:
        prompt = (
            "You are refining a SQL query.\n"
            f"Question: {question}\n"
            f"Schema: {schema}\n"
            f"Current SQL: {current_sql}\n"
            "Provide an improved SQL query that better answers the question."
        )
        if plan_text:
            prompt = f"Plan: {plan_text}\n" + prompt
        return self._complete(prompt)

    def mcmc_search(
        self,
        question: str,
        schema: str,
        *,
        planning: bool = True,
        steps: int = 10,
        initial_sql: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        """Run a Metropolis-Hastings chain to explore SQL queries.

        Returns a dictionary containing the best SQL and the trace of scores.
        """

        rng = rng or random.Random()
        plan_text = self.plan(question, schema) if planning else None
        current_sql = initial_sql or self._complete(
            self._base_sql_prompt(question, schema, plan_text)
        )
        current_score = self._score_sql(question, schema, current_sql)

        best_sql = current_sql
        best_score = current_score
        trace: List[Dict[str, Any]] = [
            {"step": 0, "sql": current_sql, "score": current_score, "accepted": True}
        ]

        for step in range(1, steps + 1):
            proposal_sql = self._propose_sql(question, schema, current_sql, plan_text)
            proposal_score = self._score_sql(question, schema, proposal_sql)
            log_accept = self.beta * (proposal_score - current_score)
            accept_prob = min(1.0, math.exp(log_accept))
            accepted = rng.random() < accept_prob
            if accepted:
                current_sql = proposal_sql
                current_score = proposal_score
                if proposal_score >= best_score:
                    best_sql = proposal_sql
                    best_score = proposal_score
            trace.append(
                {
                    "step": step,
                    "sql": proposal_sql,
                    "score": proposal_score,
                    "accepted": accepted,
                    "accept_probability": accept_prob,
                }
            )

        return {"best_sql": best_sql, "best_score": best_score, "trace": trace}

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def generate_sql(
        self,
        question: str,
        schema: str,
        *,
        planning: bool = True,
        mcmc_steps: int = 0,
    ) -> str:
        """Generate SQL, optionally running MCMC refinement."""

        if mcmc_steps <= 0:
            return self.nl_to_sql(question, schema, planning=planning)

        result = self.mcmc_search(
            question,
            schema,
            planning=planning,
            steps=mcmc_steps,
        )
        return result["best_sql"]

    def execute(
        self,
        question: str,
        schema: str,
        *,
        connector: DatabaseConnector,
        planning: bool = True,
        mcmc_steps: int = 0,
    ) -> Dict[str, Any]:
        """Generate SQL and execute it on the provided connector."""

        sql = self.generate_sql(
            question,
            schema,
            planning=planning,
            mcmc_steps=mcmc_steps,
        )
        rows = list(connector.execute(sql))
        return {"sql": sql, "rows": rows}


__all__ = [
    "AlphaSQLEngine",
    "DatabaseConnector",
    "PostgresConnector",
    "SnowflakeConnector",
    "StarRocksConnector",
]
