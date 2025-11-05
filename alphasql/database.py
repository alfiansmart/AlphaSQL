"""Database connection helpers for AlphaSQL."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class DatabaseType(str, Enum):
    """Enumeration of database backends supported by AlphaSQL."""

    POSTGRES = "postgres"
    SNOWFLAKE = "snowflake"


@dataclass
class ConnectionConfig:
    """Parameters required to establish a database connection."""

    database_type: DatabaseType
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    account: Optional[str] = None
    warehouse: Optional[str] = None
    role: Optional[str] = None
    schema: Optional[str] = None
    dsn: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    keepalive: Optional[int] = None


class DatabaseConnector:
    """Factory that produces database connections for MCMC evaluation."""

    def __init__(self, config: ConnectionConfig) -> None:
        self._config = config
        self._connection = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def connect(self):
        """Instantiate a database connection using the configured backend."""

        if self._config.database_type == DatabaseType.POSTGRES:
            return self._connect_postgres()
        if self._config.database_type == DatabaseType.SNOWFLAKE:
            return self._connect_snowflake()
        raise ValueError(f"Unsupported database type: {self._config.database_type}")

    def __enter__(self):
        self._connection = self.connect()
        return self._connection

    def __exit__(self, exc_type, exc, exc_tb):
        try:
            if self._connection is not None:
                self._connection.close()
        finally:
            self._connection = None
        return False

    def close(self) -> None:
        """Close the managed connection if it is open."""

        if self._connection is not None:
            self._connection.close()
            self._connection = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_environment(
        cls,
        database_type: DatabaseType,
        *,
        prefix: str = "ALPHASQL_",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "DatabaseConnector":
        """Create a connector using environment variables.

        Parameters
        ----------
        database_type:
            The backend to target.
        prefix:
            Prefix applied to environment variables. Defaults to ``ALPHASQL_``.
        overrides:
            Optional dictionary of values that take precedence over the
            environment.
        """

        overrides = overrides or {}
        env = os.environ

        def get(name: str, default: Optional[str] = None) -> Optional[str]:
            if name in overrides:
                return overrides[name]
            return env.get(prefix + name, default)

        params: Dict[str, Any] = {}
        raw_params = get("PARAMS")
        if raw_params:
            for entry in raw_params.split(","):
                if not entry:
                    continue
                key, _, value = entry.partition("=")
                if key:
                    params[key.strip()] = value.strip()

        config = ConnectionConfig(
            database_type=database_type,
            host=get("HOST"),
            port=int(get("PORT")) if get("PORT") else None,
            database=get("DATABASE"),
            user=get("USER"),
            password=get("PASSWORD"),
            account=get("ACCOUNT"),
            warehouse=get("WAREHOUSE"),
            role=get("ROLE"),
            schema=get("SCHEMA"),
            dsn=get("DSN"),
            params=params,
        )

        timeout = get("TIMEOUT")
        if timeout:
            config.timeout = int(timeout)
        keepalive = get("KEEPALIVE")
        if keepalive:
            config.keepalive = int(keepalive)

        return cls(config)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------
    def _connect_postgres(self):
        """Create a PostgreSQL connection using psycopg (v3) or psycopg2."""

        conn_args: Dict[str, Any] = dict(self._config.params)
        if self._config.dsn:
            conn_args.setdefault("conninfo", self._config.dsn)
        else:
            if self._config.database:
                conn_args.setdefault("dbname", self._config.database)
            if self._config.user:
                conn_args.setdefault("user", self._config.user)
            if self._config.password:
                conn_args.setdefault("password", self._config.password)
            if self._config.host:
                conn_args.setdefault("host", self._config.host)
            if self._config.port:
                conn_args.setdefault("port", self._config.port)
        if self._config.timeout is not None:
            conn_args.setdefault("connect_timeout", self._config.timeout)
        if self._config.keepalive is not None:
            conn_args.setdefault("keepalives", 1)
            conn_args.setdefault("keepalives_idle", self._config.keepalive)

        try:
            import psycopg  # type: ignore

            return psycopg.connect(**conn_args)
        except ImportError:
            try:
                import psycopg2  # type: ignore

                return psycopg2.connect(**conn_args)
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError(
                    "psycopg (v3) or psycopg2 must be installed to use the PostgreSQL backend"
                ) from exc

    def _connect_snowflake(self):
        """Create a Snowflake connection using the official connector."""

        conn_args: Dict[str, Any] = dict(self._config.params)
        if self._config.account:
            conn_args.setdefault("account", self._config.account)
        if self._config.user:
            conn_args.setdefault("user", self._config.user)
        if self._config.password:
            conn_args.setdefault("password", self._config.password)
        if self._config.warehouse:
            conn_args.setdefault("warehouse", self._config.warehouse)
        if self._config.role:
            conn_args.setdefault("role", self._config.role)
        if self._config.database:
            conn_args.setdefault("database", self._config.database)
        if self._config.schema:
            conn_args.setdefault("schema", self._config.schema)

        try:
            import snowflake.connector  # type: ignore

            return snowflake.connector.connect(**conn_args)
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "snowflake-connector-python must be installed to use the Snowflake backend"
            ) from exc
