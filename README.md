# AlphaSQL

AlphaSQL is a lightweight research toolkit that combines a flexible
Metropolis-Hastings Markov Chain Monte Carlo (MCMC) engine with utilities for
exploring SQL query spaces against PostgreSQL or Snowflake backends. The goal
is to make it easy to prototype adaptive SQL systems that search over query
templates while learning from database feedback.

## Features

* **Generic MCMC sampler** – drive any scoring function that returns a scalar
  value for a SQL query.
* **Template-aware SQL proposals** – mutate `SELECT` statements by swapping
  tables, columns, predicates, limits, ordering clauses, and `DISTINCT` usage.
* **Database connectivity** – simple helpers for connecting to PostgreSQL
  (via `psycopg`/`psycopg2`) or Snowflake using environment variables or
  explicit configuration objects.
* **Pluggable evaluation** – run queries, fetch results, and score them with a
  Python callback.

## Installation

AlphaSQL is a pure Python package. Install it alongside the database drivers
required for your target backend:

```bash
pip install psycopg psycopg2-binary snowflake-connector-python
```

Only the drivers you need must be installed. The package itself has no third
party runtime dependencies beyond the standard library.

Clone this repository and make it importable (either by installing it in a
virtual environment or by adding the repository root to your `PYTHONPATH`).

## Configuration

### Environment variables

The `DatabaseConnector.from_environment` helper reads configuration from
environment variables with the `ALPHASQL_` prefix. The most common options are:

| Variable | Description |
| -------- | ----------- |
| `ALPHASQL_HOST` / `ALPHASQL_PORT` | Database host/port (PostgreSQL). |
| `ALPHASQL_DATABASE` | Database name (both backends). |
| `ALPHASQL_USER` / `ALPHASQL_PASSWORD` | Authentication credentials. |
| `ALPHASQL_ACCOUNT` | Snowflake account identifier. |
| `ALPHASQL_WAREHOUSE` | Snowflake warehouse. |
| `ALPHASQL_ROLE` / `ALPHASQL_SCHEMA` | Optional Snowflake role/schema. |
| `ALPHASQL_DSN` | PostgreSQL connection string (optional alternative to host/port). |
| `ALPHASQL_TIMEOUT` | Connection timeout in seconds. |
| `ALPHASQL_KEEPALIVE` | PostgreSQL TCP keepalive idle seconds. |
| `ALPHASQL_PARAMS` | Extra parameters formatted as comma-separated `key=value` pairs. |

Set the variables appropriate for either PostgreSQL or Snowflake before running
the sampler or supply equivalent values programmatically through
`ConnectionConfig`.

### Manual configuration

Alternatively, instantiate a `ConnectionConfig` and pass it to
`DatabaseConnector`:

```python
from alphasql import ConnectionConfig, DatabaseConnector, DatabaseType

config = ConnectionConfig(
    database_type=DatabaseType.POSTGRES,
    host="localhost",
    port=5432,
    database="analytics",
    user="researcher",
    password="secret",
)
connector = DatabaseConnector(config)
connection = connector.connect()
```

## Running the sampler

An end-to-end example is provided in `examples/run_sampler.py`.

```bash
python examples/run_sampler.py --iterations 200 --burn-in 20 --backend postgres
```

By default the script expects valid database credentials via environment
variables. For local experimentation without a running database, enable the
in-memory evaluator:

```bash
python examples/run_sampler.py --dry-run
```

The script prints the best query discovered, its score, and basic diagnostics
for the Markov chain.

## Using AlphaSQL in your project

1. Define a scoring callback that accepts the rows returned by a query and any
   metadata you wish to track, returning a floating point score.
2. Instantiate `QueryEvaluator` with either a real database connection or a
   mocked connection for offline experimentation.
3. Create a `SQLMutationProposal` describing the tables, columns, filters, and
   limits you want the sampler to explore.
4. Configure `MCMCConfig` with the desired number of iterations, burn-in, and
   random seed.
5. Run `MCMCSampler.run(initial_query, metadata=initial_metadata)` and consume
   the resulting trace or the best state.

The library is intentionally modular—swap in your own proposal distribution,
database connection management, or evaluation strategy as needed.
