# AlphaSQL

The implementation of the ICML 2025 AlphaSQL approach for natural-language-to-SQL translation, adapted for Azure OpenAI's `o3-mini` model.

This repository hosts a minimal yet end-to-end reproduction of the
[AlphaSQL](https://arxiv.org/abs/2502.17248) system. It integrates Azure OpenAI's
reasoning models, provides an optional **planning mode** where the model drafts a
plan prior to emitting SQL, includes an experimental Metropolis-Hastings
(MCMC) refinement loop, and supplies database connectors for Postgres,
Snowflake, and StarRocks so generated SQL can be executed directly.

## Installation

```bash
pip install openai
```

Set the Azure OpenAI credentials in your environment:

```bash
export AZURE_OPENAI_KEY="<your-key>"
export AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com/"
```

Optional database drivers (install the ones you need):

```bash
pip install psycopg[binary] snowflake-connector-python starrocksdb mysql-connector-python
```

## Example

```python
from alphasql import AlphaSQLEngine, PostgresConnector

engine = AlphaSQLEngine()
schema = "table users(id, name, registered_at)"
sql = engine.generate_sql(
    "How many users registered in 2023?",
    schema,
    planning=True,
    mcmc_steps=8,
)
print(sql)

connector = PostgresConnector()  # reads POSTGRES_DSN from the environment
try:
    result = engine.execute(
        "How many users registered in 2023?",
        schema,
        connector=connector,
        mcmc_steps=8,
    )
    print(result["rows"])
finally:
    connector.close()
```

The engine first creates a plan (enabled by default) and then generates SQL.
When MCMC steps are requested the model iteratively proposes refinements and
accepts them according to a Metropolis-Hastings acceptance rule. The connectors
wrap existing database client libraries and expect credentials to be available
through environment variables.

### CLI usage

```bash
python -m alphasql.cli "How many users registered in 2023?" "table users(id, name, registered_at)" --mcmc-steps 8
```

Use `--no-plan` to skip planning. Provide `--db-target postgres|snowflake|starrocks`
to execute queries using environment-provided credentials. Required variables:

| Target | Required variables |
| --- | --- |
| `postgres` | `POSTGRES_DSN` or keyword parameters supported by `psycopg`/`psycopg2` |
| `snowflake` | `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_ACCOUNT`, optionally `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_ROLE` |
| `starrocks` | `STARROCKS_HOST`, `STARROCKS_USER`, `STARROCKS_PASSWORD`, optionally `STARROCKS_PORT`, `STARROCKS_DATABASE` |

Ensure you have installed the relevant database drivers (`psycopg` or
`psycopg2`, `snowflake-connector-python`, `starrocksdb` or
`mysql-connector-python`).
