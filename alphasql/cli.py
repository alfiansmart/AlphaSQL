"""Command line interface for AlphaSQL."""
from __future__ import annotations

import argparse

from . import (
    AlphaSQLEngine,
    PostgresConnector,
    SnowflakeConnector,
    StarRocksConnector,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaSQL NL2SQL")
    parser.add_argument("question", help="Natural language question")
    parser.add_argument("schema", help="Database schema description")
    parser.add_argument("--no-plan", action="store_true", help="Disable planning mode")
    parser.add_argument(
        "--mcmc-steps",
        type=int,
        default=0,
        help="Number of MCMC refinement steps (0 disables MCMC)",
    )
    parser.add_argument(
        "--db-target",
        choices=["postgres", "snowflake", "starrocks"],
        help="Execute the generated query against a live database",
    )
    args = parser.parse_args()

    engine = AlphaSQLEngine()
    planning = not args.no_plan

    if args.db_target:
        if args.db_target == "postgres":
            connector = PostgresConnector()
        elif args.db_target == "snowflake":
            connector = SnowflakeConnector()
        else:
            connector = StarRocksConnector()

        result = engine.execute(
            args.question,
            args.schema,
            connector=connector,
            planning=planning,
            mcmc_steps=args.mcmc_steps,
        )
        print(result["sql"])
        for row in result["rows"]:
            print(row)
        connector.close()
    else:
        sql = engine.generate_sql(
            args.question,
            args.schema,
            planning=planning,
            mcmc_steps=args.mcmc_steps,
        )
        print(sql)


if __name__ == "__main__":  # pragma: no cover
    main()
