"""Proposal distributions for AlphaSQL MCMC sampling."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional, Sequence, Tuple

from .mcmc import MCMCState, Proposal, ProposalError


@dataclass(frozen=True)
class SQLTemplate:
    """Representation of a simplified SQL ``SELECT`` statement."""

    table: str
    select_columns: Tuple[str, ...]
    filters: Tuple[str, ...] = ()
    limit: Optional[int] = None
    order_by: Optional[str] = None
    distinct: bool = False

    def render(self) -> str:
        columns = ", ".join(self.select_columns) if self.select_columns else "*"
        keyword = "DISTINCT " if self.distinct else ""
        query = f"SELECT {keyword}{columns} FROM {self.table}"
        if self.filters:
            query += " WHERE " + " AND ".join(self.filters)
        if self.order_by:
            query += f" ORDER BY {self.order_by}"
        if self.limit is not None:
            query += f" LIMIT {self.limit}"
        return query

    def clone(
        self,
        *,
        table: Optional[str] = None,
        select_columns: Optional[Sequence[str]] = None,
        filters: Optional[Sequence[str]] = None,
        limit: Optional[Optional[int]] = None,
        order_by: Optional[Optional[str]] = None,
        distinct: Optional[bool] = None,
    ) -> "SQLTemplate":
        return SQLTemplate(
            table=table if table is not None else self.table,
            select_columns=tuple(select_columns) if select_columns is not None else self.select_columns,
            filters=tuple(filters) if filters is not None else self.filters,
            limit=self.limit if limit is None else limit,
            order_by=self.order_by if order_by is None else order_by,
            distinct=self.distinct if distinct is None else distinct,
        )


class SQLMutationProposal(Proposal):
    """Mutation-based proposal that samples SQL queries from a template space."""

    def __init__(
        self,
        tables: Mapping[str, Sequence[str]],
        *,
        filter_bank: Optional[Mapping[str, Sequence[str]]] = None,
        orderings: Optional[Mapping[str, Sequence[str]]] = None,
        candidate_limits: Optional[Sequence[int]] = None,
        default_table: Optional[str] = None,
    ) -> None:
        if not tables:
            raise ValueError("At least one table with columns must be provided")
        self._tables = {table: tuple(columns) for table, columns in tables.items()}
        self._filters = {table: tuple(filters) for table, filters in (filter_bank or {}).items()}
        self._orderings = {table: tuple(ordering) for table, ordering in (orderings or {}).items()}
        self._limits = tuple(candidate_limits) if candidate_limits else ()
        self._default_table = default_table or next(iter(self._tables))
        if self._default_table not in self._tables:
            raise ValueError("default_table must be one of the supplied tables")

    def _ensure_template(self, state: MCMCState) -> SQLTemplate:
        template = state.metadata.get("template")
        if isinstance(template, SQLTemplate):
            return template
        columns = self._tables[self._default_table]
        if not columns:
            raise ProposalError(f"No columns registered for table '{self._default_table}'")
        default_template = SQLTemplate(
            table=self._default_table,
            select_columns=(columns[0],),
        )
        return default_template

    def _pick_other(self, current: str, choices: Sequence[str], rng: random.Random) -> str:
        alternatives = [choice for choice in choices if choice != current]
        if not alternatives:
            return current
        return rng.choice(alternatives)

    def _choose_table(self, rng: random.Random, current_table: Optional[str] = None) -> str:
        if current_table and current_table in self._tables and rng.random() > 0.2:
            return current_table
        return rng.choice(list(self._tables))

    def _mutate_columns(self, template: SQLTemplate, rng: random.Random) -> Sequence[str]:
        available_columns = self._tables[template.table]
        columns = list(template.select_columns)
        if not columns:
            columns.append(rng.choice(available_columns))
        action = rng.choice(["swap", "add", "remove"])
        if action == "swap" and columns:
            idx = rng.randrange(len(columns))
            columns[idx] = self._pick_other(columns[idx], available_columns, rng)
        elif action == "add" and len(columns) < len(available_columns):
            missing = [col for col in available_columns if col not in columns]
            if missing:
                columns.append(rng.choice(missing))
        elif action == "remove" and len(columns) > 1:
            idx = rng.randrange(len(columns))
            columns.pop(idx)
        return columns

    def _mutate_filters(self, template: SQLTemplate, rng: random.Random) -> Sequence[str]:
        bank = self._filters.get(template.table)
        filters = list(template.filters)
        if not bank:
            return filters
        action = rng.choice(["add", "remove", "swap"])
        if action == "add":
            remaining = [flt for flt in bank if flt not in filters]
            if remaining:
                filters.append(rng.choice(remaining))
        elif action == "remove" and filters:
            idx = rng.randrange(len(filters))
            filters.pop(idx)
        elif action == "swap" and filters:
            idx = rng.randrange(len(filters))
            filters[idx] = rng.choice(bank)
        return filters

    def _mutate_limit(self, template: SQLTemplate, rng: random.Random) -> Optional[int]:
        if not self._limits:
            return template.limit
        if template.limit is None:
            return rng.choice(self._limits)
        if rng.random() < 0.3:
            return None
        return rng.choice(self._limits)

    def _mutate_order(self, template: SQLTemplate, rng: random.Random) -> Optional[str]:
        choices = self._orderings.get(template.table)
        if not choices:
            return template.order_by
        if template.order_by and rng.random() < 0.4:
            return None
        return rng.choice(choices)

    def propose(self, state: MCMCState, rng: random.Random) -> MCMCState:
        template = self._ensure_template(state)
        columns = template.select_columns
        table = template.table

        operations = ["table", "columns", "filters", "limit", "order", "distinct"]
        operation = rng.choice(operations)

        new_template = template
        if operation == "table":
            table = self._choose_table(rng, template.table)
            available_columns = self._tables[table]
            columns = (available_columns[0],)
            filters = self._filters.get(table, ())
            order_by = self._orderings.get(table, ())
            new_template = template.clone(
                table=table,
                select_columns=(available_columns[0],),
                filters=filters[:1] if filters else (),
                order_by=order_by[0] if order_by else None,
            )
        elif operation == "columns":
            columns = tuple(self._mutate_columns(template, rng))
            new_template = template.clone(select_columns=columns)
        elif operation == "filters":
            new_filters = tuple(self._mutate_filters(template, rng))
            new_template = template.clone(filters=new_filters)
        elif operation == "limit":
            new_limit = self._mutate_limit(template, rng)
            new_template = template.clone(limit=new_limit)
        elif operation == "order":
            new_order = self._mutate_order(template, rng)
            new_template = template.clone(order_by=new_order)
        elif operation == "distinct":
            new_template = template.clone(distinct=not template.distinct)

        new_metadata: MutableMapping[str, object] = dict(state.metadata)
        new_metadata["template"] = new_template
        new_metadata["operation"] = operation
        new_metadata["table"] = new_template.table
        new_metadata["columns"] = new_template.select_columns

        return MCMCState(query=new_template.render(), score=state.score, metadata=new_metadata)  # score updated by sampler

    def log_transition_probability(self, from_state: MCMCState, to_state: MCMCState) -> float:
        # This proposal is symmetric with respect to the discrete move set, so
        # the transition probabilities cancel in the MH ratio.
        return 0.0
