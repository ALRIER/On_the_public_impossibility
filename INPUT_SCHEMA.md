# Neutral input contract

The runtime reads only neutral identifiers.

## `inputs/r04/evidence.csv`

Required metadata columns:

- `ROW_ID`: opaque row identifier.
- `GROUP_ID`: opaque validation group identifier.
- `LOCKBOX`: 0 for development evidence, 1 for reserved surrogate evaluation.
- `P00`: opaque structural-profile ID.

Feature columns are `X001`, `X002`, ...  
Response columns are `Y001`, `Y002`, ...

## `inputs/r04/scales.csv`

Columns:

- `response_id`: one of the `Y###` response IDs.
- `scale`: strictly positive numerical normalization scale.

## `inputs/r04/residual_ids.csv`

Single column `response_id`. These are neutral response IDs eligible for targeted residual acquisition when their predictive-quality gate is satisfied.

No semantic mapping belongs in this repository.
