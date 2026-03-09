# Database Sync

Syncs the local `generations.jsonl` file with a Railway Postgres database (`analysis_jobs` table).

## Setup

Set `DATABASE_URL` in your `.env` file pointing to the Railway Postgres instance.

## Commands

```bash
python -m generation.sync <command> [--override]
```

### push (local -> DB)

Upserts local JSONL records into the database. Matching records (by `id`) are updated; new records are inserted.

```bash
python -m generation.sync push              # upsert
python -m generation.sync push --override   # delete all DB rows first, then insert
```

### pull (DB -> local)

Merges database records into the local JSONL. On ID conflicts, DB records win.

```bash
python -m generation.sync pull              # merge (DB wins on conflict)
python -m generation.sync pull --override   # replace local JSONL entirely with DB data
```

### backup

Dumps all DB rows to a timestamped JSONL file in `data/backups/`.

```bash
python -m generation.sync backup
# -> data/backups/generations_20260309_233035.jsonl
```

### migrate

One-time schema migration: consolidates legacy columns (`annotations`, `annotation_citations`, `annotation_data`, `progress`) into a single `json_content` JSONB column and reorders columns to match the local insert order. Run `backup` first.

```bash
python -m generation.sync backup   # always backup before migrating
python -m generation.sync migrate
```

## DB Schema

Table: `analysis_jobs`

| Column              | Type        | Notes                                              |
|---------------------|-------------|-----------------------------------------------------|
| id                  | UUID (PK)   | Generated client-side via `uuid4`                   |
| pmid                | TEXT        | PubMed ID (nullable)                                |
| pmcid               | TEXT        | PubMed Central ID                                   |
| title               | TEXT        | Article title (nullable)                            |
| status              | TEXT        | `in_progress`, `completed`, or `error`              |
| markdown_content    | TEXT        | Full article text (maps to `text_content` locally)  |
| json_content        | JSONB       | Contains `annotations`, `annotation_citations`, `annotation_data` |
| generation_metadata | JSONB       | Pipeline run metadata (config, models, timing, git sha) |
| error               | TEXT        | Error message if status is `error` (nullable)       |
| created_at          | TIMESTAMPTZ | Record creation timestamp                           |
| updated_at          | TIMESTAMPTZ | Last sync timestamp                                 |

## Field Mapping

Some fields are renamed between the local `GenerationRecord` model and the DB:

| Local field       | DB column          |
|-------------------|--------------------|
| `text_content`    | `markdown_content` |
| `timestamp`       | `created_at`       |
| `annotations`     | `json_content.annotations` |
| `annotation_citations` | `json_content.annotation_citations` |
| `annotation_data` | `json_content.annotation_data` |
