# Refactor

## High Level Goals
- Have the benchmark and generation pipeline in separate folders at the root
- both should just interact with the data folder without needing to know about each other
- ideally a single command should be able to run the entire pipeline against a pmid (converts to pmcid, skips/throws an error if pmcid doesn't exist, runs generation)
- generation outputs should be saved to jsonl file in the data folder with the following columns:
    - id (unique identifier for the generation)
    - pmid
    - pmcid
    - title
    - text_content 
    - annotations
    - annotation_citations
    - timestamp (of generation)
