# Custon Normalization - Our Package
Importing our made term_normalization package and doing a basic evaluation of results.
Let's get the raw variants and run them through our custom normalizer and see if things look in line with the ground truths? 

## Process
1. Run a regex search (import from highest performing regex experiment (v5?)) to get the terms from the benchmark papers. Save this to raw_variants.txt
2. Run our custom normalizer on the terms
3. Save the mappings with some details (what was the original term, what was the normalized term, what was the data source, what confidence level, etc.)
4. Also include some summary statistics (how many terms did we get, how many were normalized, etc.)