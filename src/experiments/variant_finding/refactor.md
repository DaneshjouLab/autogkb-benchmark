# Variant Extraction Refactor

## Goal
The goal of the module is to be able to take in an article/pmcid and return a list of variants found in the article. This should then be compared
against the ground truth variatns (from the benchmark) to evaluate the performance of the variant extraction method. Recall is the primary metric.

## Changes
This is a list of changes for the refactor (non-exhaustive)
- VariantExtractor interface should be implemented by experimental methods
- Prompts should be instantiated as part of the class
- We should have one consolidated eval function that takes the results of a variant extractor and the ground truth variants and generates an evaluation report
- We should have a shared utils file for variant extraction for all the needed functionality
- llm calls should be made using the general call_llm function
- The right variant extractor should be evaluted by creating the correct class via a generator function