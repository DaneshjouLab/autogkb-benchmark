# Batch Judge Ask

## Overview
This should be very similar to llm_judge_ask experiment except instead of querying the model + text for each variant, we should batch process all variants for a given PMCID at once. So we should have a single prompt that includes all variants for a given PMCID and the model should generate all sentences and explanations at once.

Should have identical file outputs and judging as done in llm_judge_ask.

