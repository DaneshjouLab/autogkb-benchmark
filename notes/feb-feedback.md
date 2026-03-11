Feedback on AutoGKB Feb 2026

## Summary of Pipeline Updates Needed

1. **Remove the Clinical Implications section** — consistently flagged as unhelpful/speculative across all reviewers.
2. **Ground extraction in Results & Methods** — citations and extracted claims should come from Results, Methods, and supplementary materials, not the Discussion section.
3. **Feed Key Findings into the annotation pipeline** — the annotation summary generation step should have access to Key Findings so that all findings (e.g., bleeding risk, PM vs IM comparisons, "no association" results) appear in the Variant Association section.
4. **Only annotate variants actually studied by the paper** — prioritize variants explicitly mentioned in Methods, Results, and relevant supplementary materials. Don't surface variants only mentioned in passing in the Discussion (e.g., rs147390019) or only genotyped but not analyzed (e.g., rs28399499).
5. **Don't fabricate genotype comparisons or omit reported statistics** — only report comparisons explicitly stated in the paper; don't claim p-values were "not provided" when they were.
6. **Split multi-phenotype associations into separate annotations** — don't group different effect directions for different phenotypes into a single annotation (e.g., decreased risk for hematological toxicity but increased risk for mucositis should be two entries).
7. **Capture all associations per variant** — e.g., if a variant has both a PK association and a "no association" finding for efficacy, generate annotations for both.
8. **Resolve star allele / rsID aliasing** — link aliases (e.g., DPYD*9 = rs1801265, TPMT*3C = rs1142345) and extract information reported under either name.
9. **Keep association text consistent across star alleles in the same metabolizer group** — when findings are about metabolizer phenotypes, the association sentence should be identical for each constituent allele.
10. **Stop using "null" for empty genotype fields** — use blank or "N/A" since "null" has a specific meaning in pharmacogenomics (gene deletion alleles).
11. **Report alleles on the plus chromosomal strand** — complement alleles for minus-strand genes when needed, and note when this is done.
12. **Fix "was not mentioned in the article" text** — variants that are mentioned in the article are being incorrectly labeled as not mentioned.
13. **Fix the analysis/scoring system** — scores (33-36%) don't align with curator assessment. False positive classifications are often wrong (valid findings labeled as false positives). Phenotype Annotation predictions are too focused on wildtype (*1) alleles.

---


https://autogkb-app.vercel.app/dashboard

General comments and questions
The information in the “Formatted View” tab is a  good starting point for experienced curators writing VAs. 
Curators think this information could be a time-saver.
Also, a great way to bring papers to curators’ attention for review.
It may not be useful for novice curators and cannot be simply submitted (even with review) directly into the DB.
We typically rely heavily on the Results and Methods section of papers, rather than the Discussion section and suggest prompting similarly if possible.
Some of the information seems to be “interpreted” from the paper or derived from the Discussion section where authors try to provide reasons and context for the results.
For example, the Clinical Implications section seems like it may be based on interpretation of data. What was the original purpose of this section? What is the Explanation part of the Association?
Initial, brief review leads us to think that the model is not great at catching study parameters and likely not functional assays either (we need to do more testing to verify this).
Analysis tab: The score does not seem to align with how we would evaluate the output.
Questions:
Is the Variant Association section based at all on our Variant Annotations, or just the straight AI interpretation of the paper? We ask because it seems like there are differences between the Key Findings section and the Variant Association section, with the latter mirroring more of our annotations rather than all of the paper findings. So we were wondering if there is a step after the initial prompt to get the Key Findings that somehow includes the existing VAs.
Related question - how is output prioritized for viewing in the tool? The paper may contain multiple phenotypes, but it looks like only 1 is being prioritized for the Association section.
Sorry, I forget - are you able to use supplement files from papers?



From Caroline

Notes on autogkb annotation of PMCID12331468/PMID:40786508


1. Please don't use "null" to indicate empty genotype field, this could lead to confusion as the term null is used for alleles where there is a gene deletion.

2. We report all our associations on the plus chromosomal strand. This means sometimes for genes that are located on the minus strand (and people tend to report cDNA changes rather than genomic changes) we have to complement the alleles. We (should) note when this is done.

3. In the section Formatted view Variant Associations it repeatedly says "was not mentioned in the article" but think should be is mentioned. One that it gets correct is rs717620 where it says "rs717620 mentioned in article but not studied by paper."

4. Overall, it does a good job of finding the associations in the paper. I think the Background and Key findings are good. Yes, several of them are clearly stated in the abstract but it also finds others in the results that had not been annotated but should/could have been. 
So why does it get analytic score of 36%?
Not sure why it put into Drug Annotations false positives the associations it also put into Phenotype Associations (need to look at another paper to see if this happens there too)
It had trouble with the run-on sentence about GSTP1 where it has decreased risk for two phenotypes (hematological and nervous system disorders) and increased for one (mucositis). I split the annotations. 

5. It does not show in the Variant annotations section Formatted view that the paper tells you what DPYD*9 is (paper quote = "DPYD c.85T>C (DPYD*9) (rs1801265)")
and other non-rs names 

From Li:
PMC2859392
the key findings section looks good
clinical implications section less useful, more extrapolation from the discussion 
variant associations section:
in table, no need to mention rs28399499, and CYP2B6*6, only genotyped rs3745274, not really *6
CYP2B6*516 is wrong,  it’s 516G>T, rs3745274, why repeat the same annotation as rs3745274?
seems like only one variant annotation generated for rs3745274, PK is there, can do another one for efficacy? (no association)
don’t quite get the analysis view, often “false positives are not false positives"

PMC4916189
rs3745274: captured the association with discontinuation, missed the No association with virological response
some of the citations don’t link properly to the full text, it’s very helpful to see where in the text support the annotation,  recommend adding section header in text . eg. results, discussion, methods etc.
caught a few annotations that we didn’t do (especially the no association ones), not false positives; might be helpful to have a column for curators to check/mark whether to accept, accept with revision, or reject the annotations?
like the key findings sections, good summarization


From Evangelia
PMC6465603
 “NUDT15 genotype influenced thiopurine exposure and dosing: heterozygotes (CT) had higher 6-thioguanine nucleotide (6-TGN) levels (median 276.6 vs. 145.3 pmol/8×10^8 RBC; P=0.002) and required lower maintenance AZA doses (median 0.96 vs. 1.23 mg/kg/day; P=0.028).” However, this information is captured below, where details about the variant are included in the table, and in the association: “Genotype CT of rs116855232 is associated with increased risk of leukopenia and decreased maintenance dose of azathioprine in people with autoimmune hepatitis (AIH), AIH–primary biliary cholangitis variant syndrome, and related cirrhosis compared to genotype CC (statistically significant).”

In the explanation for the variant rs116855232, it states: “In a single-center cohort of 149 Chinese patients on azathioprine, the study genotyped rs116855232 and found a significant association with AZA-induced leukopenia and lower tolerated maintenance doses; heterozygous carriers (CT) also had lower AZA dose-to-6-TGN concentration ratios, though exact p-values were not provided in the excerpt.” This mentions that the p-values were not provided, but they were.

Regarding rs147390019, is there a reason it appears first in the table and in the subsequent section, given that it is an rsID that was not studied and was mentioned only once in the discussion? Additionally, four citations are provided, none of which include information about this rsID.

For rs1142345, the statement reads: “Association: Genotypes AG + GG of rs1142345 are not associated with the risk of leukopenia or maintenance dose of azathioprine in people with autoimmune hepatitis (AIH), AIH–primary biliary cholangitis variant syndrome, and related cirrhosis compared to genotype AA (not statistically significant).” I could not find anywhere in the manuscript or supplementary materials addressing the genotypes AG + GG vs. AA, and although there are four citations, none address these specific genotypes.

Additionally, for rs1142345, there should be another citation for this excerpt in the manuscript: “… while TPMT∗3C alleles were only observed in four subjects, including three patients who were heterozygotes (TC) and one patient who was homozygous (CC).” While it is cited from the discussion that this variant is rare, the excerpt is not included. This may be because TPMT*3C is mentioned instead of rs1142345.

From Katrin
PMC11971672
I would find it helpful to have a bulletin list of drug(s), gene(s), variation (rsID, alleles, metabolizer phenotypes), outcomes (diseases, side effects, PK parameter) included in the article 
Formatted View
Key Findings section content looks mainly correct 
E.g. HR range values are used twice for unadjusted and adjusted, while the unadjusted has a different range, resulting in the different p-value which was picked up right

Clinical Implication section not helpful for article annotation

Variant Associations

IF you want to index the finding per star allele the sentence should not change rather be the same for each variant for this example since the findings are on metabolizer groups and not individual alleles
e.g. *2 variant statement
Genotypes *1/*2 + *2/*2 + *2/*3 + *2/*17 of CYP2C19*2 are associated with increased likelihood of composite cardiovascular events within 6 months when treated with clopidogrel in people with acute ischemic stroke in South Korea as compared to noncarrier genotypes (*1/*1, *1/*17, *17/*17) (log-rank P = .048). 
Removed *1/*3 + *3/*3 + *3/*17 which might be needed to reach significance
Key findings are missing in the Variant Association section
E.g. 	bleeding risk
	Comparison between PM and IM
Don’t understand the connection between Formatted View and Analysis and why it scores only 33%
What is the purpose of keeping it to ClinPGx’s annotation forms?
Difference between the Drug Annotation and Phenotype Annotation form is that the Phenotype form includes a district outcome that is mapped to ClinPGx vocabulary (often disease) while the Drug Annotation is mostly used for PK outcomes like differences in concentration, metabolism, or generic outcome measures such as response, remission, etc but both are meant to capture in-vivo variation-drug-outcome associations

Drug Annotation section Extra Predictions (False Positives) seem to be random terms
Phenotype Annotation section seems to be heavenly focused on *1 and picks appear rather random, why not use the “non-normal” alleles

