The Boughter dataset used in makijling/polyreactivity does not originate from a single study—it is a composite built from several previously‑published ELISA experiments.  In their eLife study on antibody polyreactivity, Boughter et al. described a database of 1 053 heavy‑chain antibody sequences; roughly half (529) bound ≥4 out of 6–7 antigens and were labelled poly‑reactive while 524 bound none of the test ligands and were labelled non‑polyreactive.  The sequences came from multiple experiments: influenza‑reactive B‑cell antibodies (eLife 2020; Immunity 2020), gut‑reactive HIV antibodies (Cell Reports 2018 & 2019), naturally‑occurring anti‑HIV antibodies (Nature 2010), memory‑B‑cell HIV antibodies (PLOS ONE 2011) and natural polyreactive mouse IgA antibodies (Science 2017).  The data are released in the AIMS GitHub repository and were later reformatted in makijling/polyreactivity.

### How the dataset was curated

* **Reading sequences and counts from AIMS.**
  The script `rebuild_boughter_from_counts.py` in makijling/polyreactivity reads DNA FASTA files and per‑antigen ELISA counts from `data/AIMS_manuscripts/app_data/full_sequences`.  It uses Biopython to translate nucleotide sequences into amino‑acid heavy‑chain sequences, trims them to the variable heavy (VH) domain using ANARCI numbering, and extracts the CDRH3 loop.  Each record is given an ID and lineage identifier (family | CDRH3) and is annotated with a species flag (mouse datasets are flagged as mouse, others as human).

* **Mapping counts to polyreactivity.**
  ELISA results are stored as counts (“NumReact”) or binary flags (“Y/N”).  For numeric files (flu, gut\_HIV, nat\_HIV, nat\_cntrl), the script reads the integer number of ligands bound; for binary files (plos\_HIV and mouse\_IgA) it maps “Y” to an “active value” representing a high count (4 for PLOS‑HIV, 7 for mouse IgA) and “N” to 0.  This mapping approximates the number of antigens tested in those assays.  The script records the total count of sequences with zero, mild (1–3), and high (>3) reactivity for auditing.

* **Filtering and deduplicating.**
  Boughter et al. noted that antibodies binding only 2–3 ligands may differ mechanistically from those binding 4–7 and therefore removed intermediate cases when training classifiers.  The script follows this recommendation: it keeps only sequences with `reactivity_count == 0` (labelled non‑polyreactive) or `reactivity_count > 3` (polyreactive).  Sequences with counts 1–3 are discarded.  Duplicate heavy‑chain sequences are collapsed by keeping the copy with the highest reactivity count.  Sequences longer than 1 022 amino‑acids are removed to satisfy language‑model input limits.  A binary label is then assigned (`label=1` for >3 ligands, `0` for 0 ligands).

* **Output files.**
  The script writes two CSV files: a “full” file containing every heavy‑chain with its count and a “filtered” file (boughter\_counts.csv) containing only sequences used for training.  An audit JSON lists the number of sequences retained and dropped by each filter step.  In the reproduced repository the filtered Boughter dataset contains 970 heavy‑chain sequences with 501 positives and 469 negatives (positive rate ≈0.52), which matches the counts reported in the `dataset_split_summary` file and reproduces the \~half‑polyreactive distribution reported by Boughter et al.  The file `boughter_counts_rebuilt.csv` shows that heavy‑chain sequences are taken from the influenza, HIV and mouse IgA datasets and labelled according to the above rules.

### Verification of the curation

* **Consistency with the published study.**
  In the eLife paper Boughter et al. describe a “parsed” dataset where antibodies binding 4–7 ligands are labelled polyreactive, antibodies binding none are labelled non‑reactive and those binding 1–3 ligands are removed.  The `polyreactivity` script applies the same threshold and removal of intermediate counts.  The table in the eLife article shows that the full dataset contains 1053 sequences (529 polyreactive, 524 non) with a roughly even split among Mouse IgA, HIV‑reactive and influenza‑reactive sets.  The curated dataset in makijling/polyreactivity drops sequences with ambiguous counts and duplicate heavy chains, yielding 970 sequences, which explains why the reproduced dataset is slightly smaller than the 1053 sequences in the published study.

* **Cross‑reference to external descriptions.**
  The Oxford Protein Informatics Group’s blog summarises the Boughter dataset as “combined previously described datasets of poly‑reactive antibodies determined by ELISA” and states that it contains 1 053 antibodies with roughly half poly‑reactive and 445 mouse antibodies.  This independent description aligns with the eLife counts and with the numbers derived from the processed dataset.

* **Data availability and provenance.**
  The `AIMS_manuscripts` README lists the source papers for each sub‑dataset and notes that quantitation of polyreactive ligands is unavailable for the mouse IgA dataset; the intermediate processing for that set was not archived.  Nonetheless, the heavy‑chain sequences and binary reactivity flags are provided.  The eLife article’s Data‑availability section states that all data and code are available in the AIMS GitHub repository, confirming that the dataset used in polyreactivity comes directly from the authors’ published repository.

### Points of caution and potential issues

* **Binary mapping for mouse IgA and PLOS‑HIV data.**  Because the raw IgA dataset lacks per‑antigen counts, the script arbitrarily maps “Y” to a count of 7 (implying binding to seven ligands) and “N” to 0.  Similarly, “Y” in the PLOS‑HIV dataset is mapped to 4 (four ligands).  This assumption was made so that “Y” antibodies exceed the >3 threshold, but without the original quantitation this mapping may overstate or understate reactivity.

* **Loss of intermediate data.**  Removing sequences with counts 1–3 simplifies classification but discards potentially informative intermediate cases.  Researchers using this dataset should be aware that the training set reflects only extreme polyreactive and non‑reactive antibodies.

* **Heavy‑chain only and deduplication.**  The dataset includes only VH sequences (light‑chain sequences are left empty), which ignores potential light‑chain contributions to polyreactivity.  Deduplicating identical heavy sequences across families reduces the dataset size from 1 053 to 970 and may alter the distribution of antigen specificities.

In summary, the Boughter dataset in makijling/polyreactivity is faithfully reconstructed from the AIMS data repository and uses the same “parsed” classification threshold described in the eLife paper.  It combines influenza‑reactive, HIV‑reactive and mouse‑IgA antibodies, translates and trims heavy‑chain sequences, and labels them based on the number of antigens bound.  The final training set contains 970 heavy‑chain sequences with roughly equal numbers of polyreactive and non‑polyreactive antibodies, closely matching the proportions reported in the original study, but users should be aware of the assumptions and filtering steps described above.
