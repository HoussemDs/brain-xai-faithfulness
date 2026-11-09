# Sensitivity-null results (masking halves merged, academic faithfulness re-run)

Sensitivity null (500 contiguous draws, post-hoc). Source: Kaggle kernels houssemeddineds/prenatal-xai-masking-fixes-a (REGION_HALF=0),
houssemeddineds/prenatal-xai-masking-fixes-b (REGION_HALF=1), and the academic-account
houssemeddineed/prenatal-xai-faithfulness-perscan.

- region_masking_merged.csv: both halves concatenated; BH-FDR recomputed per fill over all 48 regions. Authoritative.
- method_vs_masking_correlation_merged.csv: Spearman of each method's region scores vs zero-fill masking effect (48 regions), permutation p (10,000).
- region_masking_revised.png: figure used in the paper.
- method_region_scores.csv: per-region attribution scores (identical in both halves to float precision).
- random_order_baseline.csv: random-order deletion/insertion baseline.
- perscan_academic/: faithfulness per scan, insertion CI by child, paired ReX vs Guided Backprop test.
- halves/: per-half raw p-value tables and the first half's results note.

Regenerate the merged files with:
  python code/merge_masking_halves.py --half-a <dir>/results --half-b <dir>/results --out results_revised_null

Note: the per-half FDR in each Kaggle log is over 24 regions and is superseded by region_masking_merged.csv.
The registered-null results (200 random regions) are in kaggle_output_prenatal_xai_96/results and remain the primary region test; see the paper, Results: Anatomical region test.
