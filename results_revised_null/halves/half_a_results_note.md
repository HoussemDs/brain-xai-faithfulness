# Results note — XAI validation on a pediatric SFCN (public controls)

**0. Headline.** Across 9 methods, sanity-check similarity ranged -0.24–0.73; mean pairwise region-level agreement was 0.36.

**1. Question.** Which explanation methods are faithful, stable across methods, and anatomically valid on a frozen pretrained SFCN?

**1a. Relation to Das et al. (2024).** Extends the explainability step they flagged as *preliminary* (single Grad-CAM, one region tested). Not an audit.

**2. Data.** 96 children, one scan each, public Calgary Preschool controls. **Controls only, no PAE; age group is the proxy task.** Class balance (0=younger,1=older): [48, 48].

**3. Preprocessing.** ANTs N4 + antspynet brain extraction + affine to MNI152 1 mm, cropped to (160, 192, 160) (done beforehand; all 96 passed visual QC).

**4. Model.** Original SFCN (Peng et al. 2021), frozen six-block feature extractor; 512-unit head (2-logit softmax).

**5. Proxy classifier (sanity check, not a result).** acc=0.643±0.120 over 50 fits (10×5-fold stratified-grouped CV), sens=0.632, spec=0.668. Linear-probe acc=0.635, permutation p=nan.

**6. Explanation comparison (faithfulness, deletion AUC lower=better).**
                 method  deletion_auc           ci95  insertion_auc
     M9_rex_exploratory         0.141 [0.119, 0.165]          0.755
M4_integrated_gradients         0.528 [0.475, 0.584]          0.512
     M8_guided_backprop         0.556 [0.489, 0.624]          0.742
          M5_smoothgrad         0.556 [0.502, 0.609]          0.537
            M3_saliency         0.568 [0.518, 0.619]          0.490
             M6_vargrad         0.580 [0.531, 0.628]          0.501
       M1_gradcam_final         0.644 [0.601, 0.687]          0.501
                 M7_lrp         0.674 [0.627, 0.720]          0.598
       M2_gradcam_early         0.777 [0.730, 0.822]          0.488

**7. Sanity check (randomized weights).**
                 method  spearman_trained_vs_random                verdict
       M2_gradcam_early                   -0.241657 PASS (model-dependent)
     M9_rex_exploratory                    0.019175 PASS (model-dependent)
       M1_gradcam_final                    0.120963 PASS (model-dependent)
            M3_saliency                    0.140814 PASS (model-dependent)
             M6_vargrad                    0.153459 PASS (model-dependent)
     M8_guided_backprop                    0.168242 PASS (model-dependent)
          M5_smoothgrad                    0.172064 PASS (model-dependent)
                 M7_lrp                    0.255413 PASS (model-dependent)
M4_integrated_gradients                    0.728186    FAIL (input-driven)

**8. Region test.** 0 region(s) survive BH-FDR q=0.05 across both fills. Full log: results/region_masking_all_pvalues.csv.

**9. Hypotheses.**
- H1 (resolution): most diffuse method = M2_gradcam_early (lowest top-5% mass).
- H2 (region effects): inconclusive/null (0 regions survive correction).
- H3 (agreement): mean pairwise Spearman = 0.36.
- H4 (sanity): worst (most input-driven) = M4_integrated_gradients (r=0.73).
- H5 (layers): {'block2': 0.6242105263157894, 'block6': 0.6352631578947369}.
- H6 (age confound): Ridge R2 = 0.11109150162043036.

**10. Limitations.** n=96, single proxy task, atlas choice, no PAE data, pretrained-weights source not author-confirmed.

**11. What this does and does not show.** Validates an explanation protocol on an age-group proxy in public controls. Says **nothing** about alcohol exposure, FASD, or the paper's clinical regions, which need the private PAE data. The value is the **method verdict**, not the accuracy.

**12. Next steps.** Optional fine-tuning (H7); PAE data access via collaboration.
