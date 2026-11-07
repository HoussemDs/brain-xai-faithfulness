# Preregistered plan: explanation checks for a pediatric SFCN

OSF Preregistration: [jycnr](https://osf.io/jycnr)

Commit this file before the first attribution run. Changes after that go in the decision log (section 8).

## 1. Question

The public Calgary preschool data has controls only, so it cannot test alcohol exposure. This study uses age group (age dichotomized at the cohort median, older vs younger) as a proxy task on the same frozen pretrained SFCN. Age is chosen because the frozen SFCN is a brain-age model, so age is the signal its features actually encode; a sex proxy was tested first and decoded at chance (see decision D11). The questions are:

1. Are the explanations faithful to the model?
2. Do explanation methods agree with each other?
3. Do claims about named regions survive a corrected test?

Out of scope: alcohol exposure, FASD, reproducing the paper's accuracy, and fine-tuning (H7, later).

## 2. Source paper

Das, Duarte, Lebel and Bento (2024), doi:10.3389/fncom.2024.1434421. Checked against the PDF:

- Frozen pretrained SFCN, six convolutional blocks, new head.
- 279 public controls and 122 PAE scans (PAE not public). 5-fold CV, 80/10/10 per fold. The paper does not say whether splits were grouped by child.
- Explanations: Grad-CAM only, described as preliminary.
- One test on explanations: white matter volume, t = -0.73, p = 0.47. No test for the other named regions (corpus callosum, cerebellum, pons).
- Voxel size 0.9 mm isotropic.

This study examines six open questions about the original analysis. Each will be marked confirmed, refuted, or not testable:

1. The 5 x 6 x 5 explanation grid is coarser than the named structures.
2. Only one named region was tested, and not significantly.
3. Possible site or recruitment confound (PAE and control scans came from different studies).
4. Possible age confound.
5. Possible child-level leakage from repeated scans.
6. The sample (279 controls) does not match the labelled public data (section 3).

## 3. Data and cohort

- Source: OSF fork `5kz2p` (CC-BY 4.0), labels from the spreadsheet `All_Datasets` sheet.
- 126 children have sex labels. The T1 sheet lists 279 scans, but these belong to only 96 unique children (repeat visits); those 96 are exactly the labelled children who have any T1 scan. The manifest has all 96, one scan per child.
- The cohort is loaded in two passes because the spreadsheets hyphenate some IDs differently: the labels sheet writes `PS1064-1`, the T1 sheet writes `PS1064-10-1`, and the file is `PS1064101.nii.gz`. The first pass matches the IDs that line up directly; the second pass (`code/fetch_extra_scans.py`) joins on the T1-sheet ID with hyphens removed and recovers the remaining 11 children (decision D1).
- Analysis cohort: one labelled scan per child, the 96 in the manifest (decision D1).
- Ages and sex, current manifest (n = 96): age 3.84 years (SD 0.83, range 2.58 to 6.90); 50 male, 46 female. Proxy task is age dichotomized at the cohort median (3.64 years): 48 older, 48 younger.

## 4. Model and preprocessing

**Model.** Pretrained SFCN (Peng et al. 2021 architecture), six convolutional blocks frozen, output feature map 64 x 5 x 6 x 5 (9600 values).

**Head.** An age-group classifier on the frozen features (older vs younger, age split at the cohort median): flatten, 512-unit dense, ReLU, dropout 0.5, linear layer with 2 outputs, Adam, learning rate 1e-3, 100 epochs. The paper's head was trained on alcohol-exposure labels, which are not public, so it cannot be reproduced; this head replaces it (decision D12).

**Preprocessing (complete).** ANTs N4 bias correction, antspynet brain extraction, affine registration to MNI152 (1 mm), center crop to 160 x 192 x 160, normalization by the mean of positive voxels. The paper used FSL-BET + SynthStrip for stripping (D3) and registered to a pediatric atlas; this study registers to MNI152 to match the frozen SFCN's pretraining space (D4). N4 bias correction is used in both.

Status: complete for all 96 children. Preprocessing ran on Kaggle in two batches, matching the two loader passes (section 3). All 96 volumes are finite and 160 x 192 x 160, all 96 feature vectors are 9600 values and finite, and labels match the spreadsheet. Three scans needed a second preprocessing pass and now pass visual QC: PS14_086 (first output was noise, now replaced), PS0536101 (raw file truncated, re-downloaded from OSF) and PS1432101 (first output streaked). Visual QC of all 96 mid-slices was completed, so all 96 scans are in the analysis.

## 5. Hypotheses

Write down before running. Mark each supported, refuted, or inconclusive.

**H1 (resolution).** Grad-CAM (M1) is smoother and coarser than Integrated Gradients (M4) and SmoothGrad (M5).
Measure: fraction of attribution in the top 5% of voxels; blob count.

**H2 (region effects).** Only large regions survive masking and correction. Small named structures (pons, corpus callosum) do not.
Measure: corrected p-values from the region masking test.

**H3 (method agreement).** M4 and M5 agree with each other more than either agrees with M1.
Measure: pairwise Spearman correlation of region-level scores.

**H4 (sanity).** For at least one method, randomized-weight maps differ from trained-weight maps. The test decides which methods pass.
Measure: Spearman correlation, trained vs randomized.

**H5 (layers).** A linear probe on block 6 predicts age group better than a probe on block 2.
Measure: cross-validated probe accuracy per block.

**H6 (signal gate).** Age-group accuracy beats chance under a child-level label-permutation test; near chance means the explanations are not interpretable.
Measure: cross-validated probe accuracy and its permutation p-value (older-vs-younger, median split).

**H7 (future).** Fine-tuning the last block changes deeper layers more than early ones. Measure: linear CKA per block. Not in this analysis.

## 6. Methods

Same test scans and trained SFCN for all. IDs match `code/run_all_methods.py`.

| ID | Method | Implementation |
|---|---|---|
| M1 | Grad-CAM, final conv block (paper's method) | captum `LayerGradCam` |
| M2 | Grad-CAM, earlier block | captum `LayerGradCam` |
| M3 | Saliency | captum `Saliency` |
| M4 | Integrated Gradients, 32 steps | captum `IntegratedGradients` |
| M5 | SmoothGrad, 20 samples, sigma 0.1 | captum `NoiseTunnel` (smoothgrad) |
| M6 | VarGrad | captum `NoiseTunnel` (vargrad) |
| M7 | LRP, EpsilonPlusFlat rule | zennit |
| M8 | Guided Backprop (negative control) | captum |
| M9 | ReX, 3D occlusion responsibility on the 5 x 6 x 5 grid | `code/rex_3d.py` |


## 7. Metrics and statistics

- **Faithfulness:** deletion and insertion AUC (top 5, 10, 20, 30% of voxels).
- **Sanity:** Spearman correlation, trained vs randomized weights, per method and scan.
- **Resolution:** fraction of attribution in the top 5%; blob count.
- **Agreement:** pairwise Spearman correlation of region scores.
- **Region test:** Harvard-Oxford regions (decision D5); mean-intensity masking with zero as a check; 200 random same-size regions as the null.
- **Uncertainty:** 95% bootstrap CIs (10,000 resamples), resampled by child.
- **P-values:** permutation tests, 10,000 permutations.
- **Multiple comparisons:** Benjamini-Hochberg, q = 0.05, across all method-by-region tests. Report raw and corrected p-values, including non-significant ones.
- **Classifier:** 5-fold stratified, grouped by child, repeated 10 times. Permutation test (1,000) against chance.
- **Power:** all 96 scans are used. At n = 96 the study is adequately powered for medium-to-large region effects; small effects may be inconclusive. H3 and H4 do not depend on effect size.
- Log every test (method, region, raw p, corrected p, n). Tests run after seeing data are exploratory.

## 8. Stop rules and decisions

**Stop and report if:**
- The classifier is at chance after the preprocessing is checked.
- The randomized-weight check fails for every method. This suggests a bug, not a finding.

**Decision log**

| ID | Decision | Status |
|---|---|---|
| D1 | Cohort: all 96 children on the fork with a demographic label (age, sex) and a usable T1, one scan each. Loaded in two passes (section 3); the second pass (`code/fetch_extra_scans.py`) recovers the 11 children whose T1-sheet IDs use different hyphenation. Cohort finalized at 96 after data QC and before any attribution or classifier run | Decided: 96 |
| D2 | Data source: OSF fork `5kz2p` (`axz5r` is empty) | Done |
| D3 | Skull stripping: antspynet brain extraction instead of the paper's FSL-BET + SynthStrip. antspynet is a modern deep-learning extractor of comparable robustness, kept in one Python toolchain (BET alone was a concern on pediatric heads). Bias correction is N4 in both (paper: N4 via SimpleITK; here: N4 via ANTs), so N4 is not a deviation. QC'd on all 96 scans; deviation stated in the write-up | Done |
| D4 | Registration: affine (linear, FLIRT-equivalent) to 1 mm MNI152, crop to central 160 x 192 x 160. The paper registered to a pediatric atlas; MNI152 is chosen instead because the frozen SFCN was pretrained on UK Biobank T1 linearly registered to 1 mm MNI152 and cropped to 160 x 192 x 160 (Peng/Gong et al. 2021), so inputs stay in the model's native space and the frozen features remain valid. Trade-off: affine of preschool brains to an adult template adds some warping (see the bimodal mask fraction in results/preprocessing.md) | Done |
| D5 | Region atlas: Harvard-Oxford cortical (MNI152 1 mm, already in prep_output/nilearn_data), cropped to 160 x 192 x 160 to match the volumes. Subcortical and cerebellar Harvard-Oxford atlases are added at the region test for pons/brainstem and cerebellum | Decided |
| D6 | Masking: mean intensity, zero as a check | Done |
| D7 | LRP rule: EpsilonPlusFlat | Done |
| D8 | Split unit: child, repeated grouped 5-fold | Done |
| D9 | Attribution caps: IG 32 steps, SmoothGrad 20 samples | Done |
| D10 | ReX is method M9 | Decided |
| D11 | Proxy task: age group (age dichotomized at the cohort median, older=1). Sex was the original proxy, but the frozen brain-age SFCN features decode sex at chance (child-grouped CV acc approx 0.50, permutation p approx 0.33), so explanations of a sex head would be uninterpretable. Age is the signal the brain-age model encodes and is used instead. PAE excluded | Changed |
| D12 | Head: age-group classifier (2 outputs), replacing the paper's alcohol-exposure head, which cannot be reproduced without the labels | Decided |

Open items before the first attribution run: none for the data.