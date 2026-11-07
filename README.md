# Explanation checks for a pediatric SFCN on public Calgary preschool MRI

A public-data follow-up to Das, Duarte, Lebel and Bento (2024), *Deep learning for detecting prenatal alcohol exposure in pediatric brain MRI* (Front. Comput. Neurosci. 18:1434421, doi:10.3389/fncom.2024.1434421).

The public data has controls only, so this project does not detect alcohol exposure. It uses age group (age dichotomized at the cohort median, older vs younger) as a proxy task on a frozen pretrained SFCN to test whether common explanation methods are faithful, agree with each other, and give region-level claims that survive correction. Age is used because the frozen SFCN is a brain-age model, so age is the signal its features encode (a sex proxy was tested first and decoded at chance; see decision D11 in the plan). The protocol is in [prenatal_xai_plan.md](docs/prenatal_xai_plan.md).

## Status

- **Full run complete.** Classifier and all nine attribution methods ran on Kaggle (GPU T4 x2) over the full cohort of 96 children. Results and figures: [REPORT.md](results/REPORT.md).
- Headline: occlusion-based ReX (M9) is the most faithful method and early-block Grad-CAM (M2) the least; eight of nine methods pass the randomized-weight sanity check but Integrated Gradients (M4) fails it (input-driven, r = 0.73); 15 cortical regions survive BH-FDR q = 0.05; proxy-task accuracy 0.64. The deliverable is the method verdict, not the accuracy.
- Cohort: 96 children, one scan each (decision D1). All 96 volumes are preprocessed (160 x 192 x 160, finite) and all 96 feature vectors (96 x 9600) are in `prep_output/`.
- All 96 mid-sagittal slices were checked by eye (`results/figures/05_qc_contact_sheet.png`) and passed visual quality control. Every scan in the cohort is included in the analysis.
- Weights: `weights/sfcn_brainage.p` is the UK Biobank brain-age SFCN from github.com/ha-ha-ha-han/UKBiobank_deep_pretrain. It is gitignored.

## Data

- Source: OSF fork `5kz2p`, CC-BY 4.0. The original project `axz5r` is public but its storage is empty.
- Layout: `T1_Dataset/<PreschoolID>/<session>/<scan>.nii.gz`. Labels (age, sex) are in sheet `All_Datasets` of `Calgary_Preschool_Dataset_Updated_*.xlsx`. Sex is coded Female = 0, Male = 1.
- Current cohort: 96 unique children, one scan each (50 male, 46 female; age 3.84 +/- 0.83 years, range 2.58 to 6.90). See `data/manifest.csv` and `results/table1_cohort.csv`.

## Cohort assembly and deviations from the paper

1. **The cohort is assembled in two loader passes.** The spreadsheets hyphenate some scan IDs differently: the labels sheet writes `PS1064-1`, the T1 sheet writes `PS1064-10-1`, and the file on disk is `PS1064101.nii.gz`. The first pass (`code/fetch_osf_scans.py`) matches the IDs that line up directly; the second pass (`code/fetch_extra_scans.py`) joins on the T1-sheet ID with hyphens removed and recovers the remaining 11 children. Together they give the full cohort of 96: every child on the fork with a sex label and a usable T1.
2. **Sample size (reconciled).** The T1 sheet lists 279 scans, but these belong to only 96 unique children (repeat/longitudinal visits). Of the 126 children with a sex label, exactly those 96 have at least one T1 scan. This study uses one scan per child, so the cohort is 96: every labelled child who has a T1 scan. The paper's "279 controls" counts scans, not children; using one scan per child avoids child-level leakage from repeated scans (see the explanation-grid and leakage questions in the plan).
3. **Preprocessing differs from the paper, by design.** Skull stripping uses antspynet (a modern deep-learning extractor) instead of the paper's FSL-BET + SynthStrip; both are comparable, and this keeps one Python toolchain. Bias correction is N4 in both. The meaningful difference is the registration target: the paper registered to a pediatric atlas, while this study registers (affine) to 1 mm MNI152 and crops to 160 x 192 x 160, the space the frozen SFCN was pretrained in (UK Biobank, Peng/Gong et al. 2021), so the transfer-learned features stay valid. The cost is more preschool-to-adult warping, visible as the bimodal mask fraction in `results/preprocessing.md`.
4. **Kaggle paths.** The notebook reads `/kaggle/input` and writes `/kaggle/working`. Local runs use `./work`. The Kaggle input dataset must contain all 96 scans and `manifest.csv`.
5. **Head.** The planned head is an age-group classifier (older vs younger, age split at the cohort median) with 2 outputs. The paper's head used alcohol-exposure labels, which are not public. Sex was the original proxy but decoded at chance from the frozen brain-age features, so age is used instead (plan decision D11).

## Repository layout

| Path | Contents |
|---|---|
| `docs/prenatal_xai_plan.md` | Pre-specified protocol and decision log |
| `notebooks/prenatal_xai_prep.ipynb` | Preprocessing notebook. Download, preprocessing, frozen SFCN feature extraction. |
| `code/fetch_osf_scans.py` | First loader pass. Fetches the labelled T1 scans whose IDs match directly and writes the manifest |
| `code/fetch_extra_scans.py` | Second loader pass. Recovers the 11 children whose T1-sheet IDs use different hyphenation, completing the cohort at 96 |
| `code/make_prep_figures.py` | Preprocessing QC figures into `results/figures/` |
| `code/sfcn.py` | SFCN architecture and feature-extractor loader (used by the attribution code) |
| `code/run_all_methods.py` | Attribution methods M1 to M9 |
| `code/rex_3d.py` | M9, 3D ReX occlusion responsibility on the 5 x 6 x 5 grid |
| `code/eval_metrics.py` | Faithfulness, sanity check, bootstrap CIs, permutation tests, Benjamini-Hochberg |
| `data/manifest.csv` | Scan manifest (metadata only) |
| `results/` | Cohort table, preprocessing report, QC figures |
| `weights/` | SFCN checkpoint (gitignored) |

Not committed (see `.gitignore`): `full_ds/` (raw scans and spreadsheet), `upload/`, `prep_output/`, `weights/`, and archives.

## Reproducing the preprocessing

1. Download the labelled scans: `python code/fetch_osf_scans.py ./full_ds`
2. Second pass to complete the cohort: `python code/fetch_extra_scans.py` (run from the project root)
3. Run `notebooks/prenatal_xai_prep.ipynb` on Kaggle with all 96 scans attached as a dataset, or locally in `./work`. Check the printed line `preprocessed N/M | failures: K`.
4. Copy `prep_output/` back to this project.

Dependencies are not pinned. The notebook installs `torch`, `nibabel`, `scikit-learn`, `scipy`, `antspyx`, `antspynet` and `nilearn` if they are missing. The attribution code also needs `captum` and `zennit`.

## Citation

Das A, Duarte K, Lebel C, Bento M (2024). Deep learning for detecting prenatal alcohol exposure in pediatric brain MRI. *Frontiers in Computational Neuroscience* 18:1434421. doi:10.3389/fncom.2024.1434421

Dataset: Calgary Preschool MRI Dataset, OSF project `axz5r` (fork `5kz2p`), CC-BY 4.0.

## License

Data: CC-BY 4.0 (OSF fork). Code: not yet chosen.
