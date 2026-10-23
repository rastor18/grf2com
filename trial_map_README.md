# AdaptExo trial map — read this before using `trial_map.csv`

`trial_map.csv` is a tidy, one-row-per-trial table (11 participants x 8 trials = 88 rows) saying which
exoskeleton controller each trial was, plus known data-quality complications.
Load it with `pandas.read_csv("trial_map.csv")`. Blank cells are empty strings (use `keep_default_na=False` if you want that).

## Condition codes (authoritative, confirmed by the experimenter)

| controller_code | controller_name | Notes |
|---|---|---|
| 1 | OFF | "Zero torque" (written `ZT` in the complication notes). Config: FOURPOINTSPLINE, WALKING, max PEAK_TORQUE = 3 |
| 2 | PBC | Position/4-point-spline controller (written `4PTS` in the complication notes). Config: FOURPOINTSPLINE, WALKING, PEAK_TORQUE 11-13 |
| 3 | VNMC | VIRTUALNEUROMUSCULARCONTROLLER |
| 4 | MLE | Config: FOURPOINTSPLINE with TASK = WALKJOGDLGAITPHASE |

Name aliases used in the lab's notes and filenames: `ZT` = `OFF` (met files say `OFF`); `4PTS` = `PBC`.
The classification rule is the same as `get_controller.m` in the AdaptExo folder.

## Experimental design

- Trials 1-4 = block 1; trials 5-8 = block 2 (`block` column; `position_in_block` = 1-4).
- Reverse counterbalance: if block 1 is A B C D, block 2 is D C B A. So trial 5 = trial 4's condition, trial 6 = trial 3's, trial 7 = trial 2's, trial 8 = trial 1's.
- `t_order.mat` (in `../AdaptExo/`) holds variable `permutations`, an 11x4 matrix: row = participant (AE01..AE11 in order), column = trials 1-4, value = controller_code. Trials 5-8 are not stored there; they follow from reversal.

## How the labels were derived (provenance)

Each trial's controller was read from the first-row/all-row settings of its `*_CONFIG.csv`
(`STANCE_CONTROL_STYLE`, `TASK`, `PEAK_TORQUE`). It was NOT taken from the metabolics filenames, because
AE01's trial folders contain two `*_met.xlsx` files each with conflicting condition names (e.g. T1 has both `T01_OFF_met.xlsx` and `T1_VNMC_met.xlsx`). The config is authoritative; for AE01 the config agrees with `t_order.mat`.

`controller_label_source` says where each row's label came from:
- `CONFIG_file` (84 rows): read from the trial's CONFIG csv.
- `INFERRED_from_t_order_reverse_counterbalance` (4 rows): no CONFIG file exists, so the label comes from `t_order.mat` plus reversal. Rows: **AE05 T6, T7, T8** and **AE10 T4**. These are corroborated by the complication notes (AE05 T6 = VNMC, T7 = ZT, T8 = MLE; AE10 T4 = MLE) and, for AE05 T6/T7, by the metabolics filenames (`T6_VNMC_met`, `T7_OFF_met`).

## Comparison with t_order.mat

- Trials 1-4: **all 11 participants x 4 trials match `t_order.mat` exactly** (44/44; includes AE10 T4 where the config is missing and the label is inferred).
- Trials 5-8: the config-derived labels equal the reverse of trials 1-4 for every trial that has a config file (41 of 44; AE05 T6-8 have no config and are inferred this way).
- So `matches_t_order` = 1 on all 88 rows. Columns `t_order_expected_code` and `matches_t_order` let you re-verify. The CONFIG file and `t_order.mat` agree, with no discrepancies.

## Complication columns

`has_complication` = 1 on 12 rows. Verbatim text from the experimenter is preserved:
- `complication_label_as_written`: condition name as written in the note (every one agrees with `controller_name` after alias mapping).
- `complication_text`: what went wrong. `consequence_text`: the experimenter's planned handling (verbatim).
- Structured interpretation of the consequence, one column per data stream: `SS_metabolics` (steady-state), `SC_metabolics` (speed-change), `adaptability`, `survey`.
  Values: `ok` (use normally), `skip` (exclude), `partial` (some of the data missing/skip part; see `data_note` and `consequence_text`), `ok_v1` / `ok_v2` (use the v1 or v2 version of the data file). All values are definitive.
  Rows with no complication are `ok` for every stream.
- `data_note`: extra clarification, e.g. which versioned file to use.

**Interpretation caveat:** the structured columns are my reading of free-text notes. Where they conflict, `consequence_text` wins. The experimenter confirmed the originally tentative notes (AE05 T6/T7, AE09 T3), so no value is marked uncertain.

### Complication list (quick view)

| Participant | Trial | Condition | Summary of handling |
|---|---|---|---|
| AE02 | 3 | MLE | skip adaptability, SC metabolics, survey |
| AE05 | 6 | VNMC | only SS metabolics usable |
| AE05 | 7 | OFF (ZT) | only SS metabolics usable |
| AE05 | 8 | MLE | skip entirely (no data files) |
| AE06 | 2 | PBC (4PTS) | missing speed changes (adaptability partial) |
| AE06 | 3 | VNMC | skip metabolics (suspected air leak) |
| AE06 | 6 | VNMC | missing some speed changes (adaptability partial) |
| AE06 | 8 | MLE | skip metabolics and right side |
| AE07 | 3 | PBC (4PTS) | v1 for SS, v2 for SC |
| AE09 | 3 | MLE | skip SC metabolics and left-foot adaptability |
| AE09 | 6 | MLE | v1 for SS, v2 for SC and adaptability; nothing skipped |
| AE10 | 4 | MLE | skip entirely (no data files) |

## All columns

`participant` (AE01..AE11), `participant_index` (1-11, the row in t_order.mat), `trial` (1-8), `block` (1/2), `position_in_block` (1-4),
`controller_code` (1-4), `controller_name`, `controller_label_source`, `t_order_expected_code`, `matches_t_order`,
`config_file` (filename within `AdaptExo_Copy/<participant>/T<trial>/`, blank if missing), `config_detail`
(`STANCE_CONTROL_STYLE|TASK|max_PEAK_TORQUE`), `has_complication`, `complication_label_as_written`, `complication_text`,
`consequence_text`, `SS_metabolics`, `SC_metabolics`, `adaptability`, `survey`, `data_note`.

## Other gotchas

- AE10's config files carry the REMOTE_FILENAME `AE010_T..` (extra zero); the folder is `AE10`.
- Folder `AdaptExo_Copy` and `AdaptExo` give identical classifications; `t_order.mat` lives only in `AdaptExo`.
- Block 2 of AE05 (T6, T7, T8) and AE10 T4 have no CONFIG/LEFT/RIGHT csv; those trials have only (or no) metabolics files.

## Familiarization trials (`F*` folders) — `familiarization_map.csv`

Separate file (68 rows) so `trial_map.csv` stays a clean 88-row T-trial table. Same code key as above, labels read from each F trial's CONFIG csv (all `CONFIG_file`; none inferred). Columns: `participant`, `participant_index`, `familiarization_trial` (the F number), `block`, `position_in_block`, `controller_code`, `controller_name`, `controller_label_source`, `config_file`, `config_detail`.

- Day-1 familiarization was **3 conditions (PBC, VNMC, MLE; no OFF), not 4**. `f_rand.m` randomizes that 3-condition order.
- F1-F3 = block 1, F4-F6 = block 2, and block 2 is the exact reverse of block 1 for **all 11 participants** (F4=F3, F5=F2, F6=F1), the same reverse-counterbalance rule as the T trials. Each condition therefore appears twice per participant.
- `F0` is a separate OFF (zero torque, max PEAK_TORQUE = 3) warm-up that exists only for **AE01 and AE02** (`block` = `0_zero_torque_warmup`). AE03-AE11 have no F0 folder.
- AE02 also has an earlier OFF config in `AE02/misc/` (timestamp 1300 vs the F0 folder's 1302). It is not in the CSV.
- The F trials are NOT in `t_order.mat` (it only encodes the T trials), so there is no stored reference to check the F order against. The only check done is the reversal above.
- No complications or per-stream exclusion notes were provided for F trials, and F folders contain no metabolics files, so there are no flag columns.
- Config REMOTE_FILENAME strings in F folders are inconsistent (e.g. AE01's F trials say `AEF01_T02..`, and AE02's say `AEF00_T..`). Ignore them; use the folder name.
- Trial order within the F folders reflects folder number, which matched file timestamps in the cases checked, but timestamps were not systematically verified.
