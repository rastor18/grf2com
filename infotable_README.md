# infotable.csv

Python-friendly export of `infotable.mat`, a MATLAB `table` (11 rows, one per participant AE01 to AE11, and 9 columns). It was exported from MATLAB with `writetable`, so values keep full double precision (about 15 digits). Read it with:

```python
import pandas as pd
info = pd.read_csv("infotable.csv")   # `subjects` is the participant id; no missing values
```

The table has no units or descriptions stored in it, so the notes below are inferred from the values and from how the analysis code uses them. Check anything you rely on with the lab.

| Column | Type | Notes |
|---|---|---|
| `subjects` | text | Participant id, `AE01` to `AE11` |
| `ages` | number | Years (20 to 34) |
| `sexes` | 0 / 1 | Coding is not recorded in the file (1 for AE01 to AE04, AE06; 0 for AE05, AE07 to AE11) |
| `heights` | number | Likely cm (160 to 180) |
| `shoes` | number | Likely shoe size (6 to 11, with half sizes) |
| `dephys` | number | Likely boot (exoskeleton) size (7 to 11); not documented |
| `feet` | number | Likely foot length in cm (23 to 27) |
| `legs` | number | Leg length, in **cm** (82.5 to 96): `transient_trajectory.m` divides it by 100 for the WBAM normalization |
| `mass` | number | Body mass in **kg** (52.6 to 105.3): `transient_trajectory.m` uses it directly for the WBAM normalization |
