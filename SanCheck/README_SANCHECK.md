# sancheck

## What is this?
`sancheck` is a minimal-tuning CLI tool for quickly assessing the statistical cleanliness of numeric columns in CSV datasets.  
It provides a fast, high-level overview before deeper analysis or modeling.

## When should I use it?
- Before exploratory data analysis (EDA)
- Before training statistical or machine learning models
- When you want a quick sanity check without manual inspection

## What it does NOT do
- It does not clean or modify data
- It does not model relationships
- It does not replace proper EDA or data validation pipelines

## Quick start
Run the tool on a CSV file:

```bash
sancheck [csv_data] [n_feature_per_plot or 'all']
```

## Example output
 Summary of columns
- Valid numeric columns: 9
- Ignored non-numeric columns: 7

📌 Column problems
- Column with NaN/Inf/invalid: 0
  - Age: invalid=0/5000 (0.000)
  - Class: invalid=0/5000 (0.000)
  - Study_Hours_Per_Day: invalid=0/5000 (0.000)
  - Attendance_Percentage: invalid=0/5000 (0.000)
  - Math_Score: invalid=0/5000 (0.000)
  - Science_Score: invalid=0/5000 (0.000)
  - English_Score: invalid=0/5000 (0.000)
  - Previous_Year_Score: invalid=0/5000 (0.000)
  - Final_Percentage: invalid=0/5000 (0.000)
- Type inconsistency column: 0
  - Age: bad_type=0 (0.000)
  - Class: bad_type=0 (0.000)
  - Study_Hours_Per_Day: bad_type=0 (0.000)
  - Attendance_Percentage: bad_type=0 (0.000)
  - Math_Score: bad_type=0 (0.000)
  - Science_Score: bad_type=0 (0.000)
  - English_Score: bad_type=0 (0.000)
  - Previous_Year_Score: bad_type=0 (0.000)
  - Final_Percentage: bad_type=0 (0.000)
- Similar feature pairs (|corr| >= 0.95):
  - Severity similarity: 0.000
  - English_Score <-> Final_Percentage: |corr|=0.592
  - Science_Score <-> Final_Percentage: |corr|=0.572
  - Math_Score <-> Final_Percentage: |corr|=0.564
  - Study_Hours_Per_Day <-> Science_Score: |corr|=0.038
  - Class <-> Attendance_Percentage: |corr|=0.035
  - Study_Hours_Per_Day <-> Attendance_Percentage: |corr|=0.027
  - Class <-> Math_Score: |corr|=0.021
  - Math_Score <-> Science_Score: |corr|=0.020
  - Study_Hours_Per_Day <-> Previous_Year_Score: |corr|=0.020
  - Age <-> Attendance_Percentage: |corr|=0.019

📌 Row problems
- Problematic rows (NaN/Inf): 0/5000
- Severity row: 0.132
  - row 4867: score=0.625, invalid=False
  - row 82: score=0.624, invalid=False
  - row 1364: score=0.622, invalid=False
  - row 1482: score=0.619, invalid=False
  - row 4913: score=0.611, invalid=False

📌 Distribution / interpretation
- High entropy means the distribution is more even/complex; it's not automatically 'noise', it can also be multimodal.
- High spread score means the data is more dispersed robustly compared to its central tendency.

  Top entropy:
  - Attendance_Percentage: entropy=1.000 (very spread / more uniform or complex distribution)
  - Science_Score: entropy=0.998 (very spread / more uniform or complex distribution)
  - English_Score: entropy=0.998 (very spread / more uniform or complex distribution)
  - Math_Score: entropy=0.997 (very spread / more uniform or complex distribution)
  - Study_Hours_Per_Day: entropy=0.997 (very spread / more uniform or complex distribution)

  Top spread:
  - Class: spread_score=0.690 (wide / large variation), var=1.225, iqr=1.000
  - Final_Percentage: spread_score=0.471 (moderate / moderate variation), var=120.211, iqr=15.660
  - Math_Score: spread_score=0.384 (moderate / moderate variation), var=350.606, iqr=32.000
  - Science_Score: spread_score=0.380 (moderate / moderate variation), var=366.385, iqr=33.000
  - Previous_Year_Score: spread_score=0.377 (moderate / moderate variation), var=261.065, iqr=28.000

📌 Normality (per fitur)
- Age: Shapiro=0.000 | KS=0.000
- Class: Shapiro=0.000 | KS=0.000
- Study_Hours_Per_Day: Shapiro=0.000 | KS=0.000
- Attendance_Percentage: Shapiro=0.000 | KS=0.000
- Math_Score: Shapiro=0.000 | KS=0.000
- Science_Score: Shapiro=0.000 | KS=0.000
- English_Score: Shapiro=0.000 | KS=0.000
- Previous_Year_Score: Shapiro=0.000 | KS=0.000
- Final_Percentage: Shapiro=0.000 | KS=0.038

🔨 Final status
- clarity score: 0.961 / 1.000
- clarity label: very clean
- missing severity: 0.000
- type severity: 0.000
- similarity severity: 0.000
- row severity: 0.132

📊 Dataset-level distribution summary
- avg entropy: 0.979
- avg spread score: 0.420

## Interpretation tips
- Higher clarity scores indicate cleaner numeric data

- Anomalous rows are ranked, not classified — use them for inspection

- Non-numeric columns are ignored by design

- This tool is best used as a fast pre-analysis step