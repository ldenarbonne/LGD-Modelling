"""
Build the LGD modelling sample from the Lending Club dataset
(Kaggle "All Lending Club loan data" – accepted_2007_to_2018Q4.csv, CC0 licence).

Definitions (simplified, undiscounted workout LGD):
  EAD = funded_amnt - total_rec_prncp                     (outstanding principal at charge-off)
  RR  = (recoveries - collection_recovery_fee) / EAD      (recoveries net of collection fees)
  LGD = 1 - RR, bounded to [0, 1]

Scope: charged-off loans (including "Does not meet the credit policy"), originated 2007–2015.
The notebook further restricts the sample to defaults observed 2008–2015 so that workouts can be
considered complete at the extraction date (2018Q4).

Usage: python src/extract_lgd_data.py <path to accepted_2007_to_2018Q4.csv> <output .csv.gz>
"""
import sys
import pandas as pd

SRC, DST = sys.argv[1], sys.argv[2]

COLS = [
    "id", "loan_status", "issue_d", "last_pymnt_d", "funded_amnt", "term",
    "int_rate", "grade", "sub_grade", "emp_length", "home_ownership",
    "annual_inc", "verification_status", "purpose", "dti", "revol_util",
    "open_acc", "pub_rec", "delinq_2yrs", "inq_last_6mths", "fico_range_low",
    "total_rec_prncp", "total_rec_int", "recoveries", "collection_recovery_fee",
    "addr_state", "application_type",
]

chunks = []
for ch in pd.read_csv(SRC, usecols=COLS, chunksize=250_000, low_memory=False):
    chunks.append(ch[ch["loan_status"].astype(str).str.contains("Charged Off", na=False)])
df = pd.concat(chunks, ignore_index=True)
print(f"Charged-off loans (all vintages): {len(df):,}")

df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
df["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
df["vintage"] = df["issue_d"].dt.year
df = df[df["vintage"].between(2007, 2015)].copy()

df["ead"] = df["funded_amnt"] - df["total_rec_prncp"]
df = df[df["ead"] > 50]  # drop negligible residual exposures

df["net_recovery"] = (df["recoveries"] - df["collection_recovery_fee"]).clip(lower=0)
df["rr"] = (df["net_recovery"] / df["ead"]).clip(0, 1)
df["lgd"] = 1 - df["rr"]

# Derived risk factors
df["mob_at_default"] = ((df["last_pymnt_d"] - df["issue_d"]).dt.days / 30.44).round()
df["pct_principal_repaid"] = df["total_rec_prncp"] / df["funded_amnt"]
df["default_year"] = df["last_pymnt_d"].dt.year          # proxy: year of last payment before charge-off
df["term_months"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
df["emp_years"] = (df["emp_length"].astype(str)
                   .str.replace("< 1", "0", regex=False)
                   .str.extract(r"(\d+)").astype(float))

keep = ["id", "vintage", "default_year", "lgd", "rr", "ead", "funded_amnt",
        "term_months", "int_rate", "grade", "sub_grade", "emp_years",
        "home_ownership", "annual_inc", "verification_status", "purpose",
        "dti", "revol_util", "open_acc", "pub_rec", "delinq_2yrs",
        "inq_last_6mths", "fico_range_low", "mob_at_default",
        "pct_principal_repaid", "addr_state", "application_type"]
out = df[keep].dropna(subset=["lgd", "mob_at_default"])
out.to_csv(DST, index=False, compression="gzip")
print(f"Modelling sample: {len(out):,} defaults – mean LGD {out.lgd.mean():.3f}")
