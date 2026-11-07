#!/usr/bin/env python3
"""Download all sex-labeled Calgary Preschool T1 scans from OSF fork 5kz2p,
build manifest.csv. Runs locally (OSF reachable here, unlike Kaggle IPs)."""
import json, re, sys, time, urllib.request
from pathlib import Path
import pandas as pd

OSF_API = "https://api.osf.io/v2"
NODE = "5kz2p"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./full_ds")
RAW = OUT / "scans"; RAW.mkdir(parents=True, exist_ok=True)

def g(url):
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/vnd.api+json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))

def root(node):
    out, url = [], f"{OSF_API}/nodes/{node}/files/osfstorage/"
    while url:
        p = g(url); out += p["data"]; url = p["links"].get("next")
    return out

def walk(href, parents=()):
    url = href
    while url:
        p = g(url)
        for it in p["data"]:
            a = it["attributes"]
            if a["kind"] == "folder":
                sub = it["relationships"]["files"]["links"]["related"]["href"]
                yield from walk(sub, parents + (a["name"],))
            else:
                yield parents, a["name"], it["links"]["download"]
        url = p["links"].get("next")

r = root(NODE)

# newest dated demographics xlsx
cands = [it for it in r if it["attributes"]["name"].startswith("Calgary_Preschool_Dataset")
         and it["attributes"]["name"].endswith(".xlsx")]
def _d(it):
    m = re.search(r"(\d{8})", it["attributes"]["name"]); return int(m.group(1)) if m else -1
chosen = max(cands, key=_d)
demo_path = OUT / "demographics.xlsx"
if not demo_path.exists():
    urllib.request.urlretrieve(chosen["links"]["download"], demo_path)
print("demographics:", chosen["attributes"]["name"], flush=True)

# read labels (sheet All_Datasets per project notes, else first)
xl = pd.ExcelFile(demo_path)
sheet = "All_Datasets" if "All_Datasets" in xl.sheet_names else 0
demo = xl.parse(sheet).rename(columns={
    "PreschoolID": "child_id", "ScanID": "scan_id",
    "Age (Years)": "age", "Biological Sex (Female = 0; Male = 1)": "sex"})
demo["scan_id"] = demo["scan_id"].astype(str).str.strip()
labeled = demo.dropna(subset=["sex"]).set_index("scan_id")[["child_id", "age", "sex"]]
label_ids = set(labeled.index)
print(f"labeled scan_ids in xlsx: {len(label_ids)}", flush=True)

# walk T1_Dataset, download labeled scans
t1 = [it for it in r if it["attributes"]["kind"] == "folder"
      and it["attributes"]["name"] == "T1_Dataset"][0]
href = t1["relationships"]["files"]["links"]["related"]["href"]

def _scalar(v):
    return v.iloc[0] if hasattr(v, "iloc") else v

# Pass 1: walk the tree (no downloads) and collect labeled T1 candidates per child.
cand = {}  # child_id -> list of (scan_id, download_url)
skip = 0
for parents, name, dl in walk(href):
    if not name.lower().endswith((".nii", ".nii.gz")):
        continue
    scan = name.replace(".nii.gz", "").replace(".nii", "").strip()
    if scan not in label_ids:
        skip += 1
        continue
    child = str(_scalar(labeled.loc[scan, "child_id"]))
    cand.setdefault(child, []).append((scan, dl))
print(f"labeled T1 scans found: {sum(len(v) for v in cand.values())} "
      f"across {len(cand)} children (skipped {skip} unlabeled)", flush=True)

# Pass 2: one scan per child (smallest scan_id = deterministic), download just those.
rows, n = [], 0
for child in sorted(cand):
    scan, dl = sorted(cand[child])[0]
    out = RAW / f"{scan}.nii.gz"
    if not out.exists() or out.stat().st_size == 0:
        for attempt in range(6):  # OSF intermittently returns 403 under load
            try:
                urllib.request.urlretrieve(dl, out)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"  retry {scan} ({e}) in {30 * (attempt + 1)}s", flush=True)
                time.sleep(30 * (attempt + 1))
    rows.append(dict(child_id=child, scan_id=scan,
                     age=float(_scalar(labeled.loc[scan, "age"])),
                     sex=int(_scalar(labeled.loc[scan, "sex"])),
                     path=f"{scan}.nii.gz"))
    n += 1
    if n % 10 == 0:
        print(f"  downloaded {n}/{len(cand)} scans...", flush=True)

man = pd.DataFrame(rows).drop_duplicates(subset=["scan_id"]).reset_index(drop=True)
man.to_csv(OUT / "manifest.csv", index=False)
print(f"DONE: {len(man)} labeled scans | {man['child_id'].nunique()} children | "
      f"skipped {skip} unlabeled | sex balance {man['sex'].value_counts().to_dict()}", flush=True)
