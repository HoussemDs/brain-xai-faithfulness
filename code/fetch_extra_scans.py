#!/usr/bin/env python3
"""Add the children that fetch_osf_scans.py skipped.

Their scan IDs differ between sheets: the labels sheet (All_Datasets) writes
e.g. PS1064-1, the T1 sheet writes PS1064-10-1, and the OSF file is
PS1064101.nii.gz. The file name is the T1-sheet ID with hyphens removed.
This script takes the T1-sheet scan for each child, checks it exists on OSF,
takes age and sex from the All_Datasets row for the same session, downloads
the file to full_ds/scans/, and appends the row to data/manifest.csv.

Usage (from project root):  python code/fetch_extra_scans.py
"""
import csv
import json
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "full_ds" / "demographics.xlsx"
MANIFEST = ROOT / "data" / "manifest.csv"
SCANS = ROOT / "full_ds" / "scans"
NODE = "5kz2p"
API = "https://api.osf.io/v2"
EXTRA_CHILDREN = ["10140", "10141", "10142", "10143", "10146", "10147",
                  "10149", "10153", "10155", "10156", "10160"]
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_sheet(z, shared, path):
    rows = []
    for r in ET.fromstring(z.read(path)).iter(NS + "row"):
        vals = []
        for c in r.findall(NS + "c"):
            v = c.find(NS + "v")
            vals.append("" if v is None else (shared[int(v.text)] if c.get("t") == "s" else v.text))
        rows.append(vals)
    return [r for r in rows[1:] if r]


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.api+json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def osf_children():
    root = get(f"{API}/nodes/{NODE}/files/osfstorage/")
    t1 = next(f for f in root["data"] if f["attributes"]["name"] == "T1_Dataset")
    url = t1["relationships"]["files"]["links"]["related"]["href"]
    kids = {}
    while url:
        page = get(url)
        for f in page["data"]:
            if f["attributes"]["kind"] == "folder":
                kids[f["attributes"]["name"]] = f["relationships"]["files"]["links"]["related"]["href"]
        url = page["links"].get("next")
    return kids


def osf_files_under(href):
    out, url = [], href
    while url:
        page = get(url)
        out += page["data"]
        url = page["links"].get("next")
    return out


def main():
    z = zipfile.ZipFile(WORKBOOK)
    shared = [
        "".join(t.text or "" for t in si.iter(NS + "t"))
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS + "si")
    ]
    labels = read_sheet(z, shared, "xl/worksheets/sheet1.xml")  # All_Datasets
    t1_sheet = read_sheet(z, shared, "xl/worksheets/sheet2.xml")  # T1
    kids = osf_children()
    SCANS.mkdir(parents=True, exist_ok=True)
    new_rows = []
    for child in EXTRA_CHILDREN:
        # Some children also have a second-session ID (e.g. PS17_047); the
        # file-name style ID is the one with hyphens (e.g. PS1183-10-1).
        t1_ids = [r[1] for r in t1_sheet if r[0] == child and "-" in r[1]]
        if len(t1_ids) != 1:
            raise SystemExit(f"{child}: expected one T1 scan in sheet, found {t1_ids}")
        t1_id = t1_ids[0]
        fname = t1_id.replace("-", "") + ".nii.gz"
        base = t1_id.split("-")[0]  # e.g. PS1064
        lab = [r for r in labels if r[0] == child and r[1].startswith(base)]
        if len(lab) != 1:
            raise SystemExit(f"{child}: expected one labelled row for {base}, found {[r[1] for r in lab]}")
        age, sex = float(lab[0][2]), int(lab[0][3])
        files = []
        for session in osf_files_under(kids[child]):
            if session["attributes"]["kind"] == "folder":
                files += osf_files_under(session["relationships"]["files"]["links"]["related"]["href"])
        match = [f for f in files if f["attributes"]["name"] == fname]
        if not match:
            raise SystemExit(f"{child}: {fname} not found on OSF")
        out = SCANS / fname
        if not out.exists() or out.stat().st_size == 0:
            urllib.request.urlretrieve(match[0]["links"]["download"], out)
        scan_id = fname[: -len(".nii.gz")]
        new_rows.append({"child_id": child, "scan_id": scan_id, "age": f"{age}",
                         "sex": sex, "path": fname})
        print(f"{child} {scan_id} age {age} sex {sex}")
    existing = list(csv.DictReader(open(MANIFEST)))
    have = {r["child_id"] for r in existing}
    fresh = [r for r in new_rows if r["child_id"] not in have]
    with open(MANIFEST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["child_id", "scan_id", "age", "sex", "path"])
        for r in fresh:
            w.writerow(r)
    print(f"appended {len(fresh)} rows; manifest now {len(existing) + len(fresh)}")


if __name__ == "__main__":
    main()
