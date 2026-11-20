"""Verify and assemble NeuroData's 57 BNU1 DKT visit pairs for transfer testing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np


OUT = Path(__file__).resolve().parents[1] / "data/replication_candidates/corr"
BUCKET = "https://open-neurodata.s3.amazonaws.com/"
PREFIX = "m2g/Diffusion/BNU1-8-27-20-m2g-native-csa-det/Connectomes/DKT_space-MNI152NLin6_res-2x2x2/"
LIST_URL = BUCKET + "?list-type=2&prefix=" + PREFIX.replace("/", "%2F")
ATLAS_URL = ("https://raw.githubusercontent.com/neurodata/neuroparc/master/atlases/label/Human/"
             "Metadata-json/DKT_space-MNI152NLin6_res-2x2x2.json")
ATLAS_SHA = "8a5b9ebc1ea6f846751e4f8d1a0c8938addbbf617a35b302b954eae1b228e706"
LISTING_SHA = "0e61bc52cb7bbaa292d256e1481e8056df5779e8a09c59661bcfed6deaad65bf"


def get(path, url, algorithm, expected):
    if not path.exists():
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        if hashlib.new(algorithm, payload).hexdigest() != expected:
            raise ValueError(f"Unexpected source checksum for {path.name}.")
        path.write_bytes(payload)
    payload = path.read_bytes()
    if hashlib.new(algorithm, payload).hexdigest() != expected:
        raise ValueError(f"Cached file failed its {algorithm} check: {path.name}.")
    return payload


def assemble(out):
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    listing = ET.fromstring(get(out / "listing.xml", LIST_URL, "sha256", LISTING_SHA))
    atlas = json.loads(get(out / "atlas.json", ATLAS_URL, "sha256", ATLAS_SHA))
    regions = np.array(sorted(int(i) for i in atlas["rois"] if i != "0"))
    if not np.array_equal(regions, np.arange(1, 84)):
        raise ValueError("Unexpected DKT labels.")
    jobs = []
    for item in listing.findall("{*}Contents"):
        key = item.findtext("{*}Key")
        match = re.fullmatch(re.escape(PREFIX) + r"sub-(\d+)_ses-([12])_connectome\.csv", key)
        if match is None:
            raise ValueError(f"Unexpected object {key}.")
        jobs.append((key, match[1], match[2], int(item.findtext("{*}Size")),
                     item.findtext("{*}ETag").strip('"')))
    subjects = sorted({j[1] for j in jobs})
    sessions = ["1", "2"]
    if len(jobs) != 114 or len(subjects) != 57 or {(j[1],j[2]) for j in jobs} != {
            (s,v) for s in subjects for v in sessions}:
        raise ValueError("Expected exactly 57 complete visit pairs in the BNU1 release.")
    weights = np.zeros((len(sessions),len(subjects),len(regions),len(regions)), dtype=np.int32)

    def load(job):
        key, subject, session, size, md5 = job
        path = raw / Path(key).name
        payload = get(path, BUCKET+key, "md5", md5)
        if len(payload) != size:
            raise ValueError(f"Unexpected size for {path.name}.")
        edges = np.loadtxt(path)
        if (edges.ndim != 2 or edges.shape[1] != 3 or not np.isfinite(edges).all() or
                not (edges == np.floor(edges)).all() or (edges < 1).any() or
                not (edges[:,0] < edges[:,1]).all() or edges[:,1].max() > regions.max() or
                edges[:,2].max() > np.iinfo(weights.dtype).max):
            raise ValueError(f"Invalid upper-triangular weighted edge list in {path.name}.")
        a, b, w = edges.astype(np.int32).T
        if len(np.unique(np.stack((a,b),axis=1),axis=0)) != len(w):
            raise ValueError(f"Duplicate edge in {path.name}.")
        matrix = weights[sessions.index(session), subjects.index(subject)]
        matrix[a-1,b-1] = matrix[b-1,a-1] = w
        return dict(subject=subject, session=session, key=key, bytes=size, md5=md5,
                    sha256=hashlib.sha256(payload).hexdigest(), edges=len(w),
                    isolates=int(np.sum(matrix.sum(axis=0)==0)))

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(load, jobs))
    if len({rec["sha256"] for rec in records}) != len(records):
        raise ValueError("Duplicate source graphs across person/visit cells.")
    dest = out / "bnu1_dkt_retest.npz"
    np.savez_compressed(dest, subjects=np.asarray(subjects), sessions=np.asarray(sessions),
                        region_ids=regions, weights=weights)
    manifest = dict(cohort="CoRR BNU1 / Beijing Normal University C-BIRD",
                    acquisition="https://helab.bnu.edu.cn/DOWNLOADS/Data/index.htm",
                    paper="https://doi.org/10.1038/sdata.2015.56",
                    consortium_paper="https://doi.org/10.1038/sdata.2014.49",
                    processing="https://docs.neurodata.io/m2g/paper/data.html",
                    bucket=BUCKET, listing=LIST_URL, listing_sha256=LISTING_SHA,
                    atlas=ATLAS_URL, atlas_sha256=ATLAS_SHA,
                    axes=["visit", "participant", "source region", "target region"],
                    weights="Released undirected weighted edges, mirrored; unlisted pairs are zero. "
                            "All 83 nonbackground atlas regions retained, including isolates.",
                    data_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                    shape=list(weights.shape), files=records)
    (out/"bnu1_source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n", encoding="utf-8")
    print(f"{dest}: {weights.shape}; all {len(records)} source files verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    assemble(parser.parse_args().out)
