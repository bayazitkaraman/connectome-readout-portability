"""Assemble the two Schaefer resolutions from the published m2g BNU1 visit pairs."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import re
import time
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/replication_candidates/corr"
METADATA = ROOT / "results/pipeline-robustness"
BUCKET = "https://open-neurodata.s3.amazonaws.com/"
PREFIX = "m2g/Diffusion/BNU1-8-27-20-m2g-native-csa-det/Connectomes/"


def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=45) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt+1)


def assemble(size):
    atlas = f"Schaefer{size}_space-MNI152NLin6_res-2x2x2"
    prefix = PREFIX + atlas + "/"
    listing_url = BUCKET + "?list-type=2&prefix=" + urllib.parse.quote(prefix,safe="")
    atlas_url = "https://raw.githubusercontent.com/neurodata/neuroparc/master/atlases/label/Human/Metadata-json/" + atlas + ".json"
    listing_payload, atlas_payload = fetch(listing_url), fetch(atlas_url)
    listing = ET.fromstring(listing_payload)
    assert listing.findtext("{*}IsTruncated") == "false"
    entries = {r.findtext("{*}Key"):(int(r.findtext("{*}Size")),r.findtext("{*}ETag").strip('"'))
               for r in listing.findall("{*}Contents") if r.findtext("{*}Key") != prefix}
    with np.load(OUT/"bnu1_dkt_retest.npz",allow_pickle=False) as original:
        ids = original["subjects"].astype(str)
    regions = json.loads(atlas_payload)["rois"]
    labels = sorted(int(k) for k in regions if int(k)>0)
    assert labels == list(range(1,size+1))
    coordinates = np.asarray([regions[str(i)]["center"] for i in labels],float)
    assert coordinates.shape == (size,3) and np.isfinite(coordinates).all()
    arrays = np.zeros((2,len(ids),size,size),dtype=np.int32)
    jobs = [(i,v,prefix+f"sub-{person}_ses-{v+1}_connectome.csv")
            for i,person in enumerate(ids) for v in range(2)]
    assert {key for _,_,key in jobs} == set(entries)
    manifest_file = METADATA/f"schaefer{size}_manifest.json"
    known = json.loads(manifest_file.read_text()) if manifest_file.exists() else None
    recorded = {item["key"]:item for item in known["files"]} if known else {}
    if known:
        assert hashlib.sha256(atlas_payload).hexdigest() == known["atlas_sha256"]
    def grab(job):
        i,v,key = job
        payload = fetch(BUCKET+key)
        length,md5 = entries[key]
        assert re.fullmatch("[0-9a-f]{32}",md5) and len(payload) == length and hashlib.md5(payload).hexdigest() == md5
        digest = hashlib.sha256(payload).hexdigest()
        if known:
            assert digest == recorded[key]["sha256"]
        edge = np.loadtxt(BytesIO(payload),ndmin=2)
        assert edge.shape[1]==3 and np.isfinite(edge).all() and (edge == np.floor(edge)).all()
        a,b,w = edge.astype(np.int64).T
        assert ((1<=a)&(a<b)&(b<=size)&(w>0)&(w<2**31)).all() and len(set(zip(a,b)))==len(a)
        arrays[v,i,a-1,b-1] = arrays[v,i,b-1,a-1] = w
        return dict(key=key,bytes=length,md5=md5,sha256=digest)
    with ThreadPoolExecutor(max_workers=8) as pool:
        files=list(pool.map(grab,jobs))
    assert len(files)==114 and len({x["sha256"] for x in files})==114
    output = OUT/f"bnu1_schaefer{size}.npz"
    np.savez_compressed(output,weights=arrays,subjects=ids,sessions=np.array(["1","2"]),centroids=coordinates)
    manifest = dict(atlas=atlas,atlas_url=atlas_url,atlas_sha256=hashlib.sha256(atlas_payload).hexdigest(),
                    listing=listing_url,weights_sha256=hashlib.sha256(arrays.tobytes()).hexdigest(),files=files)
    if known:
        assert manifest == known
    else:
        manifest_file.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"{atlas}: {len(files)} source files verified",flush=True)


if __name__=="__main__":
    for scale in (200,300):
        assemble(scale)
