"""Extract the coarsest individual structural matrices in Griffa et al.'s Zenodo v1 release.

The MAT v5 file has one compressed struct, and its first cell contains the complete
coarsest structural array. Reading the published file prefix saves a 700 MB download.
Supply --source to use an existing full MAT file or the identical byte range.
Requirements: Python 3 and NumPy. Run from any directory.
"""
import argparse
import hashlib
import struct
import urllib.request
import zlib
from pathlib import Path

import numpy as np


SOURCE_URL = "https://zenodo.org/api/records/2872624/files/Individual_Connectomes.mat/content"
PREFIX_SIZE = 4 * 1024 * 1024
PREFIX_SHA256 = "a0b7703789e093a58e2e5118aa0a2e6edc19496886f67f9ab47f60d29b6319b6"
WEIGHTS_SHA256 = "e1a5cea2e0f987b5126721af4d468f90e23ff01016b022cbed9afb041d57b325"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "lausanne_scale1.npz"


def source_prefix(local_source):
    if local_source is not None:
        with local_source.open("rb") as file:
            data = file.read(PREFIX_SIZE)
    else:
        request = urllib.request.Request(SOURCE_URL, headers={"Range": f"bytes=0-{PREFIX_SIZE - 1}"})
        with urllib.request.urlopen(request, timeout=80) as response:
            data = response.read(PREFIX_SIZE)
    if hashlib.sha256(data).hexdigest() != PREFIX_SHA256:
        raise ValueError("Source differs from the verified prefix of Zenodo record 2872624.")
    return data


def extract(prefix):
    # The file is pinned to a single version. Verify MAT tags along the path
    # connMatrices -> SC (first struct field) -> first cell -> real float64 data.
    if not prefix.startswith(b"MATLAB 5.0 MAT-file") or prefix[126:128] != b"IM":
        raise ValueError("Expected a little-endian MAT v5 archive.")
    tag, compressed_size = struct.unpack_from("<II", prefix, 128)
    if tag != 15 or compressed_size + 136 != 700865960:
        raise ValueError("Unexpected top-level MAT element.")
    raw = zlib.decompressobj().decompress(prefix[136:], 3 * 1024 * 1024)
    if raw[48:60] != b"connMatrices" or raw[80:86] != b"SC\x00FC\x00":
        raise ValueError("Expected connMatrices with SC and FC fields.")
    if struct.unpack_from("<ii", raw, 120) != (5, 1):
        raise ValueError("Expected five structural scale cells.")
    cell_start = 136
    tag, cell_bytes = struct.unpack_from("<II", raw, cell_start)
    dims = struct.unpack_from("<iii", raw, cell_start + 32)
    dtype_tag, nbytes = struct.unpack_from("<II", raw, cell_start + 56)
    if not (tag == 14 and dims == (68, 68, 70) and dtype_tag == 9
            and nbytes == 8 * np.prod(dims) and cell_bytes == nbytes + 56):
        raise ValueError("Unexpected structural scale 1 array.")
    flat = np.frombuffer(raw, dtype="<f8", count=int(np.prod(dims)), offset=cell_start + 64)
    matrices = flat.reshape(dims, order="F")
    weights = np.ascontiguousarray(np.moveaxis(matrices, 2, 0))
    if hashlib.sha256(weights.tobytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("Extracted matrices differ from verified release values.")
    if not (np.isfinite(weights).all() and (weights >= 0).all()
            and np.array_equal(weights, weights.transpose(0, 2, 1))):
        raise ValueError("Invalid structural matrices.")
    if len({graph.tobytes() for graph in weights}) != len(weights):
        raise ValueError("Duplicate participant matrix.")
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional local copy of Individual_Connectomes.mat")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    weights = extract(source_prefix(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, weights=weights,
                        participant_index=np.arange(1, weights.shape[0] + 1),
                        region_index=np.arange(1, weights.shape[1] + 1))
    print(f"{args.output}: {weights.shape}. Preserved raw diagonal and original ordering.")


if __name__ == "__main__":
    main()

