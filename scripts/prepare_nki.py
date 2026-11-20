"""Verify and extract the unperturbed deterministic NKI visit-paired connectomes.

Run from any directory. Supply --source and --crosswalk to use local downloads.
The output is a numeric NPZ that can be read with NumPy's allow_pickle=False.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import urllib.request

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1] / 'data' / 'nki'
FILENAME = 'impactofinstability_connect_dset25x2x2x20_inputs.h5'
SOURCE = f'https://zenodo.org/records/4041549/files/{FILENAME}?download=1'
MD5 = 'f457d4a860d79654672143c1ace95ee1'
CROSSWALK = 'http://fcon_1000.projects.nitrc.org/indi/enhanced/public_crosswalk_ids_08242023.csv'
CROSSWALK_SHA = 'f9181517bac7dc899dc5aeeda050158396dbcbecc1fd2c04716fa5ae987cdd25'
REV = '9f8345846ad9ef66c3e270bec44f4cf55b72b92d'
REPO = f'https://github.com/gkpapers/2020ImpactOfInstability/blob/{REV}/code/'


def digest(path, algorithm):
    hasher = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract(source, crosswalk, out):
    # Pandas' fixed-format object block requires pickle. Only read the exact
    # published, checksum-pinned file after validating its complete contents.
    if source.stat().st_size != 233049104 or digest(source, 'md5') != MD5:
        raise ValueError('Source file fails the published size/MD5 check.')
    if digest(crosswalk, 'sha256') != CROSSWALK_SHA:
        raise ValueError('Unexpected NKI crosswalk version.')
    frame = pd.read_hdf(source, key='graphs')
    data = frame.loc[(frame.simulation == 'ref') & (frame.pipeline == 'det'),
                     ['subject', 'session', 'directions', 'graph']]
    if len(data) != 100 or data.duplicated(['subject', 'session', 'directions']).any():
        raise ValueError('Expected one reference graph per person/visit/direction half.')
    subjects_in_release = data.subject.unique()
    lookup = pd.read_csv(crosswalk).query('A_number in @subjects_in_release')
    lookup = lookup[['A_number', 'old_session_name', 'new_session_id']].drop_duplicates()
    lookup = lookup.set_index(['A_number', 'old_session_name']).new_session_id
    if not lookup.index.is_unique:
        raise ValueError('Crosswalk assigns more than one visit to a subject/code pair.')
    graphs = data.set_index(['subject', 'session', 'directions']).graph
    subjects = np.array(sorted(data.subject.unique()))
    if len(subjects) != 25:
        raise ValueError('Expected 25 participants.')
    halves = np.array(['odd', 'even'])
    labels = np.empty((2, len(subjects)), dtype='U12')
    standard = np.empty_like(labels)
    weights = np.empty((2, 2, len(subjects), 83, 83), dtype=np.float64)
    unmatched = []
    for p, subject in enumerate(subjects):
        names = sorted(data.loc[data.subject == subject, 'session'].unique())
        if len(names) != 2:
            raise ValueError(f'Expected two acquisition visits for {subject}.')
        mapping = {name: lookup.loc[(subject, name)]
                   if (subject, name) in lookup.index else 'unmapped' for name in names}
        if list(mapping.values()).count('BAS1') != 1:
            raise ValueError(f'Expected exactly one person-specific BAS1 for {subject}.')
        for v, name in enumerate(sorted(names, key=lambda x: mapping[x] != 'BAS1')):
            labels[v, p], standard[v, p] = name, mapping[name]
            if mapping[name] == 'unmapped':
                unmatched.append([str(subject), str(name)])
            for h, half in enumerate(halves):
                g = np.asarray(graphs.loc[(subject, name, half)])
                if (g.shape != (83, 83) or not np.isfinite(g).all() or not (g >= 0).all()
                        or not np.array_equal(g, g.T) or not np.array_equal(g, np.floor(g))):
                    raise ValueError(f'Unexpected graph for {subject}/{name}/{half}.')
                weights[h, v, p] = g
    if unmatched != [['A00034854', 'NFB2R']]:
        raise ValueError(f'Unexpected unmatched legacy visit codes: {unmatched}.')
    if np.any(np.all(weights[:, 0] == weights[:, 1], axis=(-2, -1))):
        raise ValueError('A participant has identical matrices across visits.')

    out.mkdir(exist_ok=True, parents=True)
    path = out / 'nki_reference.npz'
    np.savez_compressed(path, weights=weights, subjects=subjects, halves=halves,
                        sessions=np.array(['BAS1', 'other']), session_labels=labels,
                        standard_session_labels=standard)
    with np.load(path, allow_pickle=False) as saved:
        for key, expected in [('weights', weights), ('subjects', subjects),
                              ('session_labels', labels), ('standard_session_labels', standard)]:
            np.testing.assert_array_equal(saved[key], expected)
    region_mask = np.triu(np.ones((83, 83), dtype=bool), 1)
    counts = (weights[..., region_mask] > 0).sum(axis=-1)
    present = weights > 0
    idx = np.arange(weights.shape[-1])
    present[..., idx, idx] = False
    isolates = (~present.any(axis=-1)).sum(axis=-1)
    record = dict(
        archive_title='Numerically Perturbed Structural Connectomes from 100 individuals in the NKI Rockland Dataset',
        depositor='Gregory Kiar', source=SOURCE, record_doi='10.5281/zenodo.4041549',
        updated_record_doi='10.5281/zenodo.4287624', license='CC BY 4.0', filename=FILENAME,
        bytes=source.stat().st_size, md5=MD5, source_sha256=digest(source, 'sha256'),
        graph_code=REPO+'processing/tractography/dipy_tracking.py',
        authors_retest_code=REPO+'analysis/tab1_discriminability/trt_testing.py',
        crosswalk_url=CROSSWALK, crosswalk_sha256=CROSSWALK_SHA,
        code_definitions='https://rocklandsample.org/for-researchers-rockland-sample-i-2011-2022-documentation-study-codes',
        selector="graphs key of HDF, simulation == 'ref' and pipeline == 'det'",
        atlas='DKT parcellation in native deposited 83 x 83 order. Background index removed by authors.',
        units='Counts of deterministic streamlines connecting region endpoints. Native diagonals retained in this extraction.',
        axes=['direction half (odd, even)', 'visit (BAS1, other)', 'subject (ascending source ID)', 'native region', 'native region'],
        sessions='Visit 0 is the individually verified first baseline (BAS1). Visit 1 is the other independently acquired scan. Scan intervals differ or are unavailable. The released matrices differ between visits for all people in each half.',
        pairing='Both direction halves occur once for every subject and legacy visit code. Odd and even are subsamples of the same diffusion acquisition, not distinct visits.',
        second_visit_counts={str(k): int(v) for k, v in zip(*np.unique(standard[1], return_counts=True))},
        unmatched=unmatched,
        unmatched_note='For A00034854 the archive gives DS2 and NFB2R. The public crosswalk matches DS2 to BAS1 and lists only NFBAR for BAS2. The second acquisition is retained under the original code without assigning BAS2 to it.',
        shape=list(weights.shape), people=len(subjects), matrix_count=100, source_table_rows=len(frame),
        nonzero_undirected_edges_range=[int(counts.min()), int(counts.max())],
        isolated_nodes_after_diagonal_removal_range=[int(isolates.min()), int(isolates.max())],
        weights_sha256=hashlib.sha256(np.asarray(weights, dtype='<f8').tobytes()).hexdigest(),
        npz_sha256=digest(path, 'sha256'))
    (out / 'provenance.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(f'Verified {weights.shape} graphs from {len(subjects)} participants in {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--crosswalk', type=Path)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    with TemporaryDirectory() as temp:
        src = args.source or Path(temp) / FILENAME
        cross = args.crosswalk or Path(temp) / 'crosswalk.csv'
        if args.source is None:
            urllib.request.urlretrieve(SOURCE, src)
        if args.crosswalk is None:
            urllib.request.urlretrieve(CROSSWALK, cross)
        extract(src, cross, args.out)
