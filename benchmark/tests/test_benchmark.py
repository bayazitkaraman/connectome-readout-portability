from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import run_feasibility as pilot
from benchmark.analysis import Evaluation, load_data, make_map, simulate, state_comparison, transfer_indices
from benchmark.example import create
from benchmark.metrics import bin_curve, identify, reliability, wiring_similarity


class BenchmarkTest(unittest.TestCase):
    def test_direction_matched_noise_and_constant_parcel(self):
        from types import SimpleNamespace
        rng = np.random.default_rng(31)
        xs, xt = rng.normal(size=(2, 37, 3))
        y = rng.normal(size=(37, 2))
        weights = rng.normal(size=(1, 4, 2))
        scan = SimpleNamespace(weights=weights, train_mean=xs.mean(axis=0), train_sd=xs.std(axis=0),
                               train_varied=np.array([True]*3), train_rms=np.sqrt(xs.var(axis=0).mean()))
        for target in (xs, xt):
            calibrated = target.copy()
            calibrated[:, 0] = 1.
            row = state_comparison(xs,target,y,scan,0,calibrated)
            self.assertTrue(np.isfinite(row["aligned_r2"]))
            self.assertAlmostEqual(row["source_r2"] - row["repeat_r2"], row["spatial_loss"] + row["error_cross_term"])
            if target is xs:
                self.assertEqual(row["displacement_pct"], 0.)
                self.assertEqual(row["isotropic_r2"],row["repeat_r2"])
            # Enumerate independent signs for whole displacement vectors: averaging
            # both signs cancels the residual cross term, retaining spatial structure.
            scores = []
            for sign in (-1, 1):
                residual = (xs + sign*(target-xs)) @ weights[0,:-1] + weights[0,-1] - y
                scores.append((1 - (residual**2).mean(axis=0)/y.var(axis=0)).mean())
            self.assertAlmostEqual(np.mean(scores),row["spatial_noise_r2"])

    def test_qr_matches_explicit_r2_and_source_stream(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(lags=list(range(1,11)),task="recall",warmup=200)
        rng = np.random.default_rng(301)
        upper = np.triu(rng.random((15,15)),1)
        net = pilot.reservoir(upper+upper.T,.9)
        mapping = make_map(pilot.SEED,10,15,8)
        self.assertTrue(np.array_equal(mapping[0],pilot.input_map(15)[0]))
        data = simulate(net,mapping,cfg,pilot.SEED,10000,0,40)
        for a,b in zip(data,pilot.simulate(net,mapping,10000,0,40)):
            np.testing.assert_array_equal(a,b)
        model = pilot.RidgeFamily.from_design(data)
        x,y = pilot.simulate(net,mapping,10001,61,25)
        w = model.weights(1)
        got = Evaluation.from_data((x,y)).r2(np.vstack((w,model.my-model.mx@w))[None])
        expected = 1-np.mean((model.predict(x,1)-y)**2,axis=0)/np.var(y,axis=0)
        np.testing.assert_allclose(got[0],expected,atol=1e-11)
        cfg.task = "product"
        cfg.lags = [1,2,3]
        xp,yp = simulate(net,mapping,cfg,pilot.SEED,10000,0,40)
        np.testing.assert_array_equal(xp,data[0])
        np.testing.assert_array_equal(yp[:,0],data[1][:,:3].prod(axis=1))

    def test_metrics(self):
        rank,top = identify(np.array([[3,1,1],[3,2,3],[0,2,2]]))
        np.testing.assert_array_equal(rank,[1.5,1.5,2])
        np.testing.assert_array_equal(top,[.5,.5,0])
        with self.assertRaises(ValueError):
            wiring_similarity(np.full((1,16,16),.1), np.full((1,16,16),.1))
        upper=np.triu(np.random.default_rng(104).integers(0,10,(16,16)),1)
        g=upper+upper.T
        candidates=np.stack((g,3*g))
        rank,top=identify(wiring_similarity(candidates,candidates))
        np.testing.assert_array_equal(rank,[1.5,1.5])
        np.testing.assert_array_equal(top,[.5,.5])
        a=np.array([1.,4.,5.,6.])
        self.assertAlmostEqual(reliability(a,a)["icc_absolute_single"],1)
        self.assertAlmostEqual(reliability(a,a)["spearman"],1)
        self.assertTrue(np.isnan(reliability(a*0,a*0)["icc_absolute_single"]))
        self.assertLess(reliability(a,a+5)["icc_absolute_single"],1)
        self.assertAlmostEqual(reliability(a,a+5)["spearman"],1)
        curve=bin_curve(np.eye(3),np.eye(3),[.5])
        self.assertEqual([r["n_pairs"] for r in curve],[0,6,3,0])
        # Changing either held-out identity cannot change the selected penalty.
        loss=np.array([[20.,1],[10,0],[0,3],[0,2]])
        orig=transfer_indices(loss)[0,1]
        loss[:2]=[[0,1000],[0,1000]]
        self.assertEqual(orig,transfer_indices(loss)[0,1])

    def test_input_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            src=root/"input.npz"
            create(src)
            graph,ids,sessions,regions=load_data(src,None)
            self.assertEqual(graph.shape,(2,5,16,16))
            self.assertEqual(regions[-1],"15")
            np.testing.assert_allclose(np.diag(wiring_similarity(graph[0],graph[0])),1)
            np.savez(root/"halves.npz",weights=np.stack([graph,graph]),subjects=ids,sessions=sessions,
                     halves=["odd","even"])
            np.testing.assert_array_equal(load_data(root/"halves.npz","odd")[0],graph)
            with self.assertRaises(ValueError):
                load_data(root/"halves.npz",None)
            for bad in ("nan","asymmetric","duplicate_id","negative","empty"):
                corrupt=graph.copy()
                names=ids.copy()
                if bad=="nan": corrupt[0,0,1,3]=np.nan
                if bad=="asymmetric": corrupt[0,0,1,3]+=1
                if bad=="duplicate_id": names[0]=names[1]
                if bad=="negative": corrupt[0,0,1,3]=-1
                if bad=="empty": corrupt[0,0]=0
                np.savez(root/"bad.npz",weights=corrupt,sessions=sessions,subjects=names)
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    load_data(root/"bad.npz",None)
            cmd = [sys.executable,"-m","benchmark",str(src),str(root/"result"),"--task","product",
                   "--seeds","42","43","--penalties",".001","10","--train-steps","48","--validation-steps","30",
                   "--test-steps","32","--test-streams","2","--warmup","15","--state-comparisons",
                   "--alignment-steps","16"]
            subprocess.run(cmd,check=True,capture_output=True)
            with np.load(root/"result/matrix_seed42_visit0_to_1.npz") as f:
                self.assertEqual(f["fixed_r2"].shape,(5,5,2))
                self.assertTrue(np.isfinite(f["transfer_tuned_r2"]).all())
                self.assertEqual(set(np.unique(f["transfer_tuned_penalties"])).difference([.001,10]),set())
            self.assertTrue((root/"result/matrix_seed43_visit1_to_0.npz").exists())
            self.assertEqual(len((root/"result/state_pairs.csv").read_text().splitlines()), 21)
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)


if __name__ == "__main__":
    unittest.main()
