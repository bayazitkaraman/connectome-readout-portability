"""Recreate the donor priors, stream-held-out penalty selection and test scores independently."""
from pathlib import Path
import json
from verify_results import SingularRidge, fixtures, join, sampled, stream
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "nki-calibration"
RETEST = ROOT / "results" / "nki-retest"
GRID = 10.0 ** np.arange(-14, 5, dtype=float)
BUDGETS = (0, 8, 32, 128, 2352)


def adjacency(raw):
    a = raw.astype(float, copy=True)
    np.fill_diagonal(a, 0.0)
    return a * (.9 / np.max(np.abs(eigvalsh(a))))


def investigate_sources(weights, subjects, inputs):
    train, valid, models = [], [], []
    empirical_loss = []
    for j in range(len(subjects)):
        a = adjacency(weights[0, j])
        ident = 10_000 + j
        train.append(fixtures(a, inputs, ident, 0, 2352))
        valid.append(fixtures(a, inputs, ident, 1, 1176))
        model = SingularRidge(train[-1])
        models.append(model)
        validation_a = adjacency(weights[1, j])
        x, y = fixtures(validation_a, inputs, ident + len(subjects), 1, 1176)
        empirical_loss.append([3 * np.mean((model.predict(x, alpha)-y)**2) for alpha in GRID])
    return train, valid, models, np.asarray(empirical_loss)


def initial_readouts(sources, j, meta, subjects):
    train, valid, models, cross_visit_losses = sources
    # The personal source model is stabilized by penalties learned from other
    # participants' paired visits. The target's repeat visit cannot select it.
    other = np.arange(len(subjects)) != j
    alpha = GRID[np.argmin(np.mean(cross_visit_losses[other], axis=0))]
    own_prior = lambda x: models[j].predict(x, alpha)
    # Recreate the donor split and check the archived exclusion and row counts.
    donors = np.flatnonzero(other)
    shuffled = stream(70, j).permutation(donors)
    fit, validate = shuffled[:16], shuffled[16:]
    archived = meta["pooled_donor_rows"][subjects[j]]
    for selected, label, n in [(donors, "final_fit", 98), (fit, "inner_fit", 147),
                               (validate, "inner_validation", 147)]:
        assert {subjects[i]: n for i in selected} == archived[label]
    inner = SingularRidge(join([sampled(train[i],147) for i in fit]))
    pooled_alpha = inner.tune(join([sampled(valid[i],147) for i in validate]), GRID)
    pooled = SingularRidge(join([sampled(train[i],98) for i in donors]))
    return {"personal":own_prior, "pooled":lambda x: pooled.predict(x, pooled_alpha),
            "scratch":lambda x: np.zeros((len(x),10))}


def heldout_select(parts, prior):
    """Fit residuals directly on three pieces and validate all candidates on the fourth."""
    costs = np.zeros(len(GRID)+1)
    for held in range(4):
        x, y = join([part for k,part in enumerate(parts) if k != held])
        model = SingularRidge((x,y-prior(x)))
        xv, yv = parts[held]
        residual = yv-prior(xv)
        costs[0] += np.mean(residual**2)
        for index, alpha in enumerate(GRID[::-1]):
            costs[index+1] += np.mean((residual-model.predict(xv,alpha))**2)
    # Exact ties are resolved according to the production candidate order.
    best = np.argmin(costs)
    return None if best==0 else GRID[::-1][best-1]


def measure(pred, y):
    mse = np.mean(np.square(pred-y),axis=0)
    return np.array([np.mean(1-mse/y.var(axis=0)), 3*np.mean(mse)])


def audit_prediction(pieces, prior, x_test, y_test, row):
    pred = prior(x_test)
    if len(pieces[0][0]):
        penalty = heldout_select(pieces,prior)
        if penalty is None:
            assert str(row.alpha).lower()=="keep", str(row.alpha)
        else:
            np.testing.assert_allclose(float(row.alpha),penalty,rtol=1e-12,atol=0)
            x,y = join(pieces)
            model = SingularRidge((x,y-prior(x)))
            pred += model.predict(x_test,penalty)
    got = measure(pred,y_test)
    published = np.array([row.r2,row.nmse])
    np.testing.assert_allclose(got,published,atol=5e-7,rtol=3e-6)
    return got, np.max(np.abs(got-published))


def check_scores(scores):
    with np.load(ROOT/"data"/"nki"/"nki_reference.npz",allow_pickle=False) as data:
        subjects = data["subjects"].astype(str).tolist()
        weights = data["weights"][0]
        assert data["halves"][0] == "odd"
    chosen = stream(10).choice(weights.shape[-1],8,replace=False)
    inputs = np.zeros(weights.shape[-1])
    inputs[chosen[:4]]=.5
    inputs[chosen[4:]]=-.5
    meta = json.loads((RETEST/"run_info.json").read_text(encoding="utf-8"))
    assert subjects == meta["ordered_subjects"]
    sources = investigate_sources(weights, subjects, inputs)
    primary = scores.query("half=='odd' and source_visit==1").set_index(["subject","repeat","budget","method"]).sort_index()
    largest = 0.
    checked = 0
    checked_selections = 0
    for j in (0,17):
        priors = initial_readouts(sources,j,meta,subjects)
        ident = 10_000 + len(subjects) + j
        a = adjacency(weights[1,j])
        x_test,y_test = join([fixtures(a,inputs,ident,61,512,k) for k in range(4)])
        for rep in (0,5):
            streams = [fixtures(a,inputs,ident,60,588,rep*4+k) for k in range(4)]
            for budget in BUDGETS:
                pieces = [(x[:budget//4], y[:budget//4]) for x,y in streams]
                for label, prior in priors.items():
                    row = primary.loc[(subjects[j],rep,budget,label)]
                    _, discrepancy = audit_prediction(pieces,prior,x_test,y_test,row)
                    checked_selections += budget>0
                    largest = max(largest,discrepancy)
                    checked += 1
    log = [f"Independently regenerated odd-subset baseline-to-repeat scores for subjects {subjects[0]} and "
            f"{subjects[17]}, calibration repetitions 0 and 5, at budgets {list(BUDGETS)}.",
            "Source and validation stream roles 0/1 reproduced from the underlying matrices "
            "with a direct tanh recurrence and SciPy SVD. Rebuilt transfer, pooled and zero priors, "
            "including target-excluded regularization and nested 16+8 pooled donor splits.",
            f"Reconstructed {checked_selections} four-fold penalty selections from disjoint labeled streams and "
            f"{checked} pairs of held-out R² and normalized-MSE scores. Maximum absolute discrepancy {largest:.4g}.",
            "Every finite fit adds a centered ridge fit on target residuals. The unchanged-prior candidate "
            "has no intercept adaptation. Role 60 supplies only the stated calibration prefixes. "
            "Role 61 supplies four independent test streams shared by every method."]
    path = OUT/"refinement"/"scores.csv"
    if path.exists():
        refined = pd.read_csv(path,dtype={"subject":str})
        person = "A00063383"
        j = subjects.index(person)
        priors = initial_readouts(sources,j,meta,subjects)
        ident = 10_000 + len(subjects) + j
        a = adjacency(weights[1,j])
        x_test,y_test = join([fixtures(a,inputs,ident,61,512,k) for k in range(4)])
        streams = [fixtures(a,inputs,ident,60,588,k) for k in range(4)]
        rows = refined.query("subject==@person and repeat==0").set_index(["budget","method"])
        for budget in (40,44,48):
            pieces = [(x[:budget//4],y[:budget//4]) for x,y in streams]
            r2s = []
            for label,prior in priors.items():
                row = rows.loc[(budget,label)]
                got,_ = audit_prediction(pieces,prior,x_test,y_test,row)
                r2s.append(f"{label} R² {got[0]:.7f} (penalty {row.alpha})")
            log.append(f"Refinement audit for {person}, repetition 0, {budget} steps: "+", ".join(r2s)+".")
    return log


def main():
    scores = pd.read_csv(OUT/"scores.csv",dtype={"subject":str})
    keys = ["half", "source_visit", "target_visit","subject","repeat","budget","method"]
    assert len(scores)==4*25*6*9*3 and not scores.duplicated(keys).any()
    assert (scores.nmse>=0).all() and np.isfinite(scores[["r2","nmse"]]).all().all()
    # The same four test signals and the same prior are reused across calibration repetitions.
    zero = scores[scores.budget==0].groupby(["half","source_visit","subject","method"])
    assert zero.r2.nunique().max() == 1 and zero.nmse.nunique().max()==1
    log = check_scores(scores)
    early = scores.query("half=='odd' and source_visit==1 and 8<=budget<=256")
    means = early.groupby(["subject","method"]).r2.mean().unstack()
    delta = means.personal - means.pooled
    log.append(f"Raw scores contain {len(scores)} distinct conditions. The primary mean personal-minus-pooled "
               f"R² across the six low budgets is {delta.mean():.10g}, positive for {(delta>0).sum()}/25 people.")
    out = "\n".join(log) + "\n"
    (OUT/"verification.txt").write_text(out,encoding="utf-8")
    print(out)


if __name__=="__main__":
    main()
