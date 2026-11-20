"""Stress-test the matched pooled scan-swap analysis with a known signed memory code."""
import json

import nki_fewshot as adapt
import nki_transfer as transfer
import paired_pool_augmentation as swap
import numpy as np
import pandas as pd


pilot = transfer.pilot
OUT = transfer.ROOT / "results/density-controls/positive-control"
PEOPLE = 24
BUDGETS = (0, 8, 16, 32, 64)
NOISE = .25
PERSONAL_CHANNELS = 6
DIM = len(pilot.DELAYS)


def signal_design(sign, stream_id, role, draw, length):
    """The states of a noisy linear delay line with a person-dependent signed output."""
    signal = pilot.rng(735, stream_id, role, draw, 0).uniform(-1, 1, length + pilot.WARMUP)
    previous = np.arange(pilot.WARMUP, len(signal))[:, None] - pilot.DELAYS
    targets = signal[previous]
    noise = pilot.rng(735, stream_id, role, draw, 1).normal(
        scale=NOISE * np.sqrt(pilot.SIGNAL_VAR), size=targets.shape)
    states = targets + noise
    states[:, :PERSONAL_CHANNELS] *= sign
    return states, targets


def fit_from_rows(designs, rows):
    return pilot.RidgeFamily(pilot.merge([
        pilot.Moment.from_data(pilot.sampled(designs[person], count))
        for person, count in rows]))


def scan_priors(target, foil, common, bank, validation, candidates, signs, count_added):
    shuffled = pilot.rng(173, PEOPLE, target, foil).permutation(common)
    cut = round(len(common)*2/3)
    train_rows = swap.split_rows(shuffled[:cut], pilot.FIT_N)
    val_rows = swap.split_rows(shuffled[cut:], pilot.VALID_N)
    validate = pilot.combine([pilot.sampled(validation[i], n) for i, n in val_rows])
    alpha = fit_from_rows(bank, train_rows).tune(validate, transfer.PENALTIES)
    rows = swap.split_rows(common, pilot.FIT_N - count_added)
    base = pilot.merge([pilot.Moment.from_data(pilot.sampled(bank[i], n)) for i, n in rows])
    assert base.count == pilot.FIT_N - count_added
    pooled_models = []
    for person in (target, foil):
        data = pilot.sampled(candidates[signs[person]], count_added)
        pooled_models.append(pilot.RidgeFamily(pilot.merge([base, pilot.Moment.from_data(data)])))
    pooled_models.append(fit_from_rows(bank, swap.split_rows(common, pilot.FIT_N)))
    weights = [m.weights(alpha) for m in pooled_models]
    intercepts = [m.my - m.mx @ w for m, w in zip(pooled_models, weights)]
    return np.stack(weights, axis=1), np.stack(intercepts), alpha


def measure_calibration(weights, intercepts, test, calibration, variances):
    mse = (adapt.residuals(*test, weights, intercepts)**2).mean(axis=0)
    for rep, streams in enumerate(calibration):
        for budget in BUDGETS:
            if budget == 0:
                error = mse
            else:
                folds = [(x[:budget//4], y[:budget//4]) for x, y in streams]
                selected = adapt.choose(folds, weights, intercepts)
                fitted = adapt.ResidualFit(*pilot.combine(folds), weights, intercepts)
                error = fitted.test_errors(*test, weights, intercepts, selected)
            yield rep, budget, (1-error/variances).mean(axis=1)


def score(injection):
    order = pilot.rng(172, PEOPLE).permutation(PEOPLE)
    signs = np.ones(PEOPLE, dtype=int)
    if injection:
        signs[order[1::2]] = -1
    foils = swap.partners(PEOPLE)
    candidates = {s: signal_design(s, 29000, 80, 0, pilot.FIT_N) for s in set(signs)}
    bank = [signal_design(s, i, 0, 0, pilot.FIT_N) for i, s in enumerate(signs)]
    validation = [signal_design(s, i, 1, 0, pilot.VALID_N) for i, s in enumerate(signs)]
    rows = []
    for target, choices in enumerate(foils):
        sign = signs[target]
        test_parts = [signal_design(sign, target, 61, f, 512) for f in range(4)]
        test = pilot.combine(test_parts)
        variance = test[1].var(axis=0)
        truth_sign = np.ones(DIM)
        truth_sign[:PERSONAL_CHANNELS] = sign
        oracle = (1-((test[1]-test[0]*truth_sign/(1+NOISE**2))**2).mean(axis=0)/variance).mean()
        cal = [[signal_design(sign, target, 60, 4*rep+f, max(BUDGETS)//4) for f in range(4)]
               for rep in range(adapt.REPEATS)]
        for i_foil, foil in enumerate(choices):
            common = [i for i in range(PEOPLE) if i not in (target, foil)]
            for rule, n_added in (("equal_donor", round(pilot.FIT_N/(PEOPLE-1))),
                                  ("fixed_025", pilot.FIT_N//4)):
                w, b, alpha = scan_priors(target, foil, common, bank, validation, candidates, signs, n_added)
                for rep, budget, result in measure_calibration(w, b, test, cal, variance):
                    rows.append((injection, rule, target, foil, i_foil, int(sign != signs[foil]),
                                 rep, budget, *result, result[0]-result[1], oracle, alpha))
    return rows


def sustained(grid, values):
    return next((int(k) for i,k in enumerate(grid) if min(values[i:]) >= .5), None)


def main():
    columns = ["injection", "source_rule", "target", "foil", "comparison", "opposite_code",
               "draw", "budget", "own_r2", "other_r2", "bank_r2", "gain_r2", "oracle_r2", "source_alpha"]
    df = pd.DataFrame(score(0) + score(1), columns=columns)
    assert len(df) == 2*2*PEOPLE*3*adapt.REPEATS*len(BUDGETS)
    assert df.duplicated(columns[:5] + ["draw", "budget"]).sum() == 0
    identical = df[(df.injection == 0) | (df.opposite_code == 0)]
    np.testing.assert_allclose(identical.gain_r2, 0, atol=1e-12, rtol=0)
    assert df[(df.injection == 1)].opposite_code.mean() == 2/3
    summaries = []
    for (injected, rule), scores in df.groupby(["injection", "source_rule"]):
        curves = scores.groupby("budget")[["own_r2", "other_r2", "bank_r2"]].mean()
        row = {"injection": int(injected), "source_rule": rule}
        row["crossings"] = {m:sustained(curves.index, curves[m].to_numpy()) for m in curves}
        for budget in (0,8):
            grouped = scores[scores.budget == budget].groupby("target").gain_r2.mean().to_numpy()
            rng = np.random.default_rng(735810)
            bs = rng.choice(grouped,size=(10000,PEOPLE)).mean(axis=1)
            row[f"gain_R2_at_{budget}"] = [float(grouped.mean()), *np.quantile(bs, [.025,.975]).tolist()]
        row["oracle_R2_observed"] = float(scores.oracle_r2.mean())
        summaries.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "scores.csv", index=False)
    df.groupby(["injection","source_rule","budget"])[["own_r2","other_r2","bank_r2","gain_r2"]].mean().to_csv(OUT/"curves.csv")
    (OUT/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summaries,indent=1))


if __name__ == "__main__":
    main()
