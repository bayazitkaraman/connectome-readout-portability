"""Evaluate the paired noise/graph replacement contrast under prespecified settings."""
import json
from pathlib import Path

from pipeline_robustness import pilot, read_graphs, threshold, transfer
from run_visit_portability_path import fresh_test
import numpy as np
import pandas as pd


OUT = transfer.ROOT / "results" / "headline-robustness"
GRID = transfer.PENALTIES
MAP_KEYS = range(10, 20)
BOOTSTRAP_SEED = 20261120
FIXED_ALPHA = {"nki": 10.0, "bnu1": 1.0}
EQUAL_EDGES = {"nki": 177, "bnu1": 1291}
SCORE_COLUMNS = ["source_r2", "noise_r2", "repeat_r2", "excess_r2", "shift_pct",
                 "noise_loss", "projected_loss", "residual_cross_term"]


def cases(cohort, graphs):
    for key in MAP_KEYS:
        yield (f"map_{key}", "native", key, 0.90, graphs)
    for radius in (0.70, 1.00):
        yield (f"radius_{radius:.2f}", "native", 10, radius, graphs)
    yield ("log1p", "log1p", 10, 0.90, np.log1p(graphs.astype(float)))
    yield ("binary", "binary", 10, 0.90, (graphs > 0).astype(float))
    yield ("equal_edges", "equal_edges", 10, 0.90,
           threshold(graphs, EQUAL_EDGES[cohort], graphs.shape[1]))


def weights(model, alpha):
    if model.scale == 0:
        return np.zeros_like(model.proj)
    return model.weights(alpha)


def evaluate_pair(model, x_source, x_target, y, alpha):
    slopes = weights(model, alpha)
    y_var = y.var(axis=0)
    displacement = x_target - x_source
    delta_power = np.mean(displacement**2)
    projected = displacement @ slopes
    e = (x_source - model.mx) @ slopes + model.my - y
    repeat_e = (x_target - model.mx) @ slopes + model.my - y
    source_r2 = np.mean(1 - np.mean(e**2, axis=0) / y_var)
    repeat_r2 = np.mean(1 - np.mean(repeat_e**2, axis=0) / y_var)
    expected_loss = np.mean(delta_power * np.sum(slopes**2, axis=0) / y_var)
    projected_loss = np.mean(np.mean(projected**2, axis=0) / y_var)
    cross = np.mean(2 * np.mean(projected * e, axis=0) / y_var)
    noise_r2 = source_r2 - expected_loss
    excess = noise_r2 - repeat_r2
    np.testing.assert_allclose(excess, projected_loss + cross - expected_loss,
                               atol=2e-11, rtol=2e-12)
    return dict(source_r2=source_r2, noise_r2=noise_r2, repeat_r2=repeat_r2,
                excess_r2=excess, shift_pct=100 * np.sqrt(delta_power / model.scale)
                if model.scale > 0 else np.nan, noise_loss=expected_loss,
                projected_loss=projected_loss, residual_cross_term=cross)


def evaluate_case(cohort, ids, setting, representation, map_key, radius, graphs):
    n = len(ids)
    mapping = pilot.input_map(graphs.shape[-1], map_key)
    models, sources, targets, losses = [], [], [], []
    case = dict(cohort=cohort, setting=setting, representation=representation, map_key=map_key,
                radius=radius)
    for i in range(n):
        source = pilot.reservoir(graphs[0, i], radius)
        target = pilot.reservoir(graphs[1, i], radius)
        source_stream = 10000 + i
        eval_stream = source_stream + n
        training = pilot.simulate(source, mapping, source_stream, 0, pilot.FIT_N)
        model = pilot.RidgeFamily.from_design(training)
        xv, yv = pilot.simulate(target, mapping, eval_stream, 1, pilot.VALID_N)
        xv = xv - model.mx
        yv = yv - model.my
        loss = np.array([np.mean((xv @ weights(model, alpha) - yv)**2) for alpha in GRID])
        if not np.isfinite(loss).all():
            raise ValueError(f"Nonfinite validation error for {cohort} {setting} {ids[i]}")
        models.append(model)
        sources.append(source)
        targets.append(target)
        losses.append(loss)
    losses = np.stack(losses)
    validation = pd.DataFrame(losses, index=ids, columns=GRID).rename_axis("target_id")
    validation = validation.reset_index().melt(id_vars="target_id", var_name="alpha",
                                                value_name="validation_mse")
    for key, value in case.items():
        validation[key] = value
    rows = []
    for i, (model, source, target) in enumerate(zip(models, sources, targets)):
        other = np.arange(n) != i
        selected = GRID[np.argmin(losses[other].mean(axis=0))]
        eval_stream = 10000 + i + n
        xs, source_y = fresh_test(source, mapping, eval_stream)
        xt, target_y = fresh_test(target, mapping, eval_stream)
        np.testing.assert_array_equal(source_y, target_y)
        y = source_y[:, :len(pilot.DELAYS)]
        rules = {"retuned": selected, "native_fixed": FIXED_ALPHA[cohort], "light": .001}
        scores = {a: evaluate_pair(model, xs, xt, y, a) for a in set(rules.values())}
        for rule, alpha in rules.items():
            rows.append(dict(**case, rule=rule, target_id=ids[i], alpha=alpha,
                             constant_source=model.scale == 0, **scores[alpha]))
    return pd.DataFrame(rows), validation


def summarize(group):
    group = group.sort_values("target_id")
    n = len(group)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(n, size=(10000, n))
    row = dict(n_people=n, n_positive_excess=int((group.excess_r2 > 0).sum()),
               n_positive_repeat=int((group.repeat_r2 > 0).sum()),
               n_positive_noise=int((group.noise_r2 > 0).sum()),
               n_displacement_defined=int(group.shift_pct.notna().sum()))
    for name in SCORE_COLUMNS:
        values = group[name].to_numpy()
        row[name] = np.nanmean(values)
        lo, hi = np.percentile(np.nanmean(values[draws], axis=1), [2.5, 97.5])
        row[f"{name}_low"] = lo
        row[f"{name}_high"] = hi
    return row


def summarize_all(people):
    rows = []
    groups = ["cohort", "setting", "rule"]
    for keys, group in people.groupby(groups, sort=False):
        rows.append(dict(zip(groups, keys), **summarize(group)))
    alt = people[people.representation.eq("native") & people.map_key.ne(10)]
    means = alt.groupby(["cohort", "rule", "target_id"], as_index=False)[SCORE_COLUMNS].mean()
    if not (alt.groupby(["cohort", "rule", "target_id"]).size() == len(MAP_KEYS)-1).all():
        raise ValueError("Expected all nine new maps for each participant and rule.")
    for (cohort, rule), group in means.groupby(["cohort", "rule"], sort=False):
        rows.append(dict(cohort=cohort, setting="mean_nine_new_maps", rule=rule, **summarize(group)))
    return pd.DataFrame(rows)


def compare_archive(people):
    root = transfer.ROOT / "results"
    primary = pd.read_csv(root / "portability-path" / "shift_people.csv", dtype={"source_id": str})
    primary = primary[(primary.source_set == "first_visit") & (primary.rung == "between_visit")
                      & (primary.task == "linear")].rename(columns={"source_id":"target_id"})
    original = people[people.setting.eq("map_10") & people.rule.isin(["native_fixed", "light"])]
    merged = original.merge(primary, on=["cohort","target_id","alpha"], validate="one_to_one",
                            how="outer", indicator=True, suffixes=("", "_archive"))
    assert (merged._merge == "both").all() and len(merged) == 2 * 82
    pairs = {"source_r2":"source_r2_archive", "noise_r2":"isotropic_r2", "repeat_r2":"target_r2",
             "projected_loss":"actual_output_shift", "noise_loss":"expected_isotropic_loss",
             "residual_cross_term":"error_cross_term", "shift_pct":"shift_pct_archive"}
    errors = {}
    for column, archived in pairs.items():
        errors[column] = float(np.max(np.abs(merged[column] - merged[archived])))
        np.testing.assert_allclose(merged[column], merged[archived], atol=2e-10, rtol=2e-12)
    fixed = people[(people.map_key == 10) & (people.rule == "native_fixed")].copy()
    fixed["scenario"] = fixed.setting.replace({"map_10":"native"})
    old = pd.read_csv(root / "pipeline-robustness" / "participant_scores.csv",
                      dtype={"target": str}).rename(columns={"target":"target_id"})
    checked = fixed.merge(old[["cohort", "scenario", "target_id", "own_r2"]],
                          on=["cohort","scenario","target_id"], how="left", validate="one_to_one")
    assert len(checked) == 6 * 82 and checked.own_r2.notna().all()
    errors["fixed_variants_repeat_r2"] = float((checked.own_r2 - checked.repeat_r2).abs().max())
    np.testing.assert_allclose(checked.repeat_r2, checked.own_r2, atol=1e-10, rtol=1e-11)
    original_retuned = people[people.setting.eq("map_10") & people.rule.eq("retuned")]
    assert original_retuned.alpha.eq(original_retuned.cohort.map(FIXED_ALPHA)).all()
    return errors


def main():
    frames, validations, maps = [], [], []
    for cohort in ("nki", "bnu1"):
        graphs, ids, _ = read_graphs(cohort)
        for setting, representation, key, radius, matrices in cases(cohort, graphs):
            print(f"{cohort} {setting}", flush=True)
            frame, validation = evaluate_case(cohort, ids, setting, representation, key, radius, matrices)
            frames.append(frame)
            validations.append(validation)
        for key in MAP_KEYS:
            inputs, _ = pilot.input_map(graphs.shape[-1], key)
            for region in np.flatnonzero(inputs):
                maps.append(dict(cohort=cohort, map_key=key, atlas_index_zero_based=int(region),
                                 input_weight=inputs[0, region]))
    people = pd.concat(frames, ignore_index=True)
    assert len(people) == 82 * 15 * 3
    errors = compare_archive(people)
    summary = summarize_all(people)
    penalties = people.groupby(["cohort","setting","rule","alpha"]).size().rename("n_people").reset_index()
    names = dict(people=people, summary=summary, penalties=penalties, input_maps=pd.DataFrame(maps))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, table in names.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
    pd.concat(validations, ignore_index=True).to_csv(
        OUT / "validation.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    checks = dict(n_people={"nki":25, "bnu1":57}, settings_per_cohort=15,
                  distinct_evaluated_pairs=82, setting_participant_combinations=15*82,
                  primary_validation_rule="Leave evaluated identity out of transfer validation, average other people in the same cohort",
                  candidate_relative_penalties=GRID.tolist(), bootstrap_seed=BOOTSTRAP_SEED,
                  primary_map=10, native_radius=.90, tested_native_maps=list(MAP_KEYS),
                  original_primary_penalties_recovered=True,
                  validated_original_shift_rows=164, validated_fixed_sensitivity_rows=492,
                  max_absolute_archive_difference=errors,
                  constant_source_pair_settings=int(people[people.rule.eq("retuned")].constant_source.sum()),
                  max_abs_excess_identity_residual=float(np.abs(people.excess_r2 - people.projected_loss
                                                                - people.residual_cross_term
                                                                + people.noise_loss).max()))
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf8")
    print(summary.loc[summary.rule.eq("retuned"), ["cohort","setting","source_r2","noise_r2",
          "repeat_r2","excess_r2","excess_r2_low","excess_r2_high"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
