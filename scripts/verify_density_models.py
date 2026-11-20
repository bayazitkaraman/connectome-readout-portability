"""Independently reproduce the primary crossed-effects fits with statsmodels."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


FOLDER = Path(__file__).resolve().parents[1] / "results/density-controls/observational"


def run():
    data = pd.read_csv(FOLDER/"matched_pairs_by_budget.csv",dtype={"subject":str,"other_subject":str})
    reported = pd.read_csv(FOLDER/"crossed_models.csv")
    audits = []
    for cohort in ("nki","bnu1"):
        source = data.query("cohort == @cohort and mix_rule=='equal_donor' and budget==8").copy()
        source["group"] = 1
        # Scale y for the optimizer. Convert the estimated slopes back into R² units.
        source["scaled_gain"] = 100 * (source.pool_plus_own-source.pool_plus_other)
        source["density"] = (abs(source.n_target_second-source.n_other_first)
                             -abs(source.n_target_second-source.n_own_first))/3403
        source["raw"] = source.own_raw - source.other_raw
        grouped = source.groupby("subject")[["raw","gain","density"]].mean().rank()
        nuisance = np.stack([np.ones(len(grouped)),grouped.density],axis=1)
        residuals = grouped[["raw","gain"]]-nuisance@np.linalg.lstsq(nuisance,grouped[["raw","gain"]],rcond=None)[0]
        partial = np.corrcoef(residuals.to_numpy().T)[0,1]
        partial_table = pd.read_csv(FOLDER/"target_partial_rank.csv")
        expected = partial_table.query("cohort==@cohort and mix_rule=='equal_donor'").partial_rho.iloc[0]
        np.testing.assert_allclose(partial,expected,atol=1e-12)
        for name, formula in (("raw","scaled_gain ~ raw"), ("raw_plus_density","scaled_gain ~ raw + density")):
            model = smf.mixedlm(formula,source,groups="group",re_formula="0",
                                vc_formula={"target":"0 + C(subject)","donor":"0 + C(other_subject)"})
            with warnings.catch_warnings(record=True) as notes:
                result = model.fit(reml=True,method="powell",disp=False)
            assert result.converged
            rows = reported.query("cohort==@cohort and mix_rule=='equal_donor' and model==@name")
            diff = max(abs(result.fe_params[r.term]/100-r.mixed_slope) for r in rows.itertuples())
            if diff > 1e-5:
                raise ValueError(f"Alternate mixed model fit discrepancy {diff}")
            audits.append(dict(cohort=cohort,model=name,max_absolute_slope_difference=diff,
                               alternate_software="statsmodels MixedLM REML Powell",fit_converged=result.converged,
                               independent_partial_rank=partial,warnings=[str(w.message) for w in notes]))
    (FOLDER/"independent_check.json").write_text(json.dumps(audits,indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audits,indent=2))


if __name__=="__main__":
    run()
