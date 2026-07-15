"""Build globally defined parental-income deciles aligned to model panels.

The canonical input is ``Model/Inputs/real_data/parental_income_by_pubid.npz``.
The resulting decile is matched by PUBID to every superfeasible model period.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import DIR, RDATA


def build_parental_income_deciles(source_path=None):
    if source_path is None:
        source_path = Path(RDATA("parental_income_by_pubid.npz"))
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing canonical parental-income input: {source_path}. "
            "Run sync_parental_income_input.py once from the local project."
        )

    if source_path.suffix.lower() == ".npz":
        with np.load(source_path, allow_pickle=False) as source:
            data = pd.DataFrame(
                {
                    "PUBID": np.asarray(source["pubid"], dtype=np.int64),
                    "aveparinc": np.asarray(source["aveparinc"], dtype=float),
                }
            )
    elif source_path.suffix.lower() == ".dta":
        data = pd.read_stata(
            source_path,
            columns=["PUBID", "aveparinc"],
            convert_categoricals=False,
        )
    else:
        raise ValueError("Parental-income source must be .npz or .dta.")
    data = data.dropna(subset=["PUBID", "aveparinc"]).drop_duplicates("PUBID")
    if data["PUBID"].duplicated().any():
        raise ValueError("Parental-income source must contain one row per PUBID.")

    cutpoints = data["aveparinc"].quantile(np.arange(0.1, 1.0, 0.1)).to_numpy(float)
    decile = np.searchsorted(cutpoints, data["aveparinc"].to_numpy(float), side="left") + 1
    pubid = data["PUBID"].to_numpy(np.int64)
    order = np.argsort(pubid)
    pubid = pubid[order]
    decile = decile[order].astype(np.int64)

    mapping_path = RDATA("parental_income_decile_by_pubid.npz")
    np.savez_compressed(mapping_path, pubid=pubid, decile=decile, cutpoints=cutpoints)

    lookup = dict(zip(pubid.tolist(), decile.tolist()))
    written = []
    for period in range(1, 10):
        invariant_path = Path(RDATA(f"invariant_state_superfeasible_t{period}.npy"))
        if not invariant_path.exists():
            continue
        period_pubid = np.load(invariant_path)[:, 0].astype(np.int64)
        missing = sorted(set(period_pubid).difference(lookup))
        if missing:
            raise ValueError(
                f"Period {period} contains {len(missing)} PUBIDs without parental income."
            )
        aligned = np.fromiter((lookup[value] for value in period_pubid), dtype=np.int64)
        output = Path(RDATA(f"parental_income_decile_superfeasible_t{period}.npy"))
        np.save(output, aligned)
        written.append(output)

        # The cleaned annual disbursement is stored in the full period panel.
        # Align it by PUBID rather than reconstructing it from debt stocks.
        full_invariant = Path(RDATA(f"invariant_state_t{period}.npy"))
        full_flow = Path(RDATA(f"loanflow_t{period}.npy"))
        if full_invariant.exists() and full_flow.exists():
            full_pubid = np.load(full_invariant)[:, 0].astype(np.int64)
            flow = np.asarray(np.load(full_flow), dtype=float).reshape(-1)
            if len(full_pubid) != len(flow) or len(np.unique(full_pubid)) != len(full_pubid):
                raise ValueError(f"Period {period} loan-flow panel is not one row per PUBID.")
            flow_lookup = dict(zip(full_pubid.tolist(), flow.tolist()))
            missing_flow = sorted(set(period_pubid).difference(flow_lookup))
            if missing_flow:
                raise ValueError(
                    f"Period {period} contains {len(missing_flow)} PUBIDs without loan flow."
                )
            aligned_flow = np.fromiter(
                (flow_lookup[value] for value in period_pubid), dtype=float
            )
            flow_output = Path(RDATA(f"loanflow_superfeasible_t{period}.npy"))
            np.save(flow_output, aligned_flow)
            written.append(flow_output)

    print(f"Saved global PUBID mapping: {mapping_path}")
    print(f"Saved {len(written)} aligned period files under {DIR['MODEL_REALDATA']}")
    return mapping_path, written


if __name__ == "__main__":
    build_parental_income_deciles()
