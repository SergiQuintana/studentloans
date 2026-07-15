"""Create and optionally send the canonical parental-income model input.

This is a one-time bridge from the data-cleaning output to Model/Inputs. It
does not run estimation and does not create target moments.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from config import RDATA


def create_input(source_path=None):
    project_root = Path(__file__).resolve().parents[3]
    source = (
        Path(source_path)
        if source_path is not None
        else project_root / "Data" / "temporary" / "demographic_invariant.dta"
    )
    if not source.exists():
        raise FileNotFoundError(f"Missing data-cleaning output: {source}")
    data = pd.read_stata(
        source,
        columns=["PUBID", "aveparinc"],
        convert_categoricals=False,
    )
    data = data.dropna(subset=["PUBID", "aveparinc"]).drop_duplicates("PUBID")
    if data["PUBID"].duplicated().any():
        raise ValueError("Parental-income source must contain one row per PUBID.")
    order = np.argsort(data["PUBID"].to_numpy())
    output = Path(RDATA("parental_income_by_pubid.npz"))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        pubid=data["PUBID"].to_numpy(np.int64)[order],
        aveparinc=data["aveparinc"].to_numpy(float)[order],
    )
    print(f"Created canonical model input: {output}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    parser.add_argument("--ssh", default=None, help="Server SSH target, e.g. 10.1.4.56")
    parser.add_argument(
        "--server-real-data",
        default="/home/ubuntu/work/Model/Inputs/real_data",
    )
    args = parser.parse_args()
    output = create_input(args.source)
    if args.ssh:
        destination = f"{args.ssh}:{args.server_real_data.rstrip('/')}/"
        subprocess.run(["scp", str(output), destination], check=True)
        print(f"Sent {output.name} to {destination}")


if __name__ == "__main__":
    main()
