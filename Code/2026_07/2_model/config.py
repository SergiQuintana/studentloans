import os
from pathlib import Path

# -----------------------------
# Root: change once, affects all
# -----------------------------
# Default roots:
# - runner/server: MODEL_ROOT or REPL_ROOT, set by runinserver.py
# - manual server run: sibling /home/ubuntu/work/Model, outside studentloans
# - local laptop run: project-local Model folder
def _resolve_model_root() -> str:
    explicit_root = os.environ.get("MODEL_ROOT") or os.environ.get("REPL_ROOT")
    if explicit_root:
        return str(Path(explicit_root).expanduser())

    project_root = Path(__file__).resolve().parents[3]
    project_model = project_root / "Model"
    sibling_model = project_root.parent / "Model"

    if project_root.name.lower() == "studentloans" and sibling_model.exists():
        return str(sibling_model)

    if project_model.exists():
        return str(project_model)

    if sibling_model.exists():
        return str(sibling_model)

    return str(project_model)


REPL_ROOT = _resolve_model_root()
#REPL_ROOT = r"C:\Users\S.Quintana\Dropbox\PhD\Projects\Papers\1_financial_constraints\Model"
# -----------------------------
# Canonical directories
# -----------------------------
DIR = {
    "MODEL":                    os.path.join(REPL_ROOT),
    "MODEL_INPUTS":             os.path.join(REPL_ROOT, "Inputs"),
    "MODEL_FUNCOEF":            os.path.join(REPL_ROOT, "Inputs", "function_coefficients"),
    "MODEL_REALDATA":           os.path.join(REPL_ROOT, "Inputs", "real_data"),
    "MODEL_CONTINUATION":       os.path.join(REPL_ROOT, "Inputs", "continuation"),
    "MODEL_CONTINUATION_FINAL": os.path.join(REPL_ROOT, "Inputs", "continuation_final"),
    "MODEL_OUTPUT":             os.path.join(REPL_ROOT, "Output"),
    "MODEL_ESTIMATES":          os.path.join(REPL_ROOT, "Estimates"),
    "MODEL_LIKELIHOOD":         os.path.join(REPL_ROOT, "Output", "likelihood"),
}

# -----------------------------
# Joiners (never use chdir)
# -----------------------------
def OUT(*parts):    return os.path.join(DIR["MODEL_OUTPUT"], *map(str, parts))
def INP(*parts):    return os.path.join(DIR["MODEL_INPUTS"], *map(str, parts))
def FUN(*parts):    return os.path.join(DIR["MODEL_FUNCOEF"], *map(str, parts))
def RDATA(*parts):  return os.path.join(DIR["MODEL_REALDATA"], *map(str, parts))
def CONT(*parts):   return os.path.join(DIR["MODEL_CONTINUATION"], *map(str, parts))
def EST(*parts):    return os.path.join(DIR["MODEL_ESTIMATES"], *map(str, parts))
def LIK(*parts):    return os.path.join(DIR["MODEL_LIKELIHOOD"], *map(str, parts))
def STATES(*parts): return OUT("states", *map(str, parts))
# -----------------------------
# Convenience helpers
# -----------------------------
def ENSURE_DIR(path: str) -> str:
    """mkdir -p for a directory, returns the directory path."""
    os.makedirs(path, exist_ok=True)
    return path

def ENSURE_PARENTS(path: str) -> str:
    """mkdir -p for the parent of a file path, returns the file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def MK(*parts: str) -> str:
    """OUT-joined file path with parents ensured (good for np.save, to_csv, etc.)."""
    return ENSURE_PARENTS(OUT(*parts))

def ENSURE_DEFAULT_TREE(T: int = 10) -> None:
    """Create the common output subtrees (idempotent). No subfolders for states/."""
    # Top-level outputs we rely on
    for key in ("MODEL_OUTPUT", "MODEL_ESTIMATES", "MODEL_LIKELIHOOD"):
        ENSURE_DIR(DIR[key])
    # Single states/ folder (no per-period subfolders)
    ENSURE_DIR(OUT("states"))
    # Single states/ folder (no per-period subfolders)
    ENSURE_DIR(OUT("types"))
    # Per-period buckets for model artifacts (these DO have per-period subfolders)
    periodic = ("vjt", "vjt_nog", "vjt_conter", "evt", "evt_nog", "evt_conter")
    for sub in periodic:
        for k in range(1, T + 1):
            ENSURE_DIR(OUT(sub, str(k)))
    # Other flat buckets under Output (no per-period subfolders)
    for sub in ("choice", "state", "epsilon", "welfare", "grad_prob", "likelihood", "continuation", "states"):
        ENSURE_DIR(OUT(sub))

# -----------------------------
# Legacy aliases (zero-code-change)
# -----------------------------
# Some scripts refer to these variable names directly. Importing them
# from config keeps those scripts working without edits.
pathfunctions = DIR["MODEL_FUNCOEF"]
path          = DIR["MODEL_REALDATA"]
pathcont      = DIR["MODEL_CONTINUATION"]
pathest       = DIR["MODEL_ESTIMATES"]
pathlik       = DIR["MODEL_LIKELIHOOD"]
dir           = DIR["MODEL_OUTPUT"]

path_estimates = DIR["MODEL_ESTIMATES"] 
