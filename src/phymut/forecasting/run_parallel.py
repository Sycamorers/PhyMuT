import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

from phymut.paths import output_dir, PROJECT_ROOT

# List of models to run in parallel
# MODELS = ["lstm", "gru", "transformer", "nbeats", "rnn", "s2s", "tcn", "mamba"]
MODELS = ["lstm"]

# Path to your main script
BASE_SCRIPT = Path(__file__).resolve().with_name("run_forecast.py")

# Output log file pattern
LOG_PATTERN = "new_cft_{}.txt"

# Directory for temp scripts
TEMP_DIR = output_dir("forecasting", "tmp_scripts")

# Directory for log files
LOG_DIR = output_dir("forecasting", "logs")

# Function to create a temp script for a single model
def make_temp_script(model):
    """Create a temporary copy of ``new_run_all.py`` for a single model.

    Besides restricting ``MODELS`` to ``model`` this function also tweaks the
    hyperparameter grid and a few search options to enlarge the exploration
    space suggested by the previous analysis.  The original ``new_run_all.py``
    remains unchanged; all modifications are applied to the temporary copy.
    """

    temp_script = TEMP_DIR / f"run_forecast_{model}.py"
    sys_path_fix = (
        "import sys\n"
        f"sys.path.insert(0, r\"{PROJECT_ROOT / 'src'}\")\n"
    )

    # Flags for skipping blocks while rewriting lines (e.g. the original grid)
    skip_grid = False

    with open(BASE_SCRIPT, "r") as fin, open(temp_script, "w") as fout:
        fout.write(sys_path_fix)
        for line in fin:
            stripped = line.strip()

            if skip_grid:
                if stripped.startswith("}"):
                    skip_grid = False
                continue

            if stripped.startswith("MODELS ="):
                fout.write(f"MODELS = ['{model}']\n")
            elif stripped.startswith("SEQ_LENS ="):
                # Try a slightly wider range of sequence lengths
                fout.write("SEQ_LENS = [3, 4, 5, 6, 7]\n")
            elif stripped.startswith("EPOCHES ="):
                # Run a longer final training phase
                fout.write("EPOCHES = [400, 600]\n")
            elif stripped.startswith("grid ="):
                # Replace the entire grid block with a larger search space
                fout.write("                    grid = {\n")
                fout.write("                        'lr': [1e-2, 1e-3, 5e-4],\n")
                fout.write("                        'hid_dim': [32, 64, 128, 256],\n")
                fout.write("                        'num_layers': [1, 2, 3]\n")
                fout.write("                    }\n")
                skip_grid = True
            else:
                fout.write(line)
    return temp_script

# Function to run a single model script and save output
def run_model(model):
    temp_script = make_temp_script(model)
    log_file = LOG_DIR / LOG_PATTERN.format(model)
    rel_temp_script = os.path.relpath(temp_script, PROJECT_ROOT)
    with open(log_file, "w") as f:
        subprocess.run(
            ["python", rel_temp_script],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
        )
    # Optionally, remove the temp script after running
    # os.remove(temp_script)

def main():
    # You can adjust max_workers based on your GPU/CPU
    with ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
        futures = [executor.submit(run_model, model) for model in MODELS]
        for future in futures:
            future.result()  # Wait for all to finish

    print("All models finished. Check the log files for output.")


if __name__ == "__main__":
    main()

# USAGE:
# python run_parallel_models.py
# This will run each model in parallel, saving output to out_new_fixed_cft_<model>.txt 
