import glob
import os
import subprocess
import sys
import constants


def run_batch_suite(
    target_folder: str,
    cpus: int | None = None,
    debug: bool = False,
    pref: float = constants.Pref,
    wref: float = constants.Wref,
    vref: float = constants.Vref,
    results: str = constants.results_type,
) -> dict:
    """Sequentially run pyBEM (main.py) on all *.inp files in target_folder."""
    target_dir = os.path.abspath(target_folder)
    if not os.path.isdir(target_dir):
        print(f" [!] ERROR: Directory '{target_dir}' does not exist.")
        return {"total": 0, "passed": 0, "failed": []}

    inp_files = sorted(glob.glob(os.path.join(target_dir, "*.inp")))
    if not inp_files:
        print(f" [!] No *.inp files found in: {target_dir}")
        return {"total": 0, "passed": 0, "failed": []}

    common_flags = [
        f"--Pref={pref}",
        f"--Wref={wref}",
        f"--Vref={vref}",
        f"--results={results}",
    ]
    if cpus is not None:
        common_flags.append(f"--cpus={cpus}")
    if debug:
        common_flags.append("--debug")

    pybem_main = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")

    print("\n" + "=" * 60)
    print(f" Starting pyBEM Batch Suite ({len(inp_files)} Jobs Found)")
    print(f" Target Folder: {target_dir}")
    print("=" * 60)

    passed_jobs = 0
    failed_jobs = []

    for idx, inp_path in enumerate(inp_files, start=1):
        filename = os.path.basename(inp_path)
        print(f"\n[{idx}/{len(inp_files)}] Running: {filename}...")
        print("-" * 60)

        cmd = [sys.executable, pybem_main, inp_path] + common_flags
        result = subprocess.run(cmd, cwd=target_dir, check=False)

        if result.returncode == 0:
            print(f" [OK] PASSED: {filename}")
            passed_jobs += 1
        else:
            print(f" [!] FAILED: {filename} (Exit Code: {result.returncode})")
            failed_jobs.append(filename)

    print("\n" + "=" * 60)
    print(" Batch Suite Execution Complete!")
    print(f" Total Processed: {len(inp_files)} | Passed: {passed_jobs} | Failed: {len(failed_jobs)}")
    if failed_jobs:
        print(f" Failed Files: {', '.join(failed_jobs)}")
    print("=" * 60 + "\n")

    return {
        "total": len(inp_files),
        "passed": passed_jobs,
        "failed": failed_jobs,
    }