import argparse
import os
import glob
import subprocess
import sys
import version
import constants

def run_test_suite():
    # --- 1. CONFIGURE CLI PARSER ---
    parser = argparse.ArgumentParser(
        description=f"pyBEM {version.__version__} Batch Suite Runner — Sequentially runs pyBEM on all *.inp files in a directory.",
        exit_on_error=False,
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Path to directory containing *.inp files (default: None)",
    )
    parser.add_argument(
        "--cpus", type=int, default=1, 
        help="Number of CPUs to use for parallel Freqs solve.  Defaults: 1 CPU but multi-thread for matrix solve. pyBEM sets this automatically per machine specs, in order to minimise racing conditions."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging to '*_debug.log' filefor all suite runs"
    )
    parser.add_argument(
        "--Pref", type=float, default=constants.Pref, 
        help=f"dB PRESSURE reference (default: {constants.Pref}MPa)",
    )
    parser.add_argument(
        "--Wref", type=float, default=constants.Wref, 
        help=f"dB POWER reference (default: {constants.Wref}mW)",
    )

    args = parser.parse_args()

    # --- 2. VALIDATE TARGET DIRECTORY ---
    target_dir = os.path.abspath(args.folder)
    if not os.path.isdir(target_dir):
        print(f" [!] ERROR: Directory '{target_dir}' does not exist.")
        sys.exit(1)

    # Search and sort all .inp files alphabetically (e.g., bc1, bc2, bc3...)
    inp_files = sorted(glob.glob(os.path.join(target_dir, "*.inp")))

    if not inp_files:
        print(f" [!] No *.inp files found in: {target_dir}")
        sys.exit(0)

    # --- 3. BUILD COMMON CLI FLAGS ---
    common_flags = [
        f"--cpus={args.cpus}",
        f"--Pref={args.Pref}",
        f"--Wref={args.Wref}",
    ]
    if args.debug:
        common_flags.append("--debug")

    print("\n" + "=" * 60)
    print(f" Starting pyBEM Test Suite Execution ({len(inp_files)} Jobs Found)")
    print(f" Target Folder: {target_dir}")
    print("=" * 60)


    # Path to main.py relative to run_suite.py
    # Assuming run_suite.py is in pyBEM_code folder
    pybem_main = os.path.join(os.path.dirname(__file__), "main.py")

    # --- 4. SEQUENTIAL EXECUTION LOOP ---
    passed_jobs = 0
    failed_jobs = []

    for idx, inp_path in enumerate(inp_files, start=1):
        filename = os.path.basename(inp_path)
        print(f"\n[{idx}/{len(inp_files)}] Running: {filename}...")
        print("-" * 60)

        # Directly run main.py using the current active Python interpreter
        cmd = [sys.executable, pybem_main, inp_path] + common_flags

        # subprocess.run blocks until pyBEM finishes the current model
        result = subprocess.run(cmd, cwd=target_dir, check=False)

        if result.returncode == 0:
            print(f" [OK] PASSED: {filename}")
            passed_jobs += 1
        else:
            print(f" [!] FAILED: {filename} (Exit Code: {result.returncode})")
            failed_jobs.append(filename)

    # --- 5. SUMMARY REPORT ---
    print("\n" + "=" * 60)
    print(" Test Suite Execution Complete!")
    print(f" Total Processed: {len(inp_files)} | Passed: {passed_jobs} | Failed: {len(failed_jobs)}")
    if failed_jobs:
        print(f" Failed Files: {', '.join(failed_jobs)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_test_suite()