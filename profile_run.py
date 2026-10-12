"""
Author: Huy Hiep Nguyen
Copyright (c) 2026 Huy Hiep Nguyen
"""
import cProfile
import pstats
from main import main

if __name__ == "__main__":
    prof_file = "profile.prof"

    cProfile.run("main()", prof_file)

    # Print top results to console
    stats = pstats.Stats(prof_file).strip_dirs().sort_stats("cumtime")
    stats.print_stats(40)

    # Optional: Open visualization in browser with snakeviz
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "snakeviz", prof_file], check=False)
