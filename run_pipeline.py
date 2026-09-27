from pathlib import Path
import runpy, os
root=Path(__file__).resolve().parents[1]; os.chdir(root)
runpy.run_path("src/load_data.py"); runpy.run_path("src/visualize.py")
print("Week 2 pipeline completed.")
