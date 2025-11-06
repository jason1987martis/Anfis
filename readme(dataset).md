README — dataset & run instructions

This short README explains how to run the example ANFIS project and how to change the dataset and training parameters.

1) Requirements
- Use the project's virtual environment (created at `.venv` inside the repo). Python executable (example):
  D:/code_space/Anfis/.venv/Scripts/python.exe
- Required packages (installed into the venv): scikit-fuzzy, scikit-learn, scipy, matplotlib, numpy

2) Run the example
- From PowerShell in the repository root (D:\code_space\Anfis) you can run:

  D:/code_space/Anfis/.venv/Scripts/python.exe main.py

- The `main.py` script will:
  - Load a CSV from `input/` (default: `input/iris.csv`)
  - Train the ANFIS model using `myANFIS.py`
  - Show plots (one window at a time) and print final RMSE and R²

3) Change the dataset
- `main.py` loads the CSV with:
  data = np.loadtxt('input/iris.csv', delimiter=',')

- To use another dataset, either:
  - Replace `input/iris.csv` with your CSV file (same name), or
  - Edit `main.py` and change the path string to your file, e.g.:

    data = np.loadtxt('input/your_file.csv', delimiter=',')

- CSV format requirements:
  - Features must be in the first columns and the target (label) must be the last column.
  - Example shape: N rows x (M+1) columns, where M = number of features.

4) Change training parameters
- Open `main.py` and edit these variables near the top:

  epoch_n = 10        # number of training epochs
  mf = 2              # number of membership functions per input
  step_size = 0.1     # learning rate
  decrease_rate = 0.9 # adaptive decrease factor for step size
  increase_rate = 1.1 # adaptive increase factor for step size

- Increasing `epoch_n` gives more training iterations. Increasing `mf` grows the rule base exponentially (use with caution).

5) Headless / remote servers (no GUI)
- If the machine has no display, change the matplotlib backend in `main.py` before importing pyplot:

  import matplotlib
  matplotlib.use('Agg')

- Then modify plotting calls to save figures instead of showing them (use `plt.savefig('out.png')`), or rely on the existing plotting functions but they will not open windows under `Agg`.

6) Troubleshooting
- Missing module error -> install into venv, for example:

  D:/code_space/Anfis/.venv/Scripts/python.exe -m pip install scikit-fuzzy scikit-learn scipy matplotlib numpy

- Plots hang / don't appear -> close the displayed plot window to continue, or switch to `Agg` backend to save images instead of opening windows.
- CSV shape mismatch -> verify `inputs = data[:, :-1]` and `output = data[:, -1:]` correspond to your file.

7) Files changed/added (for reviewers)
- `myANFIS.py` — bug fixes (initialized variables), plotting improvements (create & close figures), stability tweaks.
- `main.py` — example runner that shows how to load data and call `myanfis`.
- `readme(dataset).md` — this file: how to run and change dataset/parameters.

If you'd like, I can also:
- Add an automated test that runs `myanfis` for one epoch on a tiny synthetic dataset.
- Add a non-interactive mode to `main.py` that saves plots to `output/` automatically.

