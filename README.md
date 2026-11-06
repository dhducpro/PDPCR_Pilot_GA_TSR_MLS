# PDP-CR Pilot

GA, tabu search (TSR), and multi-start local search (MLS) for cooperative pickup and delivery. The objective is to minimize makespan with 10 robots.

The default experiment uses 9 instances: LC1, LR1, and LRC1 with 50, 100, and 200 tasks. Data are adapted from the [Li-Lim benchmark](https://www.sintef.no/projectweb/top/pdptw/li-lim-benchmark/): each complete pickup-delivery pair becomes one task, with Euclidean processing and travel times.

## Run

```bash
python -m pip install -r requirements.txt
jupyter notebook run_pilot.ipynb
```

Open the notebook from the project folder. Edit the configuration cell, then run the experiment cells. Each method uses 5 seeds with a limit of 10, 30, or 60 minutes per seed for 50, 100, or 200 tasks.

Set `DEMO_MODE = True` for a short offline demo. Choose a new `RESULT_NAME` when changing settings or solver code.

## Google Colab

[Open in Colab](https://colab.research.google.com/github/dhducpro/PDPCR_Pilot_GA_TSR_MLS/blob/main/run_pilot_colab.ipynb)

Run the cells in order and allow Google Drive access. GA, TSR and MLS run on Google's CPU runtime; CPLEX is excluded. Results are saved in `MyDrive/PDPCR_Pilot_GA_TSR_MLS_colab/results/`. Rerun with the same settings to reuse completed seeds. With two workers, the full campaign has about 45 hours of scheduled solver time, so multiple sessions may be needed. An interrupted seed starts over. Keep the same result name after a disconnect; setup restores the saved experiment settings before preparing data. The run cell updates older Drive checkouts and can recover missing configuration metadata by checking completed seed files; older records cannot verify the original population, worker count, or search options. Use a different result name when changing sizes, families, workers or budgets.

## CPLEX

The last cell runs CPLEX once per instance and compares it with the saved GA best-of-5. Set `CPLEX_EXE` to your installed `cplex.exe`, then run that cell. It uses one thread and the same time limit per run. CPLEX must be installed separately.

## Results

Results are saved in `results/<RESULT_NAME>/`:

- `comparison.csv`: GA, TSR, and MLS best-of-5 and percentage improvements.
- `runs.csv` and `summary.csv`: individual seeds and summary statistics.
- `best/` and `seeds/`: saved solutions and search histories.
- `cplex_vs_ga/`: CPLEX results, bounds, gaps, and the comparison with GA.

Best-of-5 means the lowest makespan across five seeds. Positive GA improvement means GA is better. RPD uses the best solution found in the experiment as its reference; it is not a certified optimality gap. CPLEX results include whether optimality was proved.
