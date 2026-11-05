# PDP-CR Pilot: GA, TSR, MLS and CPLEX

A reproducible pilot study for the cooperative pickup-and-delivery problem (PDP-CR), comparing a flow-based genetic algorithm (GA), tabu search (TSR), and multi-start local search (MLS). An additional notebook cell compares a time-limited CPLEX MILP run with saved GA best-of-5 results.

All methods minimize synchronized makespan using the same task requirements, processing times, travel times, and robot fleet. The pilot evaluates feasible solution quality; it does not establish general superiority or global optimality of the heuristic results.

## Methods

- **GA:** a population of flow-DAG solutions with crossover, mutation, diversity management, and bounded local descent. Each independent process has its own population of 16.
- **TSR:** tabu search using reroute, reverse, and cut-redistribution neighborhoods. The implementation is an independent reconstruction, rather than the original authors' code.
- **MLS:** repeated randomized greedy construction followed by the same bounded local descent used by GA, without a population or crossover.
- **CPLEX:** an integer-flow MILP with arc activation, synchronized start times, and order constraints that exclude cycles even when processing or travel times are zero.

## Benchmark and conversion

The default pilot contains nine instances:

| PDP-CR tasks | Nominal Li-Lim source nodes | Families | Instances |
|---:|---:|---|---:|
| 50 | 100 | LC1, LR1, LRC1 | 3 |
| 100 | 200 | LC1, LR1, LRC1 | 3 |
| 200 | 400 | LC1, LR1, LRC1 | 3 |

Adding LC2 and LR2 produces 15 instances across the three task counts.

Original data: [SINTEF Li-Lim benchmark](https://www.sintef.no/projectweb/top/pdptw/li-lim-benchmark/). The preparation script downloads raw pickup/delivery node data from the [zhu-he/pdptw-data mirror](https://github.com/zhu-he/pdptw-data), caches it in `data/li_lim/`, and records source SHA-256 hashes in `instances/manifest.json`.

Each complete pickup-delivery request becomes one PDP-CR task:

- Processing time: `c_i = Euclidean(pickup_i, delivery_i)`.
- Travel time: `t_ij = Euclidean(delivery_i, pickup_j)`, without rounding.
- Source and sink use the original depot coordinates.
- Task robot requirements: `r_i ~ Uniform{1, 2, 3, 4, 5}`, using requirement seed 0.
- Fleet size: `m = 10` robots.

Original time windows, service times, capacity constraints, demands, fleet size, and speed are not used as PDP-CR constraints. The original Li-Lim PDPTW objective and its BKS values are not PDP-CR reference optima. This independent conversion follows the stated Euclidean conversion rules; it is not claimed to reproduce the authors' private instance selection or random requirement seed.

Source size labels are nominal. For example, `lc101.txt` contains 53 complete requests. Preparation validates every source pair, then selects the first n complete requests in original pickup-ID order when more than n requests are available. No pair is split. The manifest records `source_task_count`, `selection`, and `selected_pairs`. A source with fewer than n complete requests raises an error. Coincident pickup/delivery coordinates retain their exact processing time of zero.

## Installation

Use Python with the packages in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
jupyter notebook run_pilot.ipynb
```

CPLEX is optional for the GA/TSR/MLS pilot. To run the final comparison cell, install IBM ILOG CPLEX separately. No Python CPLEX package is required.

## Run the notebook

Open `run_pilot.ipynb` from the project directory. The main configuration cell contains:

```python
DEMO_MODE = False
SIZES = [50, 100, 200]
FAMILIES = ["LC1", "LR1", "LRC1"]
SEEDS = [0, 1, 2, 3, 4]
WORKERS = 5
GA_POPULATION = 16
SECONDS_BY_N = {50: 30.0, 100: 60.0, 200: 120.0}
RESULT_NAME = "pilot_9"
PLOT_N = 100
PLOT_FAMILY = "LC1"
```

The main experiment runs five seeds per algorithm and instance, with the same time limit per process. Methods run sequentially, with up to five independent seed processes in parallel within a method. Results are saved after each seed, allowing completed runs to be reused.

Use a new `RESULT_NAME` when changing experiment settings or algorithm code. Resume checks compare the main experimental parameters; they do not certify that source code or input data have remained unchanged.

The convergence plot shows the best GA seed on the selected instance, with horizontal TSR and MLS best-of-5 references.

### Offline demonstration

Set `DEMO_MODE = True` to use the bundled `LR2_2_6.pdp` file. The demo selects its first 50 complete tasks and uses a two-second limit per seed. It is separate from the default nine-instance benchmark and writes to `results/demo_offline/`.

For an offline full pilot, place the corresponding raw `.txt` or `.pdp` sources in `data/li_lim/` and disable automatic downloading.

## Run from the command line

```bash
python prepare_instances.py --sizes 50 100 200 --families LC1 LR1 LRC1
python run_pilot.py --sizes 50 100 200 --families LC1 LR1 LRC1 --seconds50 30 --seconds100 60 --seconds200 120 --workers 5 --population 16
```

For 15 instances:

```bash
python prepare_instances.py --families LC1 LR1 LRC1 LC2 LR2
python run_pilot.py --families LC1 LR1 LRC1 LC2 LR2 --output results/pilot_15
```

For locally cached sources:

```bash
python prepare_instances.py --offline
```

## Best-of-5 comparisons

For each algorithm and instance, best-of-5 is the minimum makespan across its five saved seeds. The GA percentage improvement relative to a competing method is:

```text
100 * (competitor_best_of_5 - GA_best_of_5) / competitor_best_of_5
```

A positive value means GA has the lower makespan. A negative value means the competing method has the lower makespan. When grouping by instance size, average the per-instance percentages across the families in that group.

`best_found` is the best feasible objective observed in the pilot. Reported RPD values measure distance from that empirical reference, not from a certified optimum. Mean and standard deviation across seeds are additional summaries and are distinct from best-of-5.

## CPLEX versus saved GA results

Run only the last notebook cell to compare CPLEX with the existing GA results. It does not rerun GA. The cell uses this executable path, which can be edited for another installation:

```text
C:\Program Files\IBM\ILOG\CPLEX_Studio2212\cplex\bin\x64_win64\cplex.exe
```

For each instance, CPLEX runs once with seed 0, one thread, and the time limit recorded in the saved pilot configuration: 30, 60, or 120 seconds for the bundled task sizes. It receives a common deterministic greedy MIP start, without using GA, TSR, or MLS results as a warm start. The comparison is explicitly **GA best-of-5 versus one CPLEX run**, with the same time limit per run.

The CPLEX limit applies to optimization using wall-clock time. Model generation and CLI overhead are reported separately. Saved CPLEX results are reused only when the recorded instance, runner, executable, and solver settings match.

The comparison includes the CPLEX incumbent, solver status, lower bound, relative MIP gap when available, and GA percentage improvement. Status distinguishes `optimal`, `feasible`, and `no_incumbent`. A time-limited incumbent is not automatically optimal. If no incumbent is exported, the CPLEX objective and percentage comparison remain unavailable.

## Saved outputs

The included `results/pilot_9/` directory contains the saved experiment outputs. Each experiment directory uses:

| Path | Contents |
|---|---|
| `config.json` | Experiment settings and environment information |
| `runs.csv` | Per-seed objective, timing, validity, and RPD |
| `summary.csv` | Per-instance/method best-of-5, mean, standard deviation, and timing |
| `comparison.csv` | GA/TSR/MLS best-of-5 and percentage comparisons |
| `overall.csv` | RPD summaries grouped by task count |
| `seeds/` | Per-seed solution, history, and diagnostic records |
| `best/` | Best saved solution for each method and instance |
| `cplex_vs_ga/comparison_best_of_5_vs_cplex.csv` | CPLEX incumbent versus GA best-of-5 |
| `cplex_vs_ga/<instance>/result.json` | CPLEX status, bounds, incumbent, replay, and settings |
| `cplex_vs_ga/<instance>/cplex.log` | CPLEX solver log |

Generated CPLEX LP, SOL, and MST files are excluded from Git and recreated by the notebook. Completed solver logs and result JSON files remain available.

## Validation and limitations

Run the regression checks without launching optimization:

```bash
python -m unittest discover -s tests -v
```

Saved solutions are checked against flow conservation, acyclicity, makespan evaluation, and physical robot-route replay. CPLEX integration tests use mocked optimization; reading a model or validating a feasible solution alone does not prove optimality.

Python heuristic deadlines are soft: a neighborhood operation can finish after its deadline. Per-seed records include solver wall time, CPU time, and overrun. CPLEX records its optimization time separately from model generation and CLI execution.

The pilot has one selected source per family and task count, five seeds, and a fixed requirement seed. Broader claims require additional independent instances, seeds, budgets, and component ablations. The sources selected for different task counts can overlap, so the nine cases should not be treated as nine independent random samples.

## Third-party software and data

IBM ILOG CPLEX is proprietary. Its executable and license are not included. Li-Lim data originate from the sources linked above; this repository does not grant a new license for third-party data.
