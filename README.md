# Does the Jacobian Lens Help Choose Which SAE Features to Edit?

Code, configurations and data splits for the paper. The package `saeselect` is a controlled causal benchmark of signals for choosing which sparse-autoencoder (SAE) features to edit: the Jacobian lens (J-lens), its logit-lens twin, attribution patching (ATP) and simple association, compared by depth and edit site on concept-transfer tasks in Gemma 3 12B (instruction-tuned and base) and Llama 3.1 8B.

## Install

Python 3.12 or later and a CUDA GPU for the model runs. The analyses and the tests run on the CPU.

```bash
pip install -e ".[dev,lens]"
```

`requirements.lock` lists the exact environment the results were produced in. The `lens` extra installs the J-lens reference implementation at the pinned commit.

## Layout

| Path | Contents |
|---|---|
| `src/saeselect/` | The package: model wrapper, SAEs, lenses, signals, interventions, statistics, run contract |
| `src/saeselect/tasks/` | Data build: relations R1 to R6 and the neutral corpus |
| `src/saeselect/stages/` | Model runs: inventory, raw completions, selection benchmark and its extensions |
| `src/saeselect/analysis/` | CPU analyses of stored runs, and the figures |
| `configs/` | One YAML file per run, grouped by step (`models`, `data`, `inventory`, `completions`, `selection`, `extensions`, `analysis`) |
| `data/built*/` | Split assignments and benchmark cards of the built tasks (records are rebuilt locally) |
| `tests/` | Unit and end-to-end tests on a tiny random Gemma 3 |

## Run contract

Every command writes a run directory with `SUMMARY.md`, `report.json`, provenance (commit, hashes of the loaded source files, data hashes, environment) and ends with one status line, `STATUS=<code> REPORT=<dir>/SUMMARY.md SEND_BACK=<files>`. Values that are not declared in a config are derived on development pairs only and written to `derived.json` with the procedure that produced them. A development run freezes its protocol (`frozen_protocol.json`); the test run takes that file with `--frozen`, must run on the same code commit, and runs once on the held-out pairs.

## Pipeline

Commands take the form `python -m saeselect <command> --config <file> --out <dir>`, with `--model configs/models/<model>.yaml` for model runs and `--device cuda` on the GPU. The steps below use Gemma 3 12B-IT; the base model and Llama use the configs with `gemma_base` or `llama` in their names.

1. Build the tasks (CPU; fetches the pinned sources and checks them by SHA-256):

   ```bash
   python -m saeselect build-data --config configs/data/relations.yaml --out runs/build
   ```

   `configs/data/new_tasks.yaml` builds R5 and R6; the `_llama` configs build the same splits with the Llama tokenizer.

2. Inventory and numerics checks per model (layer grid, SAE fidelity, precision modes, lens parity):

   ```bash
   python -m saeselect inventory --config configs/inventory/gemma_it.yaml --model configs/models/gemma3_12b_it.yaml --out runs/inventory-gemma-it --device cuda
   ```

3. Raw completions, read before the answer sets are used:

   ```bash
   python -m saeselect raw-completions --config configs/completions/gemma_it.yaml --model configs/models/gemma3_12b_it.yaml --out runs/completions-gemma-it --device cuda
   ```

4. Selection benchmark, development then test:

   ```bash
   python -m saeselect select --config configs/selection/gemma_it.yaml --model configs/models/gemma3_12b_it.yaml --phase dev --out runs/selection-gemma-it-dev --device cuda
   ```

   ```bash
   python -m saeselect select --config configs/selection/gemma_it.yaml --phase test --frozen runs/selection-gemma-it-dev/frozen_protocol.json --out runs/selection-gemma-it-test --device cuda
   ```

   The same two steps run the new tasks (`new_tasks_*`), the off-target endpoint (`offtarget_*`), the SAE widths (`sae_widths`) and the SAE swap (`sae_swap_*`).

5. Extensions on a finished selection run, each with a development and a test phase: `jacobian-controls` (`configs/extensions/jacobian_variants.yaml` and `jacobian_controls.yaml`), `greedy-reference`, `joint-selection` and `generate`.

6. Analyses of the stored test runs (CPU):

   ```bash
   python -m saeselect analyze --config configs/selection/gemma_it.yaml --run runs/selection-gemma-it-test --out analysis/selection-gemma-it
   ```

   | Command | Config | Content |
   |---|---|---|
   | `compare-sets` | `analysis/compare_sets.yaml`, `compare_sets_gemma_base.yaml`, `sae_swap.yaml` | Set recovery across models and SAEs |
   | `posthoc` | `analysis/posthoc.yaml` | SESOI sensitivity, absolute endpoints, edit size, leave-one-relation-out |
   | `additivity` | `analysis/additivity.yaml` | Ceilings, superadditivity, off-target ratio |
   | `decomposition` | `analysis/decomposition.yaml` | Recovery split into singleton utility and remainder |
   | `matched-size` | `analysis/matched_size.yaml` | ATP against the J-lens at matched edit size |
   | `figures` | `analysis/figures.yaml` | The paper's figures |

Run `python -m saeselect --help` for every command and option.

## Tests

```bash
pytest -q
```

The tests use a tiny random Gemma 3 and small local stand-ins for the data sources; they need no GPU and no downloads.
