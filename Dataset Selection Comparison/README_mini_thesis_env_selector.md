# Mini Thesis Environment Selector

This package is for a **quick, thesis-focused comparison** between two Grid2Op environments:

- `l2rpn_neurips_2020_track1_small`
- `l2rpn_wcci_2022`

It is meant for **environment selection**, not full model training.

## What it does

The selector runs a short probe on both environments and compares them using metrics that fit the thesis goal:

- **scope fit**: how well the environment matches the current neuro-symbolic thesis design
- **efficiency**: how fast the environment can be stepped through in a short probe
- **event richness**: whether the environment quickly produces useful non-normal states
- **graph simplicity**: how manageable the environment is for an end-to-end GNN + symbolic shield pipeline

The final decision is based on a **weighted thesis score**, not on raw prediction accuracy.

## Why this is useful

Running full training on both large environments just to choose one is inefficient. This mini benchmark gives a defendable way to choose the better primary environment before the full thesis pipeline is trained.

## Files

- `mini_thesis_env_selector.py` — short benchmark script
- `mini_thesis_env_selection_results.json` — saved result from the probe
- `dataset_selection_report.md` — report in Markdown
- `dataset_selection_report.docx` — report in Word format

## Requirements

Install dependencies with:

```bash
pip install -r requirements_mini_thesis_env_selector.txt
```

## How to run

```bash
python mini_thesis_env_selector.py
```

## Expected output

The script will create a JSON file with:

- environment sizes
- short-probe metrics
- thesis-fit scores
- final recommendation

## How to explain the result

A good explanation for supervisors or reviewers is:

> The environment was selected using a short thesis-fit benchmark instead of full dual training. Both candidates were tested under the same small rollout budget and compared using symbolic compatibility, engineering feasibility, event richness, and graph complexity. This makes the selection more efficient and more aligned with the thesis scope than choosing based on accuracy alone.

## Notes

- This is a **screening benchmark**, not the final evaluation.
- The event labels in the probe are coarse screening labels, not the full thesis labeling pipeline.
- If Grid2Op downloads the environments on first use, the first run may take longer.
