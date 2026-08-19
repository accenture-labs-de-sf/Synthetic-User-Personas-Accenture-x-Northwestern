# Synthetic User Personas — Accenture × Northwestern

Behaviorally-grounded synthetic user personas for early product-testing interviews.

Synthetic personas today are written from a text description and a list of stated
preferences, so they only ever "know" what is on the page. This project grounds a
persona in the behavioral context that surrounds what a real person says — the
affect and emotion that ride alongside their words — and tests whether that makes
the persona more realistic and more useful for interviewing before real users are
recruited.

The grounding signal comes from affect and emotion labels in public datasets, used
as conditioning at generation time. This is a data-science study: work with real
behavioral data, condition a persona on it, generate interview responses, and
evaluate whether the grounding helped. See [`docs/project-plan.md`](docs/project-plan.md)
for the full plan.

## Status

Early. This is a placeholder scaffold; the structure below is a starting point and
will change once the student team onboards.

## Two build paths

The plan offers two ways to build the same experiment (teams can pick one or mix):

- **Plan A — analysis-first.** A notebook or Claude-driven session, minimal
  infrastructure, most of the time spent on the science and the evaluation.
- **Plan B — rigorous pipeline.** A reproducible Python package with a stronger
  generation and evaluation architecture and a standalone code base.

## Repository structure

```
docs/         Project plan and notes
data/         Datasets (downloaded locally, gitignored — not redistributed)
notebooks/    Plan A: analysis-first experiments
src/          Plan B: reproducible pipeline (src/personas package)
prompts/      Persona and judge prompt templates
evaluation/   Rubrics and judge / evaluation code
results/      Generated transcripts, scores, and figures
```

## Data and licensing

Datasets are downloaded into `data/` (gitignored); the repo points to them, it does
not redistribute them. Note the licenses of the sources used before any reuse: the
anchor grounding dataset is research and non-commercial only, and persona sources
vary (a commercial-friendly source is preferred). Details are in the project plan.
