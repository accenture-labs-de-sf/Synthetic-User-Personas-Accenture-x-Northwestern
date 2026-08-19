# Synthetic User Personas — Accenture × Northwestern

Behaviorally-grounded synthetic user personas for early product-testing interviews.

Synthetic personas today are written from a text description and a list of stated
preferences, so they only ever "know" what is on the page. This project grounds a
persona in the behavioral context around what a real person says, meaning the affect
and emotion that ride alongside their words, and tests whether that makes the persona
more realistic and more useful for interviewing before real users are recruited.

The grounding signal comes from affect and emotion labels in public datasets, used as
conditioning at generation time. This is a data-science study. The team works with
real behavioral data, conditions a persona on it, generates interview responses, and
checks whether the grounding helped. See [`docs/project-plan.md`](docs/project-plan.md)
for the full plan.

## Status

Early. This is a placeholder scaffold. The structure below is a starting point and
will change once the student team onboards.

## Two build paths

The plan offers two ways to build the same experiment, and teams can pick one or mix
them:

- **Plan A, analysis-first.** A notebook or Claude-driven session with minimal
  infrastructure, where most of the time goes to the science and the evaluation.
- **Plan B, rigorous pipeline.** A reproducible Python package with a stronger
  generation and evaluation architecture and a standalone code base.

## Repository structure

```
docs/         Project plan and notes
data/         Datasets (downloaded locally, gitignored, not redistributed)
notebooks/    Plan A: analysis-first experiments
src/          Plan B: reproducible pipeline (src/personas package)
prompts/      Persona and judge prompt templates
evaluation/   Rubrics and judge / evaluation code
results/      Generated transcripts, scores, and figures
```

## Data and licensing

Datasets are downloaded into `data/` (gitignored). The repo points to them, it does
not redistribute them. Check the license of each source before any reuse: the anchor
grounding dataset is research and non-commercial only, and persona sources vary, so a
commercial-friendly source is preferred. Details are in the project plan.
