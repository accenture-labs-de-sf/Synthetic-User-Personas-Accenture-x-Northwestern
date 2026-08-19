# Behaviorally-Grounded Synthetic User Personas

A project to test whether grounding synthetic user personas in richer, multimodal behavioral data makes them more realistic for early product-testing interviews, and to build the pipeline and evaluation that proves it.

## The one-line thesis

Synthetic personas become more realistic when they are grounded in the behavioral context that surrounds what a person literally says, not only the words in a chat box and the preferences a user states outright. In a real interview, **how someone reacts while they answer** (their affect, emotion, and tone) is data too, and a good researcher reads it alongside the literal words. This project grounds personas in that observed-reaction layer, using the affect and emotion labels that public datasets pair with real human speech, and tests whether that makes them more realistic.

## Where this fits

Most synthetic personas today are written from a text description and a list of stated preferences, so they only ever "know" what is in that description. The reframe behind this project is that **text is one part of a wider spectrum of user data**, and public affect datasets already capture part of the rest as labels on real speech, so a persona can be grounded in it without collecting any new signal.

This is a focused data-science study. The team works with real behavioral distributions, conditions a persona on them, generates interview responses, and checks whether the grounding helped. Building new signal-processing or biosensing methods is out of scope; the contribution is the evidence and the working pipeline.

## What the project investigates

The experiment is always the same A/B comparison: a text-only persona versus a behaviorally-grounded persona, same archetype, same interview questions. Three questions:

1. **Does behavioral grounding beat text-only?** Are grounded persona responses judged more realistic and human-aligned than text-only responses on the same questions?
2. **Is the gain substance or style?** Does grounding produce richer, more specific, more nuanced answers, or just a different tone?
3. **Does grounding stay consistent across an interview?** Does the persona hold its behavioral journey across a multi-question interview, or does the extra conditioning make it drift?

## How the grounding works

**The grounding data.** The grounding comes from CMU-MOSEI, a public dataset of real people speaking where every utterance is labeled for sentiment and emotion (described under Datasets). We do not use the raw audio or video, only these per-utterance labels.

**From labels to a behavioral journey.** Each speaker's utterances are ordered in time, so read in sequence their affect labels trace how that person's affect moves (for example, guarded at first, warming up, a spike of frustration on one topic, then leveling off). We extract that arc and summarize it into a compact **behavioral journey**: a short document or structured profile describing the person's affective trajectory and tendencies, built from the real label sequence rather than invented.

**Static conditioning of a dynamic journey.** The behavioral journey is encoded once and supplied to the persona at the start of the interview. The conditioning is **static**, authored a single time, but what it encodes is a **dynamic** arc: a trajectory and set of tendencies the persona carries across the whole interview. The movement is captured inside the conditioning artifact itself, so one up-front encoding produces realistic variation across the interview.

**Defining the journey (open design item).** Exactly how a journey is inferred from the label sequence is an item to settle early. The likely approach is a **schema** that organizes the relevant context for one speaker (their ordered affect labels, the matching transcript snippets, and any coarse identity), plus a **prompt and a trusted reasoning loop** that asks an LLM to read that context and infer the journey in the schema's shape. That inferred journey is the artifact the three application levels below consume. The plans differ in how the inference is run: **Plan A** can lean on the Claude Code harness as a trusted product utility to do the inference, while **Plan B** hand-crafts the inference step or borrows from trusted repositories so it runs as a reproducible pipeline stage.

**The two arms.** The **text-only arm (A)** is the archetype identity alone, with no behavioral journey. The **grounded arm (B)** is the same archetype plus the journey, which shapes how the persona answers and how it moves across the interview, not what facts it holds. That single difference is what the A/B measures.

**How the journey is applied.** The mechanism is context conditioning, not fine-tuning: we steer the generator through its prompt and context, we do not retrain it. Three levels of rigor match the two build paths:

1. **Prompt-level (Plan A).** Render the behavioral journey into a short natural-language conditioning block describing the person's affective arc and how it shows up in speech, with the instruction to let it shape the manner and pacing of answers and not to state it outright. Derive those descriptors from the real utterances rather than from intuition.
2. **Exemplar grounding (either plan, stronger).** Alongside the journey, include a few real utterances from the source sequence as style anchors, so the manner is grounded in actual human language and not only a description of it.
3. **Verified grounding (Plan B).** Run an independent affect classifier over the generated interview and check that its affective arc tracks the intended journey. If it does not, regenerate or flag. This reuses the same classifier the Plan B evaluation uses.

We deliberately stop at context conditioning: we do not fine-tune or train the generator.

## Early testing of an AI product concept

A product team is about to build an AI meeting-assistant feature inside their workplace app, and before spending recruitment budget they want to interview a few target users about the concept. Below is the same interview run through both arms.

**The archetype (identity, from the persona library).** Maya, 34, operations manager at a mid-size logistics firm. She runs on a stack of tools she did not choose and has been burned by software rollouts that overpromised. This is who she is, and it does not change between the two arms.

**The behavioral journey (grounding, from CMU-MOSEI).** Taken from a real speaker whose affect arc fits an interview: guarded and neutral at the open, warming when a real pain point is named, a sharp turn to concern when the topic touches data and privacy, then settling into cautious interest by the end. It is encoded once, up front.

**The interview questions (from the product scenario).** For example: what is your first reaction to an assistant that auto-summarizes your meetings? How would it fit your current workflow? What would stop you from trusting it? Would you pay for it?

**What the two arms produce.**

- **Text-only Maya** answers on an even keel throughout. Reasonable, a little generic, roughly the same temperature to every question.
- **Grounded Maya** moves: reserved on the opener, more engaged when workflow fit is raised, visibly wary when privacy comes up ("who else can see these summaries?"), and guardedly warm at the close. The privacy concern comes through as a real objection rather than a flat, evenly-weighted point.

**Why the product team cares.** Because the grounded interview tracks where a user's reaction shifts, the team can see that trust and data handling is the first blocker to address, and they get that read before recruiting anyone.

## The two build paths

Both paths run the same experiment and answer the same questions. They differ in how much the team builds and how rigorous the generation and evaluation architecture is. Teams can pick one or mix them.

### Plan A: analysis-first

The goal is fast, clean evidence with minimal infrastructure.

- **Persona engine and judge:** the Claude API, or a Claude Code session driving the loop.
- **Environment:** a notebook or a driven session, with pandas for analysis. No local model serving, no pipeline framework.
- **Generation:** a small set of archetypes (hand-authored, or a handful pulled from a persona library) conditioned on behavioral journeys extracted from the real dataset.
- **Evaluation:** LLM-as-judge scoring a human-authored rubric (see Evaluation).
- **Best when** the team is stronger on analysis and wants results quickly, spending most of the time on journey extraction, prompt design, rubric design, and interpretation.

### Plan B: rigorous pipeline

The goal is a stronger, more defensible architecture for both generation and evaluation, and a standalone code base another team can reproduce.

- **Generation architecture:**
  - Archetypes sampled systematically from a persona library using its attributes or domain taxonomy, so the archetype set is representative and reproducible rather than hand-picked.
  - A structured persona schema separating stable identity from the extracted behavioral journey.
  - A multi-turn interview loop with conversation memory, run locally through Ollama or through an API behind the same interface.
  - Reproducibility built in: every run reproducible from a config (dataset version, sampling seed, model, prompt template).
- **Evaluation architecture:** the judge is not left as a purely qualitative LLM characterization. It is combined with formal, grounded metrics so the qualitative call is checked against measurable signal (see Evaluation for the grounded and adversarial setup).
- **Multi-model ensembling (either path):** where it strengthens a step, run several LLMs together and use their collective stochasticity, combining models in the reasoning paths (such as journey inference) or the judgment paths (such as the eval), instead of trusting a single model's single sample.
- **Best when** the team has strong engineers and wants a durable code base and a more rigorous result.

### Persona source and licensing

Archetypes come from a public persona library, and either plan can use one. The difference is rigor: Plan A can pull a handful directly, while Plan B samples systematically across the library's attributes for coverage and reproducibility.

Licensing matters here, because the deliverables are meant to be reusable in client and internal settings, so prefer a source that allows commercial use:

- **NVIDIA Nemotron-Personas (recommended default).** Licensed CC BY 4.0, so attribution only, commercial allowed, no non-commercial or share-alike restriction. Ready-made persona text plus structured demographic attributes, grounded in real-world demographic distributions, which also fits the project's spirit. Loads directly from Hugging Face as parquet, so integration is a one-liner in either plan.
- **Google Synthetic-Persona-Chat.** Also CC BY 4.0, with each persona paired to example dialogue, useful if a conversational style reference helps.
- **FinePersonas (Argilla).** Very large and PersonaHub-style, under the Llama Community License, which permits commercial use under its terms but is not as clean as CC BY.
- **PersonaHub.** Usable, but CC-BY-NC-SA is non-commercial and share-alike, so keep it to the study and avoid it for anything reused commercially.

Note that CMU-MOSEI itself is research and non-commercial only, so the grounding-data half of the pipeline stays within the study regardless of the persona source.

## Datasets

**CMU-MOSEI (grounding source and human reference).** Over 23,500 utterances from more than 1,000 YouTube speakers giving opinions on varied topics, aligned across text, audio, and visual, with each utterance labeled for sentiment (-3 to +3) and six emotions (happy, sad, angry, fearful, surprised, disgusted) plus valence, arousal, and dominance. We use the transcripts and affect labels, not the raw audio or visual features. It plays two roles: it is the source of the behavioral journeys used for grounding, and it is the real-human reference the generated answers are compared against. Practical caveats: the full multimodal features are large (around 11 GB) and access-gated, the license is research and non-commercial only, and a convenience CSV in one public copy is malformed and mislabeled, so read the labels carefully. Working at the label-and-transcript level avoids the heavy signal-processing work. Using one dataset for both roles is deliberate: a grounded answer is compared against real utterances carrying the same affect, matched on domain, modality, and state, which a separate reference corpus could not offer.

**Interview questions.** These come from the specific product-testing scenario the team chooses (for example, reactions to a new feature concept), authored to fit that use case. There is no need for a separate interview dataset.

## Evaluation

Use **human-guided LLM-as-judge**, supported by descriptive statistics. Human-guided means people own the rubric and calibrate the judge; the LLM only scales it.

One limit up front: a rigorous human-alignment metric would need either a large, formal data-collection effort or an existing dataset that pairs AI and human responses to the same prompts together with behavioral-state labels, and no such dataset exists for this setting. Human alignment here is therefore **judged, not measured against a gold human-aligned corpus**, and the LLM-as-judge with human calibration is the practical stand-in.

Shared across both plans:

1. **Humans design the rubric.** Three to five dimensions tied to the research questions: realism and human-likeness, specificity, consistency in character, and optionally usefulness to a product researcher. Each dimension gets an anchored 1 to 5 scale with example answers per level.
2. **Blind judging.** For each question the judge sees the text-only and grounded answers without knowing which is which, picks the more human-like one, and scores each dimension. Randomize order to avoid position bias.
3. **Human calibration loop.** The team hand-scores a sample of pairs, compares to the judge, and reports agreement. That calibration is what makes the result defensible.
4. **Descriptive statistics as support:** type-token ratio, response length, specificity counts, and a check that the behavioral journey shows up in the answer (for example, does the affect actually shift across the interview the way the journey describes).

Plan B strengthens the judge so it is not only a qualitative characterization. It combines the LLM's judgment with formal, grounded metrics that either guide the judge or push against the generator:

- **Grounded verification.** Run an independent affect or sentiment classifier on the generated answers and check that they actually carry the intended affect. If a persona's journey calls for rising frustration but the answers read flat, the grounding did not take, regardless of what the LLM judge says qualitatively.
- **Distributional grounding.** Measure how close the generated answers sit to real CMU-MOSEI utterances of the same affect, so realism has a measurable anchor and not only a rated one.
- **Adversarial setup.** Stand up a real-versus-synthetic discriminator that tries to tell generated answers from real CMU-MOSEI utterances of the same affect. A generator that is easy to distinguish is not yet realistic, and this pressure is independent of the LLM judge's self-report.

Reported simply, the outcome is the share of blind head-to-heads the grounded personas won, the change in realism scores, and the judge-to-human agreement, with the grounded and adversarial metrics from Plan B alongside them.

## Phased plan

1. **Scope and data.** Choose the build path. Load CMU-MOSEI transcripts and affect labels and extract behavioral journeys from the real label sequences.
2. **Two personas.** Build the text-only prompt and the grounded prompt, and generate a first A/B batch.
3. **Evaluation harness.** Humans write the rubric, wire the LLM judge, and run the calibration loop. Plan B adds the grounded and adversarial metrics.
4. **Run the experiment.** Full A/B batch across archetypes and questions, with descriptive statistics.
5. **Harden and deliver.** Plan B packages the pipeline for reproducibility. Produce the final deliverables.

## Deliverables

- **Video demo.** A text-only persona and a grounded persona answering the same interview question side by side, followed by the evaluation result.
- **PoC deck.** The problem, the method, the evidence, and where it goes next.
- **Code base.** The Plan A notebook, the Plan B pipeline, or both, reproducible and documented.

## Out of scope

To keep the work focused, the project does not include raw multimodal signal extraction (audio or visual feature processing, EEG, or heart-rate and skin-response signals), knowledge-graph conditioning, live per-response state, or any learned model that generates behavioral state from raw signal. Static conditioning on the extracted behavioral journey is sufficient here.
