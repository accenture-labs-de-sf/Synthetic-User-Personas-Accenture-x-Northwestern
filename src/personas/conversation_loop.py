"""Adolfo's 5-Block Conversation Loop for Grounded Persona Testing.

Implements the architecture from `Conversation_Loop_Guide.pdf` integrated with:
- Wenjun's 3 persona variants (`full_description`, `short_version_1`, `short_version_2`
  from `Dataset/persona_standard_dataset_v3.csv`).
- Prakhar's 4 products x 7 questions (`subscription-management`, `cash-flow-planning`,
  `mental-wellness`, `sexual-health`, including multimodal screen stimuli).
- CMU-MOSEI 7-stage journey grounding metrics translated into behavioral rules for
  the three arms (`text_only`, `coarse`, `fine`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .journeys import EMOTION_COLUMNS, SelectedJourney, select_journeys

ArmName = Literal["text_only", "coarse", "fine"]
PersonaVariant = Literal["full_description", "short_version_1", "short_version_2"]

FORMAT_BLOCK = """You are being interviewed about a product. Answer every question in first person using this exact three-tag format:
<think>2-3 sentences, first person: how this question lands given your current state, your behavioral tendencies, and what you have already said.</think>
<reply>What you say out loud to the interviewer, in your own natural spoken voice (3-5 sentences, no bullet points or headings).</reply>
<state>patience: low|medium|high; trust: low|medium|high; mood: one word</state>"""


@dataclass
class GroundingMetrics:
    """Behavioral grounding metrics extracted from a CMU-MOSEI recording (7 stages)."""

    video_id: str
    journey_label: str
    shape: str
    valence_mean: float
    arousal_mean: float
    valence_drop_after_friction: float
    recovery_rate: float
    arousal_var: float
    sentiment_span: float
    dominant_emotions: List[Tuple[str, float]]
    stage_sentiments: List[float]
    stage_dominant_emotions: List[str]
    initial_state: str
    stage_target_states: List[str]


@dataclass
class Session:
    """Single interview session tracking Block A (card), Block B (state), and Block C (memory)."""

    session_id: str
    product_id: str
    product_name: str
    persona_uuid: str
    persona_name: str
    persona_variant: PersonaVariant
    arm: ArmName
    journey_label: str
    video_id: str
    card: str
    state: str = "patience: medium; trust: medium; mood: neutral"
    turns: List[Tuple[str, str]] = field(default_factory=list)  # (question_text, reply_only)
    log: List[Dict[str, Any]] = field(default_factory=list)


def first_sentence(text: str) -> str:
    """Extract the first sentence of a question or reply for Block C memory compression."""
    cleaned = " ".join(str(text).strip().split())
    parts = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
    return parts[0] if parts else cleaned


def estimate_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    """Approximate token count (~4 chars/token for text + 255 tokens for image thumbnail)."""
    total_chars = 0
    image_tokens = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total_chars += len(part.get("text", ""))
                elif part.get("type") == "image_url":
                    image_tokens += 255
    return int(round(total_chars / 3.8)) + image_tokens


def _state_from_stage_affect(sentiment: float, top_emo: str, top_emo_val: float, stage_idx: int) -> str:
    """Derive a structured `patience; trust; mood` string from a stage's affect."""
    if sentiment <= -0.8 or (top_emo in ("anger", "disgust") and top_emo_val >= 0.6):
        patience = "low"
    elif sentiment >= 0.75 and top_emo == "happy":
        patience = "high"
    else:
        patience = "medium"

    if sentiment <= -0.4 or (top_emo in ("fear", "disgust", "anger") and top_emo_val >= 0.4):
        trust = "low"
    elif sentiment >= 0.65:
        trust = "high"
    else:
        trust = "medium"

    if sentiment <= -1.0:
        mood_map = {
            "anger": "frustrated",
            "disgust": "fed-up",
            "sad": "discouraged",
            "fear": "uneasy",
            "surprise": "blindsided",
        }
        mood = mood_map.get(top_emo, "skeptical")
    elif sentiment < -0.2:
        mood_map = {
            "anger": "irritated",
            "disgust": "wary",
            "sad": "hesitant",
            "fear": "cautious",
            "surprise": "doubtful",
        }
        mood = mood_map.get(top_emo, "guarded")
    elif sentiment <= 0.35:
        mood = "observant" if stage_idx == 1 else "measured"
    elif sentiment <= 1.1:
        mood = "curious" if top_emo == "surprise" else "receptive"
    else:
        mood = "engaged" if stage_idx < 5 else "satisfied"

    return f"patience: {patience}; trust: {trust}; mood: {mood}"


def compute_grounding_metrics(journey: SelectedJourney) -> GroundingMetrics:
    """Compute Adolfo's behavioral grounding metrics from a 7-stage CMU-MOSEI journey."""
    sents = np.array([st.sentiment for st in journey.stages], dtype=float)
    diffs = np.diff(sents)

    # Arousal proxy: total intensity of discrete emotions at each stage
    arousals = np.array(
        [sum(st.emotions.values()) for st in journey.stages],
        dtype=float,
    )
    valence_mean = float(np.mean(sents))
    arousal_mean = float(np.mean(arousals))
    arousal_var = float(np.var(arousals) + 0.5 * np.var(sents))

    neg_drops = [-d for d in diffs if d < 0]
    valence_drop_after_friction = float(max(neg_drops)) if neg_drops else 0.0

    # Recovery rate: average positive bounce immediately following a negative drop
    bounces = []
    for i in range(len(diffs) - 1):
        if diffs[i] < -0.15:
            bounces.append(max(0.0, float(diffs[i + 1])))
    if bounces:
        recovery_rate = float(np.mean(bounces) / max(0.25, valence_drop_after_friction))
    else:
        recovery_rate = 0.65 if sents[-1] >= sents[0] else 0.10

    # Aggregate dominant emotions across the recording
    emo_totals: Dict[str, float] = {}
    for e in EMOTION_COLUMNS:
        emo_totals[e] = round(float(np.mean([st.emotions[e] for st in journey.stages])), 3)
    dom_emos = sorted(
        [(e, v) for e, v in emo_totals.items() if v >= 0.05],
        key=lambda item: item[1],
        reverse=True,
    )

    stage_dom_names: List[str] = []
    stage_states: List[str] = []
    for st in journey.stages:
        if st.dominant_emotions:
            top_e, top_v = st.dominant_emotions[0]
        else:
            top_e, top_v = "neutral", 0.0
        stage_dom_names.append(f"{top_e} ({top_v:.2f})" if top_v >= 0.08 else "neutral")
        stage_states.append(_state_from_stage_affect(st.sentiment, top_e, top_v, st.stage))

    return GroundingMetrics(
        video_id=journey.video_id,
        journey_label=journey.label,
        shape=journey.shape,
        valence_mean=round(valence_mean, 3),
        arousal_mean=round(arousal_mean, 3),
        valence_drop_after_friction=round(valence_drop_after_friction, 3),
        recovery_rate=round(recovery_rate, 3),
        arousal_var=round(arousal_var, 3),
        sentiment_span=journey.sentiment_span,
        dominant_emotions=dom_emos,
        stage_sentiments=[round(float(x), 2) for x in sents],
        stage_dominant_emotions=stage_dom_names,
        initial_state=stage_states[0],
        stage_target_states=stage_states,
    )


def coarse_rules(g: GroundingMetrics) -> List[str]:
    """Translate coarse journey metrics into 2-3 high-level behavioral rules (Block A)."""
    rules: List[str] = []
    if g.valence_mean < -0.15:
        rules.append("You lean skeptical about new digital tools and look for what could go wrong before giving credit.")
    elif g.valence_mean > 0.35:
        rules.append("You start generally constructive toward practical tools, provided they respect your time and routine.")
    else:
        rules.append("You evaluate product claims with a measured, matter-of-fact mindset rather than automatic enthusiasm.")

    if g.arousal_mean > 0.85:
        rules.append("You react visibly and directly when a feature either helps you or wastes your effort.")
    else:
        rules.append("You keep your emotional expression understated and grounded in everyday practicality.")

    if g.stage_sentiments[-1] < g.stage_sentiments[0] - 0.35:
        rules.append("As the interview goes on, your patience wears thinner if earlier questions left loose ends.")
    elif g.stage_sentiments[-1] > g.stage_sentiments[0] + 0.35:
        rules.append("You start somewhat reserved on the opening questions and warm up once you see concrete control.")
    return rules


def fine_rules(g: GroundingMetrics) -> List[str]:
    """Translate fine-grained journey dynamics and discrete emotions into specific conditional rules (Block A)."""
    rules = coarse_rules(g)

    if g.valence_drop_after_friction > 0.35:
        rules.append(
            "Whenever a screen or policy feels vague, intrusive, or incomplete, your trust drops sharply and you stay guarded for the next couple of answers."
        )
    if g.recovery_rate < 0.25:
        rules.append(
            "Once a missing detail or hand-off annoys you, you do not easily warm back up—you carry that unresolved objection into your final verdict."
        )
    else:
        rules.append(
            "If a screen genuinely resolves a concern you raised earlier, you acknowledge the improvement while keeping track of what is still missing."
        )

    if g.arousal_var > 0.25:
        rules.append(
            "Your conversational pacing shifts across turns: brisk and pointed when you spot friction, more expansive when a feature fits your real workflow."
        )

    emo_dict = dict(g.dominant_emotions)
    if emo_dict.get("anger", 0.0) >= 0.20 or emo_dict.get("disgust", 0.0) >= 0.20:
        rules.append(
            "When a screen leaves you in limbo or asks for trust without proof, let a dry, candid, slightly sharp edge show instead of polite corporate phrasing."
        )
    if emo_dict.get("fear", 0.0) >= 0.12 or emo_dict.get("sad", 0.0) >= 0.20:
        rules.append(
            "When a question touches on personal exposure, irreversible actions, or dropped responsibilities, voice the concrete worry behind your hesitation."
        )

    # Summarize the 7-turn state progression as qualitative turn habits (no raw decimals)
    stage_Progression = " -> ".join(
        f"Q{idx + 1}: {st.split('mood: ')[-1]}" for idx, st in enumerate(g.stage_target_states)
    )
    rules.append(
        f"Across the 7 questions, your conversational posture naturally moves through this sequence: [{stage_Progression}]."
    )
    return rules


def build_card(persona_text: str, rules: Sequence[str]) -> str:
    """Assemble Block A (fixed system prompt per arm for prefix caching)."""
    body = f"Who you are:\n{persona_text.strip()}"
    if rules:
        rules_block = "\n".join(f"- {r}" for r in rules)
        body += f"\n\nHow you behave in this interview:\n{rules_block}"
    return f"{body}\n\n{FORMAT_BLOCK}"


def build_cards_for_persona(persona_text: str, g: GroundingMetrics) -> Dict[ArmName, str]:
    """Return the three Block-A system cards (`text_only`, `coarse`, `fine`) for a persona and journey."""
    return {
        "text_only": build_card(persona_text, []),
        "coarse": build_card(persona_text, coarse_rules(g)),
        "fine": build_card(persona_text, fine_rules(g)),
    }


def build_messages(
    session: Session,
    question_text: str,
    participant_instruction: str = "",
    image_data_uri: Optional[str] = None,
    keep: int = 2,
) -> Tuple[List[Dict[str, Any]], str]:
    """Assemble the 5-block prompt (A, B, C, D, E) for a single turn of the loop.

    Returns `(messages, assembled_user_text)` so the exact user block can be inspected at Turn 4+.
    """
    # Block A: Fixed system card (cached prefix)
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": session.card}]

    # Block C (verbatim part): Last `keep` turns (question + reply only; <think> is never in memory)
    older = session.turns[:-keep] if keep > 0 else session.turns
    recent = session.turns[-keep:] if keep > 0 else []

    for prev_q, prev_r in recent:
        msgs.append({"role": "user", "content": prev_q})
        msgs.append({"role": "assistant", "content": prev_r})

    # Block C (compressed part): 1-line summaries of earlier turns
    summary_lines = [
        f"- Asked about: {first_sentence(q)} You said: {first_sentence(r)}"
        for q, r in older
    ]
    user_parts: List[str] = []
    if summary_lines:
        user_parts.append("Earlier in this conversation:\n" + "\n".join(summary_lines))

    # Block B: Live state
    user_parts.append(f"Your current state: {session.state}")

    # Block D: Current question stimulus q_t
    if participant_instruction.strip():
        user_parts.append(f"Context: {participant_instruction.strip()}")
    user_parts.append(f"Question: {question_text.strip()}")

    # Block E: One-line reminder immediately before generation
    user_parts.append(
        f"(Stay in character as yourself, currently {session.state}. "
        "Respond using <think>, <reply>, and <state> tags.)"
    )

    user_text = "\n\n".join(user_parts)
    if image_data_uri:
        content: Any = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ]
    else:
        content = user_text

    msgs.append({"role": "user", "content": content})
    return msgs, user_text


def parse_structured_output(raw_text: str) -> Tuple[Optional[str], str, Optional[str]]:
    """Parse `<think>`, `<reply>`, and `<state>` from model output."""
    def _extract(tag: str) -> Optional[str]:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", raw_text, re.S | re.I)
        return m.group(1).strip() if m else None

    think = _extract("think")
    reply = _extract("reply")
    state = _extract("state")

    if not reply:
        # Strip any <think> or <state> tags if <reply> tag was omitted
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.S | re.I)
        cleaned = re.sub(r"<state>.*?</state>", "", cleaned, flags=re.S | re.I).strip()
        reply = cleaned or raw_text.strip()

    return think, reply, state


def update_session_state(
    arm: ArmName,
    parsed_state: Optional[str],
    turn_idx: int,
    grounding: GroundingMetrics,
    current_state: str,
) -> str:
    """Update Block B state after a turn, combining model's <state> with journey stage progression."""
    if arm == "text_only":
        if parsed_state and "patience:" in parsed_state.lower() and "trust:" in parsed_state.lower():
            return parsed_state
        return current_state

    # For grounded arms (`coarse` and `fine`), ensure the next turn's state reflects the journey's
    # next stage trajectory while preserving valid `<state>` formatting.
    next_stage_idx = min(len(grounding.stage_target_states) - 1, turn_idx)
    journey_next_state = grounding.stage_target_states[next_stage_idx]

    if arm == "fine":
        # Fine arm tracks the journey's stage-by-stage state transitions directly
        return journey_next_state

    # Coarse arm blends the model's self-reported state with the journey's macro posture
    if parsed_state and "patience:" in parsed_state.lower() and "mood:" in parsed_state.lower():
        return parsed_state
    return journey_next_state
