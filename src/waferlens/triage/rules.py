"""Rule-based failure-analysis (FA) triage.

Maps detected wafer-map defect patterns to candidate process root causes and
recommended engineering checks. This is the deliberate "engineering artifact"
that lifts WaferLens above a bare Kaggle classifier: the model says *what* the
pattern is; the triage layer says *where to look*.

The mappings encode widely-documented spatial-signature heuristics from the
semiconductor yield-engineering literature. They are CANDIDATE hypotheses for a
human FA engineer to confirm - never a final diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TriageEntry:
    pattern: str
    candidate_causes: list[str]
    recommended_checks: list[str]
    typical_process_area: str


# Single-pattern knowledge table
_RULES: dict[str, TriageEntry] = {
    "Center": TriageEntry(
        pattern="Center",
        candidate_causes=[
            "CMP dishing / over-polish at wafer center",
            "Spin-coat thickness non-uniformity (center-fast)",
            "Chamber center hot/cold spot in thermal processing",
        ],
        recommended_checks=[
            "Review CMP removal-rate radial profile",
            "Check spin-coat RPM ramp and dispense volume",
            "Pull center-die cross-sections for layer thickness",
        ],
        typical_process_area="CMP / Litho coat / Thermal",
    ),
    "Donut": TriageEntry(
        pattern="Donut",
        candidate_causes=[
            "Develop/photoresist ring non-uniformity",
            "Mid-radius etch-rate band from gas-flow distribution",
            "Edge-bead removal (EBR) mis-tuning",
        ],
        recommended_checks=[
            "Inspect develop puddle dynamics and dispense arm sweep",
            "Map etch rate vs radius; check showerhead pattern",
            "Verify EBR width setpoint",
        ],
        typical_process_area="Litho develop / Etch",
    ),
    "Edge-Loc": TriageEntry(
        pattern="Edge-Loc",
        candidate_causes=[
            "Localized edge clamp / handling contact damage",
            "Asymmetric plasma at wafer edge (one azimuth)",
            "Edge-ring erosion at a specific location",
        ],
        recommended_checks=[
            "Inspect e-chuck edge ring and clamp contact points",
            "Check chamber azimuthal symmetry / focus ring wear",
            "Correlate angular position with tool fixturing",
        ],
        typical_process_area="Etch / Handling",
    ),
    "Edge-Ring": TriageEntry(
        pattern="Edge-Ring",
        candidate_causes=[
            "Edge-wide etch or deposition non-uniformity",
            "Focus-ring / edge-ring wear or mis-seat",
            "Edge temperature roll-off in RTP/anneal",
        ],
        recommended_checks=[
            "Map deposition/etch thickness 0-148mm radial",
            "Inspect and re-qualify the focus/edge ring",
            "Check edge-zone heater calibration",
        ],
        typical_process_area="Etch / Depo / RTP",
    ),
    "Loc": TriageEntry(
        pattern="Loc",
        candidate_causes=[
            "Localized particle/contamination cluster",
            "Point defect from a specific tool fixturing contact",
            "Micro-scratch or pad asperity in CMP",
        ],
        recommended_checks=[
            "Run particle scan / defect-review SEM at the cluster site",
            "Overlay with tool-contact-point map",
            "Inspect CMP pad condition and slurry filtration",
        ],
        typical_process_area="Contamination / CMP / Handling",
    ),
    "Near-full": TriageEntry(
        pattern="Near-full",
        candidate_causes=[
            "Gross process excursion (recipe/setpoint fault)",
            "Wafer-wide contamination or develop failure",
            "Major metrology/test miscalibration (false fails)",
        ],
        recommended_checks=[
            "Pull tool FDC traces for the lot timeframe",
            "Verify recipe download and chamber state",
            "Confirm tester correlation / probe-card health",
        ],
        typical_process_area="Process-wide / Test",
    ),
    "Scratch": TriageEntry(
        pattern="Scratch",
        candidate_causes=[
            "Mechanical handling scratch (robot end-effector)",
            "CMP pad debris drag line",
            "Cassette/FOUP contact during transfer",
        ],
        recommended_checks=[
            "Inspect robot blade and transfer path for contact",
            "Check CMP pad conditioning and debris",
            "Review FOUP slot mapping for the affected wafers",
        ],
        typical_process_area="Handling / CMP",
    ),
    "Random": TriageEntry(
        pattern="Random",
        candidate_causes=[
            "Random particulate / ambient contamination",
            "Background defectivity (no spatial signature)",
            "Low-level systematic noise across the lot",
        ],
        recommended_checks=[
            "Trend lot/tool defectivity vs baseline",
            "Check cleanroom particle counts and filter status",
            "Confirm no single dominant tool in the route",
        ],
        typical_process_area="Cleanroom / Defectivity",
    ),
}

# Selected mixed-type compound hints (combinations that co-occur in MixedWM38)
_MIXED_HINTS: dict[frozenset[str], str] = {
    frozenset({"Center", "Edge-Ring"}): (
        "Center+Edge-Ring together often points to a global radial non-uniformity "
        "(deposition/etch rate vs radius) rather than two independent causes."
    ),
    frozenset({"Loc", "Scratch"}): (
        "Loc+Scratch suggests a single mechanical handling event that both gouged "
        "and deposited debris - inspect one tool's transfer path first."
    ),
    frozenset({"Edge-Loc", "Edge-Ring"}): (
        "Edge-Loc+Edge-Ring is consistent with an edge-ring defect that is worse at "
        "one azimuth - re-qualify the ring and check seating."
    ),
}


def triage(detected_patterns: list[str]) -> dict:
    """Return triage entries for the detected patterns plus any compound hints."""
    entries = [_RULES[p] for p in detected_patterns if p in _RULES]
    hints = []
    detected_set = set(detected_patterns)
    for combo, hint in _MIXED_HINTS.items():
        if combo.issubset(detected_set):
            hints.append(hint)
    if not entries:
        return {
            "patterns": [],
            "summary": "No defect pattern detected above threshold (wafer reads as 'none').",
            "entries": [],
            "compound_hints": [],
        }
    return {
        "patterns": detected_patterns,
        "summary": f"{len(detected_patterns)} defect pattern(s) detected: {', '.join(detected_patterns)}.",
        "entries": [
            {
                "pattern": e.pattern,
                "candidate_causes": e.candidate_causes,
                "recommended_checks": e.recommended_checks,
                "typical_process_area": e.typical_process_area,
            }
            for e in entries
        ],
        "compound_hints": hints,
    }


def all_patterns() -> list[str]:
    return list(_RULES.keys())
