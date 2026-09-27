"""decision_bakeoff: offline/shadow eval of decision architectures A0–A5."""

__version__ = "0.1.0"

PROBLEM_TYPES = (
    "route",
    "retrieve_rerank",
    "bon_verify",
    "triage",
    "guided_questionnaire",
)

ARCHITECTURES = ("A0", "A1", "A2", "A3", "A4", "A5")
