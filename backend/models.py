from typing import Literal

from pydantic import AliasChoices, BaseModel, Field


class RouterDecision(BaseModel):
    route: Literal["paper_qa", "live_search", "verify_claim", "direct_answer"]


class RelevancyDecision(BaseModel):
    is_relevant: bool
    reason: str


class SupersedingPaper(BaseModel):
    title: str
    url: str
    # accept "reason" too — LLMs occasionally paraphrase the key
    summary: str = Field(validation_alias=AliasChoices("summary", "reason"))


class ClaimVerificationResult(BaseModel):
    is_superseded: bool
    verdict_summary: str
    superseding_papers: list[SupersedingPaper]


class PremiseCheck(BaseModel):
    """Structured verdict of the false-premise guard (paper Q&A)."""
    premise_refuted: bool = Field(
        description="True ONLY when the question asserts something the context contradicts AND the context states the correct counterpart."
    )
    correction: str = Field(
        description="Grounded corrective answer (empty string when premise_refuted is false). Cite with [Title, Page N] from the context."
    )
