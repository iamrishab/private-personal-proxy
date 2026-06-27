"""LLM structured output schema for Ollama structured JSON chat.

The assistant produces a plain-language answer first; this lightweight schema is
used by the synthesis pass to distil that answer into the small amount of
structured metadata the UI shows — crisp next steps and, when the question is
under-specified, a few clarifying questions.
"""

from pydantic import BaseModel, Field


class LLMAdvisoryOutput(BaseModel):
    """Target schema for the Ollama structured JSON synthesis pass."""

    recommendations: list[str] = Field(
        default_factory=list,
        description=(
            "Zero to five short, concrete next steps the user can take, phrased as "
            "crisp imperative bullet points. Leave empty when no action is needed."
        ),
    )
    needs_clarification: bool = Field(
        default=False,
        description=(
            "True when the question is too vague or missing key detail to answer "
            "well, so the user should be asked for more information first."
        ),
    )
    clarification_questions: list[str] = Field(
        default_factory=list,
        description=(
            "Focused questions to ask when needs_clarification is true. "
            "Empty otherwise."
        ),
    )
