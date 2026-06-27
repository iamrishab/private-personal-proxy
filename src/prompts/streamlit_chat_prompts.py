"""Prompt templates for the Streamlit chat experience."""

CHAT_MULTI_TURN_TEMPLATE = """This is an ongoing conversation. Use the full thread — including prior user \
details and assistant answers — when responding to the latest user message. Treat short follow-ups as \
continuations of the same topic unless the user clearly changes subject.

Conversation so far:
{history}

Latest user message:
{latest_message}"""

CHAT_HISTORY_LINE = "{role}: {content}"
