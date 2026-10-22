"""
Standard envelope definition for board messages using Pydantic.
Following Roadmap section 5.5.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    FINDING = "finding"
    QUESTION = "question"
    ANSWER = "answer"
    REQUEST = "request"
    CLAIM = "claim"


class MessageStatus(str, Enum):
    OPEN = "open"
    ANSWERED = "answered"
    DONE = "done"
    EXPIRED = "expired"


def generate_msg_id() -> str:
    """Generates an id formatted as msg- + 12 hex characters."""
    return f"msg-{uuid.uuid4().hex[:12]}"


class Message(BaseModel):
    id: str = Field(default_factory=generate_msg_id)
    time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    type: MessageType
    kind: str
    sender: str
    to: List[str] = Field(default_factory=list)
    topic: str
    incident_id: Optional[str] = None
    reply_to: Optional[str] = None
    level: int = Field(ge=0, le=3)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    summary: str = Field(max_length=160)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    payload: Dict[str, Any] = Field(default_factory=dict)
    status: MessageStatus = Field(default=MessageStatus.OPEN)
    expires_at: Optional[str] = None
    run_id: str = "dev"
