"""SQLModel tables. One file, four tables.

    users              accounts (roles come from the environment, never from this table)
    prediction_results every analysis ever run, signed-in or anonymous, with the full result JSON
    chat_logs          every assistant message (analysis_id / user_id are NULL for anonymous or general chat)
    analysis_reviews   reviewer sign-off, one row per reviewer per analysis
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import JSON, Column, ForeignKey, Integer, String, UniqueConstraint
from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    __tablename__ = "users"
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    name: str
    pw_hash: str
    created: str


class PredictionResult(SQLModel, table=True):
    __tablename__ = "prediction_results"
    id: str = Field(primary_key=True, max_length=32)
    user_id: Optional[int] = Field(default=None, sa_column=Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True))
    created: str = Field(index=True)
    pred_class: str
    confidence: float
    tumour_prob: Optional[float] = None
    mode: Optional[str] = None
    model_version: Optional[str] = None
    input_ext: Optional[str] = None
    result: dict = Field(sa_column=Column(JSON, nullable=False))


class ChatLog(SQLModel, table=True):
    __tablename__ = "chat_logs"
    id: Optional[int] = Field(default=None, primary_key=True)
    analysis_id: Optional[str] = Field(default=None, sa_column=Column(String(32), ForeignKey("prediction_results.id", ondelete="CASCADE"), index=True))
    user_id: Optional[int] = Field(default=None, sa_column=Column(Integer, ForeignKey("users.id", ondelete="SET NULL")))
    role: str  # 'user' | 'assistant'
    content: str
    mode: Optional[str] = None
    created: str


class Review(SQLModel, table=True):
    __tablename__ = "analysis_reviews"
    __table_args__ = (UniqueConstraint("analysis_id", "reviewer_id"),)
    id: Optional[int] = Field(default=None, primary_key=True)
    analysis_id: str = Field(sa_column=Column(String(32), ForeignKey("prediction_results.id", ondelete="CASCADE"), nullable=False, index=True))
    reviewer_id: int = Field(sa_column=Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False))
    verdict: str  # agree | disagree | uncertain
    corrected_class: Optional[str] = None
    note: Optional[str] = None
    created: str
