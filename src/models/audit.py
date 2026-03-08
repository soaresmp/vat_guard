"""
AuditLog model – immutable record of every action taken in VATGuard.
Critical for chain-of-custody evidence in fraud cases.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Actor
    actor: Mapped[str] = mapped_column(String(100), nullable=False, index=True)  # user / system
    actor_ip: Mapped[str | None] = mapped_column(String(45))
    actor_role: Mapped[str | None] = mapped_column(String(50))

    # Action
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    resource_id: Mapped[str | None] = mapped_column(String(100), index=True)

    # Detail
    description: Mapped[str | None] = mapped_column(Text)
    before_state: Mapped[dict | None] = mapped_column(JSONB)  # Snapshot before change
    after_state: Mapped[dict | None] = mapped_column(JSONB)   # Snapshot after change
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)

    # Integrity
    session_id: Mapped[str | None] = mapped_column(String(100))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    __table_args__ = (
        Index("ix_audit_actor_action", "actor", "action"),
        Index("ix_audit_resource", "resource_type", "resource_id"),
        Index("ix_audit_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<AuditLog {self.action} by={self.actor} resource={self.resource_type}/{self.resource_id}>"
