from sqlalchemy import String, txt, DateTime, Integer, Index, Computed, func
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class LogEntryFTS(Base):
    __tablename__ = "log_entries_fts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    client: Mapped[str] = mapped_column(String(50))
    version: Mapped[str] = mapped_column(String(20))
    identifier: Mapped[str] = mapped_column(String(50))
    raw_ua: Mapped[str] = mapped_column(Text)

    # Generated stored column combining searchable fields into a tsvector
    search_vector: Mapped[TSVECTOR] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(client, '') || ' ' || coalesce(identifier, '') || ' ' || coalesce(raw_ua, ''))",
            persisted=True,
        ),
    )

    __table_args__ = (
        # GIN index for fast full-text evaluation
        Index("ix_log_entries_fts_vector", search_vector, postgresql_using="gin"),
    )
