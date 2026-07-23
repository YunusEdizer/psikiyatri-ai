"""Veritabanı motoru ve oturum yönetimi.

Bağlantı dizesi `Settings.resolved_database_url` üzerinden gelir
(varsayılan: ``<data>/psikiyatri.db`` SQLite). Motor tembel (lazy)
oluşturulur; testler `reset_engine()` ile farklı bir veritabanına
geçebilir.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings
from .models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Tekil motoru döndürür (gerekirse oluşturur)."""
    global _engine, _session_factory
    if _engine is None:
        url = get_settings().resolved_database_url
        connect_args = {}
        if url.startswith("sqlite"):
            # FastAPI istekleri farklı iş parçacıklarından gelebilir.
            connect_args["check_same_thread"] = False
        _engine = create_engine(url, connect_args=connect_args)
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db() -> None:
    """Tabloları oluşturur (varsa dokunmaz) ve hafif şema yamaları uygular."""
    engine = get_engine()
    if engine.url.drivername.startswith("sqlite") and engine.url.database:
        from pathlib import Path

        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    _apply_lightweight_migrations(engine)


def _apply_lightweight_migrations(engine) -> None:  # noqa: ANN001
    """Mevcut SQLite tablolarına eksik sütunları ekler (basit, idempotent).

    ``create_all`` var olan tablolara sütun EKLEMEZ; prototipte tam bir göç
    aracı yerine, eksik ``interviews`` sütunlarını ``ALTER TABLE ADD COLUMN``
    ile ekleriz (SQLite destekler). Kayıp veri olmaz.
    """
    if not engine.url.drivername.startswith("sqlite"):
        return
    from sqlalchemy import text

    wanted = {
        "consent": "BOOLEAN DEFAULT 0",
        "consent_at": "VARCHAR(32)",
        "clinician_agreement": "VARCHAR(16)",
        "clinician_assessment": "TEXT",
    }
    with engine.begin() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(interviews)"))}
        for col, ddl in wanted.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE interviews ADD COLUMN {col} {ddl}"))


def get_db() -> Iterator[Session]:
    """FastAPI bağımlılığı: istek başına oturum."""
    get_engine()
    assert _session_factory is not None
    session = _session_factory()
    try:
        yield session
    finally:
        session.close()


def reset_engine() -> None:
    """Motoru kapatıp sıfırlar (test izolasyonu / yapılandırma değişimi)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


__all__ = ["get_db", "get_engine", "init_db", "reset_engine"]
