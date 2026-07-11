"""Kalıcı veri katmanı (SQLAlchemy).

Klinik akışın kayıtları — kullanıcı, hasta, görüşme, doktor notu ve
analiz sonucu — ilişkisel veritabanında tutulur. Varsayılan olarak yerel
SQLite dosyası kullanılır; üretimde ``PAI_DATABASE_URL`` ile PostgreSQL
gibi bir sunucuya yönlendirilir (bkz. `config.Settings.database_url`).

KVKK notu: Tablolarda doğrudan tanımlayıcı (ad, TC, dosya no) saklanmaz;
hasta yalnızca takma-kimlik (pseudonym) ile temsil edilir ve serbest
metin alanları yazılmadan önce maskelenir.
"""

from .database import get_db, get_engine, init_db, reset_engine
from .models import AnalysisResult, Base, DoctorNote, Interview, Patient, Role, User

__all__ = [
    "AnalysisResult",
    "Base",
    "DoctorNote",
    "Interview",
    "Patient",
    "Role",
    "User",
    "get_db",
    "get_engine",
    "init_db",
    "reset_engine",
]
