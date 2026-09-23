from authentication.auth import AuthService
from database.db import DatabaseManager


def test_signup_and_login_roundtrip(tmp_path):
    db = DatabaseManager(tmp_path / "auth.sqlite3")
    service = AuthService(db)

    created, message = service.signup("Ada Lovelace", "ada@example.com", "securepass123")
    assert created is True
    assert "successful" in message.lower()

    login_success, login_message, user = service.login("ada@example.com", "securepass123")
    assert login_success is True
    assert "successful" in login_message.lower()
    assert user is not None
    assert user["email"] == "ada@example.com"


def test_signup_rejects_duplicate_email(tmp_path):
    db = DatabaseManager(tmp_path / "auth.sqlite3")
    service = AuthService(db)

    service.signup("Ada Lovelace", "ada@example.com", "securepass123")
    created, message = service.signup("Grace Hopper", "ada@example.com", "anotherpass123")

    assert created is False
    assert "already exists" in message.lower()


def test_login_rejects_wrong_password(tmp_path):
    db = DatabaseManager(tmp_path / "auth.sqlite3")
    service = AuthService(db)

    service.signup("Ada Lovelace", "ada@example.com", "securepass123")
    login_success, message, user = service.login("ada@example.com", "wrongpass123")

    assert login_success is False
    assert "incorrect password" in message.lower()
    assert user is None


def test_signup_and_login_normalise_email(tmp_path):
    db = DatabaseManager(tmp_path / "auth.sqlite3")
    service = AuthService(db)

    created, message = service.signup("Ada Lovelace", "  Ada@Example.com  ", "securepass123")
    assert created is True
    assert "successful" in message.lower()

    login_success, login_message, user = service.login(" ada@example.COM ", "securepass123")
    assert login_success is True
    assert "successful" in login_message.lower()
    assert user is not None
    assert user["email"] == "ada@example.com"
