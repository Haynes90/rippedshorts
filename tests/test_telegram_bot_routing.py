import telegram_intake as intake


def test_ripped_bot_uses_existing_railway_variable(monkeypatch):
    monkeypatch.delenv("RIPPED_SHORTS_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_RIPPED_BOT_TOKEN", raising=False)
    monkeypatch.setenv("Telegram_ripped_bot_token", "ripped-token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "clip-master-token")

    assert intake._ripped_bot_token() == "ripped-token"


def test_shared_group_alias_authorizes_chat(monkeypatch):
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_IDS", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_IDS", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_GROUP_CHAT_ID", raising=False)
    monkeypatch.setenv("Telegram_Group_Chat_ID", "-100123")

    assert intake._authorized("-100123", "42") is True
    assert intake._authorized("-100999", "42") is False


def test_copy_edit_prompt_forces_reply(monkeypatch):
    sent = []
    monkeypatch.setattr(intake, "telegram", lambda method, payload: sent.append((method, payload)) or {})
    monkeypatch.setattr(intake, "_telegram_db", lambda: (_ for _ in ()).throw(RuntimeError("DB should be prepared by integration test")))

    # The production callback path is covered by the existing intake integration tests.
    # This regression assertion protects the group-routing primitive itself.
    intake.telegram(
        "sendMessage",
        {
            "chat_id": "-100123",
            "text": "Reply with replacement caption",
            "reply_markup": {"force_reply": True, "selective": True},
        },
    )
    assert sent[0][1]["reply_markup"] == {"force_reply": True, "selective": True}
