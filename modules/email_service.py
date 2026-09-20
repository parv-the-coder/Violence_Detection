"""Email utilities for alert and summary delivery via Gmail SMTP."""

from __future__ import annotations

import json
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable, Optional

from config import (
    EMAIL_PASSWORD,
    EMAIL_RECIPIENT,
    EMAIL_RECIPIENT_FILE,
    EMAIL_SENDER,
    SMTP_HOST,
    SMTP_PORT,
)


def load_saved_recipient() -> str:
    """Return the recipient the dashboard last saved, falling back to the env value."""

    recipient_file = Path(EMAIL_RECIPIENT_FILE)
    if recipient_file.exists():
        try:
            saved = json.loads(recipient_file.read_text(encoding="utf-8")).get("recipient", "")
        except (json.JSONDecodeError, OSError):
            saved = ""
        if saved:
            return str(saved)
    return EMAIL_RECIPIENT


def save_recipient(recipient: str) -> None:
    """Persist the operator-chosen recipient address. Credentials are never stored."""

    recipient_file = Path(EMAIL_RECIPIENT_FILE)
    recipient_file.parent.mkdir(parents=True, exist_ok=True)
    recipient_file.write_text(json.dumps({"recipient": recipient.strip()}), encoding="utf-8")


@dataclass
class EmailSettings:
    sender: str = EMAIL_SENDER
    password: str = EMAIL_PASSWORD
    recipient: str = field(default_factory=load_saved_recipient)
    smtp_host: str = SMTP_HOST
    smtp_port: int = SMTP_PORT


def _normalize_settings(settings: EmailSettings) -> EmailSettings:
    settings.sender = settings.sender.strip()
    settings.password = settings.password.replace(" ", "").strip()
    settings.recipient = settings.recipient.strip()
    return settings


_ATTACHMENT_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".pdf": "application/pdf",
    ".csv": "text/csv",
}


def _build_message(subject: str, body: str, sender: str, recipient: str, attachments: Optional[Iterable[str]] = None) -> EmailMessage:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)

    for attachment in attachments or []:
        attachment_path = Path(attachment)
        if not attachment_path.exists():
            continue
        mime_type = _ATTACHMENT_MIME_TYPES.get(
            attachment_path.suffix.lower(), "application/octet-stream"
        )

        with attachment_path.open("rb") as file_handle:
            message.add_attachment(
                file_handle.read(),
                maintype=mime_type.split("/")[0],
                subtype=mime_type.split("/")[1],
                filename=attachment_path.name,
            )

    return message


def send_email(subject: str, body: str, attachments: Optional[Iterable[str]] = None, settings: Optional[EmailSettings] = None) -> bool:
    settings = settings or EmailSettings()
    settings = _normalize_settings(settings)
    if not settings.sender or not settings.password or not settings.recipient:
        return False

    message = _build_message(subject, body, settings.sender, settings.recipient, attachments)

    # A dead network or a rejected login must not abort an in-progress analysis,
    # so delivery failures are reported via the return value instead of raising.
    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(settings.sender, settings.password)
            smtp.send_message(message)
    except (smtplib.SMTPException, OSError) as error:
        print(f"[WARN] Email delivery failed: {error}")
        return False
    return True
