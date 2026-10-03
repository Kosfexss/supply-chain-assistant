"""Approval-gated SMTP delivery for persisted supplier purchase drafts."""

from __future__ import annotations

import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage

from .database import Database


@dataclass(frozen=True)
class SMTPSettings:
    host: str
    port: int
    sender: str
    username: str | None
    password: str | None
    use_ssl: bool

    @classmethod
    def from_environment(cls) -> "SMTPSettings":
        host = os.environ.get("SMTP_HOST", "").strip()
        sender = os.environ.get("SMTP_FROM", "").strip()
        if not host or not sender:
            raise ValueError("Set SMTP_HOST and SMTP_FROM before sending email")
        username = os.environ.get("SMTP_USERNAME", "").strip() or None
        password = os.environ.get("SMTP_PASSWORD") or None
        if bool(username) != bool(password):
            raise ValueError("Set both SMTP_USERNAME and SMTP_PASSWORD, or neither")
        return cls(
            host=host,
            port=int(os.environ.get("SMTP_PORT", "587")),
            sender=sender,
            username=username,
            password=password,
            use_ssl=os.environ.get("SMTP_USE_SSL", "false").lower() in {"1", "true", "yes"},
        )


class EmailDispatcher:
    def __init__(self, database: Database, settings: SMTPSettings | None = None) -> None:
        self.database = database
        self.settings = settings or SMTPSettings.from_environment()

    def send_approved_order(self, order_id: int) -> None:
        order = self.database.get_order(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} does not exist")
        if order["status"] != "approved":
            raise ValueError("Only an approved, unsent purchase draft can be emailed")
        if order["recipient"].lower().endswith(".example"):
            raise ValueError("Replace the sample .example supplier address before sending")

        message = EmailMessage()
        message["From"] = self.settings.sender
        message["To"] = order["recipient"]
        message["Subject"] = order["subject"]
        message.set_content(order["body"])

        try:
            self._deliver(message)
            self.database.mark_order_sent(order_id)
        except Exception as error:
            self.database.mark_order_failed(
                order_id, f"{type(error).__name__}: {error}"
            )
            raise

    def _deliver(self, message: EmailMessage) -> None:
        context = ssl.create_default_context()
        if self.settings.use_ssl:
            with smtplib.SMTP_SSL(
                self.settings.host,
                self.settings.port,
                context=context,
                timeout=20,
            ) as server:
                self._authenticate(server)
                server.send_message(message)
            return

        with smtplib.SMTP(
            self.settings.host, self.settings.port, timeout=20
        ) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            self._authenticate(server)
            server.send_message(message)

    def _authenticate(self, server: smtplib.SMTP | smtplib.SMTP_SSL) -> None:
        if self.settings.username and self.settings.password:
            server.login(self.settings.username, self.settings.password)