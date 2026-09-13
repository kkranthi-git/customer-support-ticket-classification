import re


def combine_text(subject: str, body: str) -> str:
    """
    Combine ticket subject and body into one text field.
    """

    subject = "" if subject is None else str(subject)
    body = "" if body is None else str(body)

    text = f"{subject} {body}".strip()

    return text


def clean_text(text: str) -> str:
    """
    Basic text cleaning for NLP models.
    """

    text = str(text).lower()

    # Replace multiple whitespace characters
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def prepare_ticket_text(subject: str, body: str) -> str:
    """
    Prepare a customer support ticket for prediction.
    """

    text = combine_text(subject, body)

    text = clean_text(text)

    return text