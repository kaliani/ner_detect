import re


def remove_char(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub("\\[|\\]|\(|\)", " ", text)  # noqa
    text = re.sub(r"\"|•|\'|\\n|“|”|\\t|\b|«|\|№\w*»", "", text)
    fin_text = " ".join(text.split())
    return fin_text


def parse_phone_number(text):
    """Extract phone numbers from text using regex and return as a list."""
    phone_regex = r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    return re.findall(phone_regex, text)


def parse_email(text):
    """Extract email addresses from text using regex and return as a list."""
    email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return re.findall(email_regex, text)


def labeled_text(text, phone_numbers, emails):
    """
    Find all phone numbers and emails in the text and return the tagged text for training the BERT NER model.
    """
    if not phone_numbers and not emails:
        return " ".join(["O"] * len(text.split()))

    tags = []
    for word in text.split():
        if word in phone_numbers:
            tags += ["B-PHONE"]
            tags += ["I-PHONE"] * (len(word.split()) - 1)
        elif word in emails:
            tags += ["B-EMAIL"]
            tags += ["I-EMAIL"] * (len(word.split()) - 1)
        else:
            tags += ["O"]

    return " ".join(tags)
