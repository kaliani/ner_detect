import pandas as pd


def process_text(text, phones, emails):
    """
    Process the text and label the phone numbers and email addresses with NER tags.

    Args:
    text (str): The text to process.
    phones (list): A list of phone numbers to label.
    emails (list): A list of email addresses to label.

    Returns:
    A list of tuples, where each tuple contains a token and its NER label.
    """
    phone_tokens = []
    for phone in phones:
        start = text.find(phone)
        while start != -1:
            end = start + len(phone)
            phone_tokens.append((phone, start, end))
            start = text.find(phone, end)

    email_tokens = []
    for email in emails:
        start = text.find(email)
        while start != -1:
            end = start + len(email)
            email_tokens.append((email, start, end))
            start = text.find(email, end)

    tokens = text.split()
    labels = ['O'] * len(tokens)
    for token, start, end in phone_tokens + email_tokens:
        for i in range(len(tokens)):
            if start <= len(' '.join(tokens[:i+1]))-1 and end > len(' '.join(tokens[:i])):
                if start == len(' '.join(tokens[:i])):
                    labels[i] = f'B-{"PHONE" if token in phones else "EMAIL"}'
                else:
                    labels[i] = f'I-{"PHONE" if token in phones else "EMAIL"}'
    return list(zip(tokens, labels))


def label_text(texts, phones, emails):
    """
    Label the phone numbers and email addresses in a list of texts with NER tags.

    Args:
    texts (list): A list of texts to label.
    phones (list): A list of phone numbers to label.
    emails (list): A list of email addresses to label.

    Returns:
    A list of lists of tuples, where each tuple contains a token and its NER label.
    """
    labels = [process_text(text, phones, emails) for text in texts]
    return labels


def create_ner_dataframe(texts, phones, emails):
    """
    Create a pandas DataFrame with the text, phone number, email address, and NER labels.

    Args:
    texts (list): A list of texts to label.
    phones (list): A list of phone numbers to label.
    emails (list): A list of email addresses to label.

    Returns:
    A pandas DataFrame with the text, phone number, email address, and NER labels.
    """
    labels = label_text(texts, phones, emails)
    data = {'text': texts, 'phone': phones, 'email': emails, 'labels': labels}
    df = pd.DataFrame(data)
    return df
