import re

# Regex Patterns for PII Detection & Sanitization
PAN_PATTERN = r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b'
AADHAAR_PATTERN = r'\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b'
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
PHONE_PATTERN = r'\b(?:(?:\+?91)|0)?[6-9]\d{9}\b'
ACCOUNT_FOLIO_PATTERN = r'\b\d{8,16}\b'


def sanitize_user_input(text: str) -> str:
    """
    Detects and scrubs sensitive PII (PAN, Aadhaar, Phone, Email, Bank Account / Folio)
    from incoming user prompts, replacing them with [REDACTED] tokens.
    """
    if not text:
        return ""

    sanitized = text
    # 1. Sanitize PAN Numbers
    sanitized = re.sub(PAN_PATTERN, "[REDACTED]", sanitized, flags=re.IGNORECASE)
    
    # 2. Sanitize Email Addresses
    sanitized = re.sub(EMAIL_PATTERN, "[REDACTED]", sanitized)
    
    # 3. Sanitize Phone Numbers
    sanitized = re.sub(PHONE_PATTERN, "[REDACTED]", sanitized)
    
    # 4. Sanitize Aadhaar Numbers
    sanitized = re.sub(AADHAAR_PATTERN, "[REDACTED]", sanitized)
    
    # 5. Sanitize Bank Account & Folio Numbers (8-16 digits)
    # Avoid redacting numbers that are part of scheme query years (e.g. 2026) or small amounts (500)
    def redact_numeric_seq(match):
        val = match.group(0)
        # Preserve common years (e.g. 2025, 2026)
        if len(val) == 4 and val.startswith("20"):
            return val
        return "[REDACTED]"

    sanitized = re.sub(ACCOUNT_FOLIO_PATTERN, redact_numeric_seq, sanitized)

    return sanitized


if __name__ == "__main__":
    sample = "My PAN is ABCDE1234F, phone is 9876543210, email is test@sbimf.com, folio is 123456789012."
    print("Original:", sample)
    print("Sanitized:", sanitize_user_input(sample))
