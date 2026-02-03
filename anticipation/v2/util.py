import hashlib


def get_md5_of_string(input_str: str) -> str:
    # assuming the input string uses utf-8 encoding
    return hashlib.md5(bytes(input_str, encoding="utf8")).hexdigest()
