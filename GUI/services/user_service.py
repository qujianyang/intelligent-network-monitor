import re

def validate_email(email):
    # Regex to validate email format
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email)

def validate_password(password):
    # Password should be at least 8 characters, contain numbers, letters, and special characters
    if len(password) < 8 or len(password) > 20:
        return False
    if not re.search(r'\d', password):
        return False
    if not re.search(r'[A-Za-z]', password):
        return False
    if not re.search(r'[\W_]', password):  # Special characters
        return False
    
    return True

def sanitize_input(input_string):
    # Remove any harmful characters
    return re.sub(r'[<>]', '', input_string)


def user_audit(cursor, changed_by, action, outcome, notes):
    cursor.execute("INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s)",
                                (changed_by, action, outcome, notes)) 