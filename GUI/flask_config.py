from datetime import timedelta
import os
from flask_mail import Mail

class Config:
    # SECRET_KEY = os.urandom(24)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)
    SESSION_PROTECTION = 'strong'
    SECRET_KEY = 'your_secret_key_here'  # Required for CSRF tokens
    WTF_CSRF_TIME_LIMIT = None  # Optional: disables CSRF expiration timeout
    WTF_CSRF_ENABLED = True  # Enable CSRF protection

    # for mail server
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USE_SSL = False
    MAIL_USERNAME = 'youremail@gmail.com'
    MAIL_PASSWORD =  os.getenv('MAIL_PASSWORD') # Use environment variable for security
    MAIL_DEFAULT_SENDER = 'youremail@gmail.com'
    # MAIL_SUPPRESS_SEND = True  # For testing, emails won't be sent
