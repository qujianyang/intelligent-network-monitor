from flask import g, current_app
from contextlib import contextmanager
import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

def get_db():
    if 'db' not in g:
        g.db = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
    return g.db

def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        print("db closed")
        db.close()

@contextmanager
def db_transaction(dictionary=True, commit=False):
    db = get_db()
    cursor = db.cursor(dictionary=dictionary)
    try:
        yield cursor
        if commit:
            db.commit()
    except Exception as e:
        db.rollback()
        current_app.logger.error(f"Database transaction failed: {str(e)}")
        raise e
    finally:
        cursor.close()