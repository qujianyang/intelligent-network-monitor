from functools import wraps
from flask import session, redirect, url_for, flash, g

def permission_required(permission):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # If g.permissions was not set, user is not logged in
            if not hasattr(g, 'permissions'):
                flash('Access denied. No permission context.', 'danger')
                return redirect(url_for('auth.login'))

            if permission not in g.permissions:
                return redirect(url_for('auth.unauthorized'))

            return f(*args, **kwargs)
        return wrapper
    return decorator
