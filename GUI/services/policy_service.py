def policy_audit(cursor, policy_id, action, changed_by, outcome, notes):
    cursor.execute("""INSERT INTO policy_audit (policy_id, action_type, changed_by, changed_at, outcome, notes) VALUES (%s, %s, %s, NOW(), %s, %s);""", 
                   (policy_id, action, changed_by, outcome, notes))


def update_status(cursor, status, modified_by, policy_id):
    cursor.execute("""UPDATE policies SET status = %s,  modified_at = NOW(), modified_by = %s, WHERE id = %s;""", (status, modified_by, policy_id,))

def check_policy_status(cursor, policy_id):
    cursor.execute("""SELECT priority, policy_name, status, is_deleted FROM policies WHERE id = %s;""", (policy_id,))
    return cursor.fetchone()

def create_policy(cursor, policy_name, policy_type, scope, status, created_by, modified_by, priority, policy_description, purpose):
    cursor.execute("""INSERT INTO policies (policy_name, policy_type, scope, created_at, modified_at, status, created_by, modified_by, priority, description, purpose) VALUES (%s, %s, %s, NOW(), NOW(), %s, %s, %s, %s, %s, %s);""", 
                                (policy_name, policy_type, scope, status, created_by, modified_by, priority, policy_description, purpose))
    
def get_id_from_policy_name(cursor, policy_name):
    cursor.execute("""SELECT id FROM policies WHERE policy_name = %s;""", (policy_name,))
    return cursor.fetchone()

def create_policy_rules(cursor, policy_id, metric, condition, value, notes, severity, duration_window, raise_delay, clear_condition):
    cursor.execute("""INSERT INTO policy_rules (`policy_id`, `metric`, `condition`, `value`, `notes`, `severity`, `duration_window`, `raise_delay`, `clear_condition`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);""", 
                   (policy_id, metric, condition, value, notes, severity, duration_window, raise_delay, clear_condition))

def submit_weights(qber, visibility, key_rate, photon_loss):
    total = qber + visibility + key_rate + photon_loss
    if abs(total - 1.0) > 0.00001:  # Allow a tiny epsilon for float rounding
        return False
    else:
        return True