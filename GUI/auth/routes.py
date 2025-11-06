from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app, make_response,g 
from werkzeug.security import check_password_hash, generate_password_hash
import uuid
from services.decorator import permission_required
from db import db_transaction
import logging
import os
import services.user_service as us
import secrets
from datetime import datetime, timedelta
import pytz

auth = Blueprint('auth', __name__)

# # Specify the full path for the logs directory
# log_dir = os.path.join(os.getcwd(), 'logs')  # Using the current working directory
# log_file = 'app_errors.log'

# # Set up logging to the file in the 'logs' folder
# logging.basicConfig(
#     filename=os.path.join(log_dir, log_file),  # Full path for log file
#     level=logging.ERROR,  # Log level
#     format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
#     filemode='a'  # Append mode
# )

@auth.route('/login', methods=['GET', 'POST'])
def login():
    try:
        if getattr(g, 'user_id', None):
        # Already logged in, redirect away from login page
            flash('You are already logged in.', 'info')
            return redirect(url_for('home_redirect'))  # or wherever you want

        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            with db_transaction(commit=True) as cursor:
                if username == '' or password == '':
                    flash('Username and password cannot be empty', 'danger')
                    us.user_audit(cursor, None, "User Logged In", "Failed", "Username and password cannot be empty")
                    return render_template('login.html')

                if len(password)>= 8 and len(password) <= 20 and len(username) <= 100:                  
                    cursor.execute("SELECT password_hash, uuid, name, attempts, status FROM users WHERE username = %s", (username,))
                    user = cursor.fetchone()
                    if user:
                        if check_password_hash(user['password_hash'], password) and user['status'] not in ('Deactivated', 'Locked', 'Invited') and user['attempts'] < 5:
                            # Update status, reset attempts, and last login
                            cursor.execute("""UPDATE users SET status = 'Online', last_login = NOW(), attempts = 0  WHERE uuid = %s""", (user['uuid'],))

                            # Audit log for successful login
                            cursor.execute("""INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s) """, 
                                        (user['uuid'], 'User Logged In', 'Success', f'User {user["name"]} logged in successfully.'))

                            # for server-side session management
                            session_token = secrets.token_hex(32)  # 64-character token
                            # Define Singapore timezone
                            sg_tz = pytz.timezone('Asia/Singapore')
                            # Get current local time in Singapore
                            now_local = datetime.now(sg_tz)
                            # Set expiry 30 minutes from now
                            expiry = now_local + timedelta(minutes=30)
                            # Store in DB
                            cursor.execute("""
                                INSERT INTO user_sessions (user_id, session_token, expires_at, created_at, last_activity_at)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (user['uuid'], session_token, expiry, now_local, now_local))
                            flash(f'Welcome back, {user["name"]}', 'success')

                            resp = make_response(redirect(url_for('home_redirect')))
                            resp.set_cookie(
                                'session_token',
                                session_token,
                                httponly=True,
                                secure=False,  # Changed to False for HTTP access
                                samesite='Lax',
                                max_age=1800  # 30 minutes
                            )
                            return resp
                            
                        else:
                            # Invalid login
                            flash('Invalid username or password', 'danger')

                            # Increment attempt count
                            cursor.execute("UPDATE users SET attempts = attempts + 1 WHERE username = %s", (username,))

                            # Fetch updated attempts count
                            cursor.execute("SELECT attempts FROM users WHERE username = %s", (username,))
                            updated = cursor.fetchone()
                            attempts = updated['attempts'] if updated else 0

                            # Lock account if 5 or more failed attempts
                            if attempts >= 5 and user['status'] != 'Locked':
                                cursor.execute("UPDATE users SET status = 'Locked' WHERE username = %s", (username,))

                            # Audit log for failed login
                            cursor.execute("""INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s)""", 
                                        (user['uuid'], 'User Login Attempt', 'Failure', f'Failed login attempt for user {user["name"]}.'))
                else:
                    # User not found
                    flash('Invalid username or password', 'danger')    
    
    except Exception as e:
        # logging.error(f"Error during login: {e}", exc_info=True)
        flash('An error occurred during login. Please try again later.', 'danger')
        return render_template('login.html')
    
    return render_template('login.html')

@auth.route('/unauthorized')
def unauthorized():
    return render_template('unauthorized.html'), 403

@auth.route('/create-user', methods=["POST"])
@permission_required('user.create')
def create_user():
    try:
        name = request.form['fullname']
        email = request.form['email']
        username = request.form['username']
        role = request.form.get('role')
        user_uuid = str(uuid.uuid4())

        if name == '' or email == '' or role == '' or username == '':
            flash('All fields are required', 'danger')
            return redirect(url_for('user'))
        
        if not us.validate_email(email):
            flash('Invalid email format', 'danger')
            return redirect(url_for('user'))
        
        name = us.sanitize_input(name)
        email = us.sanitize_input(email)
        username = us.sanitize_input(username)

        if role == 'Administrator' or role == 'Super Administrator' or role == 'System Automation':
            flash('Invalid role selected', 'danger')
            return redirect(url_for('user'))

        # generate_password_hash(password) = request.form['password']
        with db_transaction(commit=True) as cursor:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            existing_user = cursor.fetchone()

            cursor.execute("SELECT id FROM roles WHERE role_name = %s", (role,))
            role_id = cursor.fetchone()

            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            existing_username = cursor.fetchone()

            if existing_user:
                flash('Username already exists', 'danger')
            
            elif not role_id:
                flash('Invalid role selected', 'danger')
            
            elif existing_username:
                flash('Username already exists', 'danger')
            else:
                changed_by = g.user_id
                # apply for role_assignment approval
                cursor.execute("INSERT INTO role_assignment_approval (assigned_to, requested_by, role_id, status) VALUES (%s, %s, %s, %s)", 
                                (user_uuid, changed_by, role_id['id'], "Pending Approval"))
                # get role_assignment_approval id
                cursor.execute("SELECT id FROM role_assignment_approval WHERE assigned_to = %s, requested_by = %s, status = %s, role_id = %s", 
                               (user_uuid, changed_by, "Pending Approval", role_id['id']))
                role_assignment_id = cursor.fetchone()
                # add into the user table
                cursor.execute("INSERT INTO users (name, email, username, status, role_id, created_at, uuid, role_assignment_approval_id) VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s)", 
                                (name, email, username,"Pending Approval", role_id['id'], user_uuid, role_assignment_id['id']))
                # add into the user_audit table
                cursor.execute("INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s)",
                                (changed_by, 'User Created', 'Success', f'User {name} with email {email} created successfully and pending approval.'))
                
                flash('Account created successfully', 'success')
   
    except Exception as e:
        # logging.error(f"Error during user creation: {e}", exc_info=True)
        flash('An error occurred while creating the user. Please try again later.', 'danger')
        return redirect(url_for('user'))

    return redirect(url_for('user'))

@auth.route('/edit-account', methods=["POST"])
@permission_required('user.edit')
def edit_account():
    try:
        userid = request.form['userId']
        full_name = request.form['fullname']
        with db_transaction(commit=True) as cursor:
            if 'user.assign_roles' in g.permissions:
                role = request.form.get('role')
                cursor.execute("SELECT id FROM roles WHERE role_name = %s", (role,))
                role_id = cursor.fetchone()
                cursor.execute("SELECT role_assignment_approval_id FROM users WHERE uuid = %s", (userid,))
                role_assignment_id = cursor.fetchone()
                changed_by = g.user_id
                
                if role_assignment_id['role_assignment_approval_id'] is None:
                    # apply for role_assignment approval
                    cursor.execute("INSERT INTO role_assignment_approval (assigned_to, requested_by, role_id, status, requested_datetime) VALUES (%s, %s, %s, %s, NOW())", 
                                    (userid, changed_by, role_id['id'], "Pending Approval"))
                    # get role_assignment_approval id
                    cursor.execute("SELECT id FROM role_assignment_approval WHERE assigned_to = %s AND requested_by = %s AND status = %s AND role_id = %s", 
                               (userid, changed_by, "Pending Approval", role_id['id']))
                    new_role_assignment_id = cursor.fetchone()
                    print("this is the role assignment id ", new_role_assignment_id['id'])
                    # update users table
                    cursor.execute("UPDATE users SET role_assignment_approval_id = %s WHERE uuid = %s", 
                                    (new_role_assignment_id['id'], userid))
                else:
                    # update role_assignment_approval table
                    cursor.execute("UPDATE role_assignment_approval SET assigned_to = %s WHERE id = %s", 
                                    (userid, role_assignment_id['role_assignment_approval_id']))
            else:
                cursor.execute("UPDATE users SET name = %s WHERE uuid = %s", 
                                (full_name, userid))

            # add into the user_audit table
            cursor.execute("INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s)",
                            (changed_by, 'User Account Updated', 'Success', f'User {full_name} account updated successfully.'))
        flash('Account updated successfully', 'success')
    except Exception as e:
        # logging.error(f"Error during account edit: {e}", exc_info=True)
        flash('An error occurred while updating the account. Please try again later.', 'danger')
        return redirect(url_for('user'))
    
    return redirect(url_for('user'))

@auth.route('/assign-roles-to-admin', methods=["POST"])
@permission_required('user.assign_admin_roles')
def assign_admin_roles():
    try:
        userid = request.form['userId']
        with db_transaction(commit=True) as cursor:
            cursor.execute("SELECT id from roles WHERE role_name = 'Administrator'")
            role_id = cursor.fetchone()
            cursor.execute("UPDATE users SET role_id = %s WHERE uuid = %s", 
                            (role_id['id'], userid))
            cursor.execute("SELECT name FROM users WHERE uuid = %s", (userid,))
            user = cursor.fetchone()
            full_name = user['name']
            changed_by = g.user_id

            # add into the user_audit table
            cursor.execute("INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s)"
                        ,(changed_by, 'Role updated', 'Success', f'User {full_name} role updated successfully.'))
            flash('Role assigned to admin successfully', 'success')
    except Exception as e:
        # logging.error(f"Error during role assignment: {e}", exc_info=True)
        flash('An error occurred while assigning the role. Please try again later.', 'danger')
        return redirect(url_for('user'))
    
    return redirect(url_for('user'))

@auth.route('/deactivate-user', methods=['POST'])
@permission_required('user.deactivate')
def deactivate_user():
    try:
        user_id = request.form['userId']
        with db_transaction(commit=True) as cursor:
            cursor.execute("SELECT * FROM users WHERE uuid = %s", (user_id,))
            user = cursor.fetchone()

            cursor.execute("UPDATE users SET status = 'Deactivated' WHERE uuid = %s", (user_id,))
            changed_by = g.user_id
            # add into the user_audit table
            cursor.execute("INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s)",
                            (changed_by, 'User Deactivated', 'Success', f"User with ID {user['name']} deactivated successfully."))
    except Exception as e:
        # logging.error(f"Error during user deactivation: {e}", exc_info=True)
        flash('An error occurred while deactivating the user. Please try again later.', 'danger')
        return redirect(url_for('user'))
    return redirect(url_for('user'))

@auth.route('/reactivate-user', methods=['POST'])
@permission_required('user.activate')
def reactivate_user():
    try:
        user_id = request.form['userId']
        with db_transaction(commit=True) as cursor:
            cursor.execute("SELECT * FROM users WHERE uuid = %s", (user_id,))
            user = cursor.fetchone()
            cursor.execute("UPDATE users SET status = 'Offline' WHERE uuid = %s", (user_id,))
            changed_by = g.user_id
            # add into the user_audit table
            cursor.execute("INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s)",
                            (changed_by, 'User Reactivated', 'Success', f"User with ID {user['name']} reactivated successfully."))
    except Exception as e:
        # logging.error(f"Error during user reactivation: {e}", exc_info=True)
        flash('An error occurred while reactivating the user. Please try again later.', 'danger')
        return redirect(url_for('user'))
    
    return redirect(url_for('user'))

@auth.route('/unlock-user', methods=['POST'])
@permission_required('user.unlock')
def unlock_user():
    try:
        user_id = request.form['userId']
        with db_transaction(commit=True) as cursor:
            cursor.execute("SELECT * FROM users WHERE uuid = %s", (user_id,))
            user = cursor.fetchone()
            cursor.execute("UPDATE users SET status = 'Offline', attempts='0' WHERE uuid = %s", (user_id,))
            changed_by = g.user_id
            # add into the user_audit table
            cursor.execute("INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s)",
                            (changed_by, 'User Unlocked', 'Success', f"User with ID {user['name']} unlocked successfully."))
    except Exception as e:
        # logging.error(f"Error during user unlock: {e}", exc_info=True)
        flash('An error occurred while unlocking the user. Please try again later.', 'danger')
        return redirect(url_for('user'))
    
    return redirect(url_for('user'))

@auth.route('/approve-user', methods=['POST'])
@permission_required('user.approve')
def approve_user():
    try:
        user_id = request.form['userId']
        role_id = request.form['requestedRole-value']
        role_approval_id = request.form['modal-role-approval-id']
        changed_by = g.user_id

        with db_transaction(commit=True) as cursor:
            cursor.execute("SELECT * FROM users WHERE uuid = %s", (user_id,))
            user = cursor.fetchone()
            if not user:
                flash('User not found', 'danger')
                return redirect(url_for('user'))
            
            # get the role of the approver
            cursor.execute("""SELECT users.name, roles.role_name FROM users JOIN roles ON users.role_id = roles.id WHERE users.uuid = %s""", (changed_by,))
            approver = cursor.fetchone()
            cursor.execute("SELECT requested_by FROM role_assignment_approval WHERE id = %s", (role_approval_id,))
            requested_by = cursor.fetchone()['requested_by']
            
            if (approver['role_name'] == "Administrator" and changed_by == requested_by):
                cursor.execute("INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s)",
                                (changed_by, 'User Role Approved', 'Failed', f"{approver['name']} attempt to approve their own request."))    
                flash('You are not able to approve your own request', 'danger')

            elif (approver['role_name'] == "Super Admin") or (approver['role_name'] == "Administrator" and changed_by != requested_by):
                # if the account was just created
                if user['status'] == "Pending Approval":
                    cursor.execute("UPDATE users SET status = 'Invited' WHERE uuid = %s", (user_id,))
                    # send the invite link
        
                # remove the role assignment id from the users table
                cursor.execute("UPDATE users SET role_id = %s, role_assignment_approval_id = NULL WHERE uuid = %s", (role_id, user_id))
                # edit the status inside the role assignment
                cursor.execute("UPDATE role_assignment_approval SET status = 'Approved' WHERE id = %s", (role_approval_id,))
                # add into the user_audit table
                cursor.execute("INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s)",
                                (changed_by, 'Role Approved', 'Success', f"{user['name']} role is approved successfully."))
                flash('User role is approved successfully', 'success')

                # invalidate the current user session
                cursor.execute("DELETE FROM user_sessions WHERE user_id = %s", (user_id,))
            else:
                # add into the user_audit table
                cursor.execute("INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s)",
                                (changed_by, 'User Role Rejected', 'Failed', f"{approver['name']} attempt to reject {user['name']} role.")) 
                flash('You are not authorized to reject this user', 'danger')
        
    except Exception as e:
        # logging.error(f"Error during user approval: {e}", exc_info=True)
        flash('An error occurred while approving the user. Please try again later.', 'danger')
        return redirect(url_for('user'))
    
    return redirect(url_for('user'))

@auth.route('/reject-user', methods=['POST'])
@permission_required('user.approve')
def reject_user():
    try:
        user_id = request.form['userId']
        changed_by = g.user_id
        role_approval_id = request.form['modal-role-reject-id']
        with db_transaction(commit=True) as cursor:
            cursor.execute("SELECT * FROM users WHERE uuid = %s", (user_id,))
            user = cursor.fetchone()
            if not user:
                flash('User not found', 'danger')
                return redirect(url_for('user'))
            
            # count the number of users that have Super Admin role that is not the user that is rejecting
            cursor.execute("SELECT COUNT(*) FROM users WHERE role_id = (SELECT id FROM roles WHERE role_name = 'Super Admin') AND uuid != %s", (changed_by,))
            super_admin_count = cursor.fetchone()['count']
            
            # get the role of the approver
            cursor.execute("""SELECT users.name, roles.role_name FROM users JOIN roles ON users.role_id = roles.id WHERE users.uuid = %s""", (changed_by,))
            approver = cursor.fetchone()
            cursor.execute("SELECT requested_by FROM role_assignment_approval WHERE id = %s", (role_approval_id,))
            requested_by = cursor.fetchone()['requested_by']
            
            if (approver['role_name'] == "Administrator" and changed_by == requested_by):
                us.user_audit(cursor, changed_by, "User Role Rejected", "Failed", f"{approver['name']} attempt to reject their own request.")     
                flash('You are not able to reject your own request', 'danger')

            elif (approver['role_name'] == "Super Admin") or (approver['role_name'] == "Administrator" and changed_by != requested_by):
            
                if user['status'] == "Pending Approval":
                    cursor.execute("UPDATE users SET status = 'Rejected' WHERE uuid = %s", (user_id,))

                # edit the status inside the role assignment
                cursor.execute("UPDATE role_assignment_approval SET status = 'Rejected' WHERE id = %s", (role_approval_id,))

                # add into the user_audit table
                us.user_audit(cursor, changed_by, "User Role Rejected", "Success", f"{user['name']} role is rejected successfully.")        
                flash('User role rejected successfully', 'success')
            
            else:
                # add into the user_audit table
                us.user_audit(cursor, changed_by, "User Role Rejected", "Failed", f"{approver['name']} attempt to reject {user['name']} role.")
                flash('You are not authorized to reject this user', 'danger')
    except Exception as e:
        # logging.error(f"Error during user rejection: {e}", exc_info=True)
        flash('An error occurred while rejecting the user. Please try again later.', 'danger')
        return redirect(url_for('user'))
    
    return redirect(url_for('user'))


@auth.route('/create-role', methods=["POST"])
@permission_required('role.create')
def create_role():
    try:
        role_name = request.form['roleName']
        description = request.form['description']
        selected_permissions = request.form.getlist('permissions')
        with db_transaction(commit=True) as cursor:
            cursor.execute("SELECT * FROM roles WHERE role_name = %s", (role_name,))
            existing_role = cursor.fetchone()
            if existing_role:
                flash('Role already exists', 'danger')
            else:
                cursor.execute("INSERT INTO roles (role_name, description, `default`, status) VALUES (%s, %s, %s, %s)", (role_name, description, False, "Pending"))
                cursor.execute("SELECT id FROM roles WHERE role_name = %s", (role_name,))
                role_id = cursor.fetchone()
                if role_id:
                    for permission in selected_permissions:
                        # get the permission id from the name
                        cursor.execute("SELECT id, is_sensitive FROM permissions WHERE permission_name = %s", (permission,))
                        permission_id = cursor.fetchone()
                        if permission_id and permission_id['is_sensitive'] == 0:  # only insert non-sensitive permissions
                            # insert into roles_permissions table
                            cursor.execute("INSERT INTO roles_permissions (role_id, permission_id) VALUES (%s, %s)", (role_id['id'], permission_id['id']))
                
                changed_by = g.user_id
                # add into the user_audit table
                cursor.execute("INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s)",
                                (changed_by, 'Role Created', 'Success', f'Role {role_name} created successfully.'))
                flash('Role created successfully', 'success')
    except Exception as e:
        # logging.error(f"Error during role creation: {e}", exc_info=True)
        flash('An error occurred while creating the role. Please try again later.', 'danger')
        return redirect(url_for('user_role'))
    
    return redirect(url_for('user_role'))

@auth.route("/approve-role", methods=['POST'])
@permission_required('role.approve')
def approve_role():
    try:
        role_name = request.form['roleName']
        user_id = g.user_id
        with db_transaction(commit=True) as cursor:
            # update the role status to approved
            cursor.execute("""UPDATE roles SET status = 'Approved' WHERE role_name = %s;""", (role_name,))
            # update the role audit
            cursor.execute("""INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s);""", (user_id, "Role Approved", "Success", f'Role {role_name} approved successfully.'))
            flash('Role approved successfully', 'success')
    except Exception as e:
        # logging.error(f"Error during role approval: {e}", exc_info=True)
        flash('An error occurred while approving the role. Please try again later.', 'danger')
        return redirect(url_for('user_role'))
    return redirect(url_for('user_role'))

@auth.route("/reject-role", methods=['POST'])
@permission_required('role.approve')
def reject_role():
    try:
        role_name = request.form['roleName']
        user_id = g.user_id
        with db_transaction(commit=True) as cursor:

        # update the role status to rejected
            cursor.execute("""UPDATE roles SET status = 'Rejected' WHERE role_name = %s;""", (role_name,))
            # update the role audit
            cursor.execute("""INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s);""", (user_id, "Role Rejected", "Success", f'Role {role_name} rejected successfully.'))
            flash('Role rejected successfully', 'success')
    except Exception as e:
        # logging.error(f"Error during role rejection: {e}", exc_info=True)
        flash('An error occurred while rejecting the role. Please try again later.', 'danger')
        return redirect(url_for('user_role'))
    return redirect(url_for('user_role'))

@auth.route('/edit-role', methods=['POST'])
@permission_required('role.edit')
def edit_role():
    try:
        role_name = request.form['editRoleName']
        description = request.form['editRoleDescription']
        selected_permissions = request.form.getlist('editPermissions')
        role_id = request.form['editRoleId']
        with db_transaction(commit=True) as cursor:
            # check if the role exists and it is not the default role
            cursor.execute("SELECT id, `default` FROM roles WHERE id = %s", (role_id,))
            role = cursor.fetchone()
            if not role or role['default']:
                flash('Role cannot be edited', 'danger')
                return redirect(url_for('user_role'))

            cursor.execute("UPDATE roles SET role_name = %s, description = %s WHERE id = %s", (role_name, description, role_id))
            cursor.execute("DELETE FROM roles_permissions WHERE role_id = %s", (role_id,))
            for permission in selected_permissions:
                # get the permission id from the name
                cursor.execute("SELECT id, is_sensitive FROM permissions WHERE description = %s", (permission,))
                permission_id = cursor.fetchone()
                if permission_id and permission_id['is_sensitive'] == 0:  # only insert non-sensitive permissions
                    # insert into roles_permissions table
                    cursor.execute("INSERT INTO roles_permissions (role_id, permission_id) VALUES (%s, %s)", (role_id, permission_id['id']))
            
            changed_by = g.user_id
            # add into the user_audit table
            cursor.execute("INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s)",
                            (changed_by, 'Role Updated', 'Success', f'Role {role_name} updated successfully.'))
            flash('Role updated successfully', 'success')
    except Exception as e:
        # logging.error(f"Error during role edit: {e}", exc_info=True)
        flash('An error occurred while editing the role. Please try again later.', 'danger')
        return redirect(url_for('user_role'))
    return redirect(url_for('user_role'))

@auth.route('/delete-role', methods=['POST'])
@permission_required('role.delete')
def delete_role():
    try:
        role_name = request.form['roleName']
        with db_transaction(commit=True) as cursor:
            # check if any users are assigned to this role
            cursor.execute("SELECT COUNT(*) AS user_count FROM users WHERE role_id = (SELECT id FROM roles WHERE role_name = %s)", (role_name,))
            user_count = cursor.fetchone()['user_count']

            if user_count > 0:
                flash('Cannot delete role with assigned users', 'danger')
                return redirect(url_for('user_role'))

            cursor.execute("SELECT id FROM roles WHERE role_name = %s", (role_name,))
            role_id = cursor.fetchone()

            if not role_id:
                flash('Role not found', 'danger')
                return redirect(url_for('user_role'))

            # Delete the role from roles_permissions
            cursor.execute("DELETE FROM roles_permissions WHERE role_id = %s", (role_id['id'],))
            
            # Delete the role from roles
            cursor.execute("DELETE FROM roles WHERE id = %s", (role_id['id'],))
            
            changed_by = g.user_id
            # add into the user_audit table
            cursor.execute("INSERT INTO user_audit (user_id, action_type, timestamp, outcome, notes) VALUES (%s, %s, NOW(), %s, %s)",
                            (changed_by, 'Role Deleted', 'Success', f'Role {role_name} deleted successfully.'))
            
            flash('Role deleted successfully', 'success')
    except Exception as e:
        # logging.error(f"Error during role deletion: {e}", exc_info=True)
        flash('An error occurred while deleting the role. Please try again later.', 'danger')
        return redirect(url_for('user_role'))
    
    return redirect(url_for('user_role'))

@auth.route('/logout', methods=['POST'])
def logout():
    try: 
        session_token = request.cookies.get('session_token')
        if not session_token:
            flash('You are not logged in.', 'info')
            return redirect(url_for('login'))  
        
        # add into the user_audit table
        with db_transaction(commit=True) as cursor:
            user_id = g.user_id
            us.user_audit(cursor, user_id, 'User Logged Out', 'Success', 'User logged out successfully.')
            cursor.execute("UPDATE users SET status = 'Offline' WHERE uuid = %s", (user_id,))

            token = request.cookies.get('session_token')
            print(token)
            if token:
                cursor.execute("DELETE FROM user_sessions WHERE session_token = %s", (token,))
            resp = make_response(redirect(url_for('auth.login')))
            resp.delete_cookie('session_token')
            flash('You have been logged out.', 'info')
            return resp
        
    except Exception as e:
        # logging.error(f"Error during logout: {e}", exc_info=True)
        flash('An error occurred while logging out. Please try again later.', 'danger')
        response = make_response(redirect(url_for('auth.login')))
        response.delete_cookie('session_token')  # Delete session cookie explicitly
        session.clear()  # Clear session data
        return response
    return response
