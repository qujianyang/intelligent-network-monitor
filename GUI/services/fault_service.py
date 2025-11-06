# get the number of active faults and their statistics
def get_active_faults_stats(cursor):
    cursor.execute("""SELECT COUNT(*) as num_active_faults, SUM(CASE WHEN severity='Critical' THEN 1 ELSE 0 END) as critical, SUM(CASE WHEN severity='Warning' THEN 1 ELSE 0 END) as warning, SUM(CASE WHEN alert_id IS NOT NULL THEN 1 ELSE 0 END) as escalated FROM faults WHERE status != 'Resolved';""")
    active_faults = cursor.fetchone()
    return active_faults


# get the average time to resolve and acknowledge faults
def get_average_times(cursor):
    cursor.execute("""SELECT resolved_at, detected_at, acknowledged_at FROM faults where status = 'Resolved';""")
    resolved_faults = cursor.fetchall()
    if not resolved_faults:
        return 0.0, 0.0  # Prevent division by zero
    
    # calculate the total time to resolve and acknowledge
    total_time_to_resolve = 0
    total_time_to_acknowledge = 0
    for fault in resolved_faults:
        time_to_resolve = fault['resolved_at'] - fault['detected_at']
        time_to_acknowledge = fault['acknowledged_at'] - fault['detected_at']
        total_time_to_resolve += time_to_resolve.total_seconds()
        total_time_to_acknowledge += time_to_acknowledge.total_seconds()
    
    # convert back to hours, in 2dp
    average_time_to_resolve = round((total_time_to_resolve / 3600) / len(resolved_faults),2)
    average_time_to_acknowledge = round((total_time_to_acknowledge / 3600) / len(resolved_faults),2)
    return average_time_to_resolve, average_time_to_acknowledge


# get the number of faults for the last 5 days
def get_faults_last_5_days(cursor):
    cursor.execute("""SELECT date_series.date as dates, COUNT(faults.detected_at) AS num_of_faults FROM 
    (SELECT CURDATE() - INTERVAL a DAY AS date FROM (SELECT 0 AS a UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4) AS days) AS date_series
    LEFT JOIN faults ON DATE(faults.detected_at) = date_series.date
    GROUP BY date_series.date
    ORDER BY date_series.date ASC; """)
    active_faults_last_5_days = cursor.fetchall()
    dates = [row['dates'].strftime('%d-%m-%Y')for row in active_faults_last_5_days]
    num_of_faults = [row['num_of_faults']for row in active_faults_last_5_days]
    return num_of_faults, dates

# count the number of resolved faults for the last 5 days
def get_resolved_faults_last_5_days(cursor):
    cursor.execute("""SELECT date_series.date, COUNT(faults.resolved_at) AS num_of_resolved_faults FROM 
    (SELECT CURDATE() - INTERVAL a DAY AS date FROM (SELECT 0 AS a UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4) AS days) AS date_series
    LEFT JOIN faults ON DATE(faults.resolved_at) = date_series.date AND faults.status = 'Resolved'
    GROUP BY date_series.date
    ORDER BY date_series.date ASC; """)
    num_of_resolved_faults = cursor.fetchall()
    num_resolved_faults = [row['num_of_resolved_faults']for row in num_of_resolved_faults]
    return num_resolved_faults

def get_affected_components(cursor):
    # count the number of components that are affected
    cursor.execute("""SELECT component_type, component_id  FROM faults WHERE status != 'Resolved';""")
    num_of_components = cursor.fetchall()
    for component in num_of_components:
        if component['component_type'] == 'Link':
            cursor.execute("""SELECT link_id FROM links WHERE id = %s;""", (component['component_id'],))
            link = cursor.fetchone()
            component['component'] = link['link_id']
        else:
            cursor.execute("""SELECT node_id FROM nodes WHERE id = %s;""", (component['component_id'],))
            node = cursor.fetchone()
            component['component'] = node['node_id']

    # Count the occurrences of each component
    component_count = {}
    for component in num_of_components:
        comp = component['component']
        if comp in component_count:
            component_count[comp] += 1
        else:
            component_count[comp] = 1

    # Prepare the final list with components and their counts
    num_components = [{"component": component, "num_components": count} for component, count in component_count.items()]

    # Sort by the number of components in descending order
    num_components = sorted(num_components, key=lambda x: x['num_components'], reverse=True)
    top_3_components = num_components[:3]
    component_array = [item['component'] for item in top_3_components]
    component_count_array = [item['num_components'] for item in top_3_components]
    return component_array, component_count_array

def get_alarm_list(cursor):
    cursor.execute("""SELECT * FROM faults;""")
    faults = cursor.fetchall()
    for fault in faults:
        if fault['component_type'] == 'Link':
            cursor.execute("""SELECT link_id FROM links WHERE id = %s;""", (fault['component_id'],))
            link = cursor.fetchone()
            fault['component'] = link['link_id']
        else:
            cursor.execute("""SELECT node_id FROM nodes WHERE id = %s;""", (fault['component_id'],))
            node = cursor.fetchone()
            fault['component'] = node['node_id']
        
        if fault['acknowledged_by'] is not None:
            cursor.execute("""SELECT name FROM users WHERE uuid = %s;""", (fault['acknowledged_by'],))
            user = cursor.fetchone()
            fault['acknowledged_by'] = user['name']
        else:
            fault['acknowledged_by'] = "Not acknowledged"
        
        if fault['resolved_by'] is not None:
            cursor.execute("""SELECT name FROM users WHERE uuid = %s;""", (fault['resolved_by'],))
            user = cursor.fetchone()
            fault['resolved_by'] = user['name']
        else:
            fault['resolved_by'] = "Not resolved"
        
        if fault['assigned_to'] is not None:
            cursor.execute("""SELECT name FROM users WHERE uuid = %s;""", (fault['assigned_to'],))
            user = cursor.fetchone()
            fault['assigned_to_uuid'] = fault['assigned_to']
            fault['assigned_to'] = user['name']
            
        else:
            fault['assigned_to'] = "Not assigned"
    return faults


def get_users_with_permission(cursor, permission):
    cursor.execute("""SELECT users.name, users.uuid
FROM users
JOIN roles ON users.role_id = roles.id
JOIN roles_permissions ON roles.id = roles_permissions.role_id
JOIN permissions ON roles_permissions.permission_id = permissions.id
WHERE permissions.permission_name = %s AND users.status IN ('Online', 'Offline', 'Locked') AND roles.role_name != "Super Admin";
""", (permission,))
    users = cursor.fetchall()
    return users


def fault_log(cursor, fault_id, performed_by, action_type, outcome, notes):
    cursor.execute("""INSERT INTO fault_log (timestamp, action_type, fault_id, performed_by, outcome, notes)
                            VALUES (NOW(), %s, %s, %s, %s, %s);""", (action_type, fault_id, performed_by, outcome, notes))
 
def alert_log(cursor, alert_id, performed_by, action_type, outcome, notes):
    cursor.execute("""INSERT INTO alert_log (alert_id, performed_by, action_type, timestamp, outcome, notes)
                            VALUES (%s, %s, %s, NOW(), %s, %s);""", (alert_id, performed_by, action_type, outcome, notes))