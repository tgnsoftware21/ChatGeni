import pyodbc
from flask import current_app
import json

def get_db_connection(db_name):
    db_config = current_app.config["DATABASE_CONNECTIONS"][db_name]
    conn = pyodbc.connect(
        'DRIVER=' + current_app.config["SQL_DRIVER"] + ';' +
        'SERVER=' + db_config["server"] + ';' +
        'PORT=1433;' +
        'DATABASE=' + db_config["database"] + ';' +
        'UID=' + db_config["username"] + ';' +
        'PWD=' + db_config["password"] + ';' +
        'TrustServerCertificate=yes;' +
        'Encrypt=no'
    )

    print(conn)
    return conn

def query_database(db_name,query):
    print(query)
    conn = get_db_connection(db_name)
    #print(query)
    cursor = conn.cursor()
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    #print(columns)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # Convert all values to strings
    results_list = [dict(zip(columns, map(str, result))) for result in results]
    
    # Print each row as a dictionary
    #for result_dict in results_list:
        #print(result_dict)
    
    return json.dumps(results_list)
