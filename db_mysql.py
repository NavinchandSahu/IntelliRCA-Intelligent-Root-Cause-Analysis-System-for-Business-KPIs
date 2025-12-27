# db_mysql.py
# MySQL helper for RCA Project

import mysql.connector
from datetime import datetime

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "password",
    "database": "rca_system"
}

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

def create_upload_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS upload_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            file_name VARCHAR(255) NOT NULL,
            upload_date DATE NOT NULL,
            upload_time TIME NOT NULL,
            file_path VARCHAR(500) NOT NULL
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def save_upload(file_name, file_path):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) FROM upload_history
        WHERE file_name=%s AND upload_date=CURDATE()
    """, (file_name,))

    if cursor.fetchone()[0] == 0:
        now = datetime.now()
        cursor.execute("""
            INSERT INTO upload_history (file_name, upload_date, upload_time, file_path)
            VALUES (%s, %s, %s, %s)
        """, (file_name, now.date(), now.time(), file_path))
        conn.commit()

    cursor.close()
    conn.close()

def fetch_upload_history():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT file_name, upload_date, upload_time, file_path
        FROM upload_history
        ORDER BY upload_date DESC, upload_time DESC
    """)
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data



