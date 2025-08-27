import sqlite3
import os

DB_PATH = 'data/missing_persons.db'

def init_db():
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            last_seen_location TEXT,
            date_missing TEXT,
            additional_info TEXT,
            contact_email TEXT,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_person(name, age, last_seen_location, date_missing, additional_info, contact_email, image_path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO persons (name, age, last_seen_location, date_missing, additional_info, contact_email, image_path) VALUES (?, ?, ?, ?, ?, ?, ?)',
              (name, age, last_seen_location, date_missing, additional_info, contact_email, image_path))
    conn.commit()
    conn.close()

def get_all_persons():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM persons')
    persons = c.fetchall()
    conn.close()
    return persons

def delete_person_by_id(person_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM persons WHERE id = ?', (person_id,))
    conn.commit()
    conn.close() 