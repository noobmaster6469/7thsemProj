import sqlite3, pickle

def init_db():
    conn = sqlite3.connect('users.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            encoding BLOB
        )
    ''')
    conn.commit()
    conn.close()

def save_user(name, encoding):
    conn = sqlite3.connect('users.db')
    conn.execute("INSERT INTO users (name, encoding) VALUES (?, ?)",
                 (name, pickle.dumps(encoding)))
    conn.commit()
    conn.close()

def get_users():
    conn = sqlite3.connect('users.db')
    users = conn.execute("SELECT id, name FROM users").fetchall()
    conn.close()
    return users

def get_user_encoding(user_id):
    conn = sqlite3.connect('users.db')
    row = conn.execute("SELECT encoding FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return pickle.loads(row[0]) if row else None
