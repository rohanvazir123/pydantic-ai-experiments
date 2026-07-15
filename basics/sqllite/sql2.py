import sqlite3

# 1. Connect directly to the file (Creates the file if it does not exist)
conn = sqlite3.connect("my_sqlite.db")
cursor = conn.cursor()

# Enable foreign key support in SQLite (Disabled by default!)
cursor.execute("PRAGMA foreign_keys = ON;")

# ==========================================
# 2. CREATE TABLES IF NOT EXISTS
# ==========================================
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT,
    user_id INTEGER,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);
""")
conn.commit()

# ==========================================
# 3. INSERT SAMPLE DATA SAFELY (Optional Check)
# ==========================================
# Let's seed a user and a post if the database is empty
cursor.execute("SELECT COUNT(*) FROM users")
if cursor.fetchone()[0] == 0:
    # Insert user
    cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("Alex", "alex@example.com"))
    alex_id = cursor.lastrowid # Gets the auto-incremented ID of Alex
    
    # Insert posts linked to Alex
    cursor.execute("INSERT INTO posts (title, content, user_id) VALUES (?, ?, ?)", 
                   ("First Steps", "Learning SQLite raw strings!", alex_id))
    cursor.execute("INSERT INTO posts (title, content, user_id) VALUES (?, ?, ?)", 
                   ("VS Code Trick", "This looks great in the extension!", alex_id))
    conn.commit()

# ==========================================
# 4. QUERY DATA WITH JOINS AND ? PLACEHOLDERS
# ==========================================
# Let's find all posts written by Alex (ID: 1)
sql_query = """
    SELECT users.name, posts.title, posts.content 
    FROM users 
    JOIN posts ON users.id = posts.user_id 
    WHERE users.name = ? AND users.id = ?
"""

# Pass data securely as a tuple matching the order of the ? marks
user_data = ("Alex", 1)
cursor.execute(sql_query, user_data)

# Fetch and print the joined results
rows = cursor.fetchall()
for row in rows:
    print(f"Author: {row[0]} | Title: {row[1]} | Content: {row[2]}")

conn.close()
