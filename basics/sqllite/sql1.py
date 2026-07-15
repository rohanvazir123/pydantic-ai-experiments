import sqlite3

# 1. Connect directly to the file
conn = sqlite3.connect("my_sqlite.db")
cursor = conn.cursor()

# 2. Write raw SQL with ? placeholders
sql_query = "SELECT * FROM users WHERE name = ? AND id = ?"

# 3. Pass data securely as a TUPLE matching the order of the ? marks
user_data = ("Alex", 1)
cursor.execute(sql_query, user_data)

# 4. Fetch the results
rows = cursor.fetchall()
for row in rows:
    print(row) # Prints as a standard Python tuple

conn.close()
