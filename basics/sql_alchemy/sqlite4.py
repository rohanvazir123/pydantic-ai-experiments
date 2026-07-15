import email

from sqlalchemy import create_engine, text

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///my_sqlite.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# 1. Define raw SQL with named placeholders
query = text("SELECT * FROM users WHERE name = :user_name AND id = :user_id")

print("Executing raw SQL query with named placeholders...")
print(f"Query: {query}")

# 2. Execute and pass the data securely as key-value pairs
with engine.connect() as connection:
    # Data is automatically sanitized and safe from injection
    result = connection.execute(query, {"user_name": "Alex", "user_id": 1})
    
    for row in result:
        print(row.name, row.email)
