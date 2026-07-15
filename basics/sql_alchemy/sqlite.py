from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. Setup the local SQLite database
DATABASE_URL = "sqlite:///my_sqlite.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# 2. Define the table structure
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    email = Column(String)

# 3. Create the database file and table
Base.metadata.create_all(engine)

# 4. Interact with your new SQLite database
Session = sessionmaker(bind=engine)
session = Session()

# Add a user
new_user = User(name="Alex", email="alex@example.com")
session.add(new_user)
session.commit()

# Query the user back
user = session.query(User).first()
print(f"Saved User: {user.name} ({user.email})")

session.close()
