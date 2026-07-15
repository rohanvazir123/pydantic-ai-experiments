from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///my_sqlite.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# ==========================================
# 1. DEFINE TABLES WITH RELATIONSHIPS
# ==========================================
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    email = Column(String)
    
    # Links to the Post class. 'back_populates' updates both sides automatically.
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")

class Post(Base):
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(String)
    
    # Foreign Key connects this post to a specific user ID
    user_id = Column(Integer, ForeignKey('users.id'))
    
    # Links back to the User class
    author = relationship("User", back_populates="posts")

# Create tables in SQLite
Base.metadata.drop_all(engine) # Clear old tables for a clean run
Base.metadata.create_all(engine)

# ==========================================
# 2. INSERT SAMPLE DATA
# ==========================================
Session = sessionmaker(bind=engine)
session = Session()

# Create a user with posts attached directly in Python
alex = User(name="Alex", email="alex@example.com")
alex.posts = [
    Post(title="First Steps", content="Learning SQLAlchemy!"),
    Post(title="SQLite Magic", content="It works in VS Code!")
]

session.add(alex)
session.commit()

# ==========================================
# 3. HOW TO JOIN TABLES
# ==========================================

print("--- APPROACH A: Using the relationship helper (Implicit Join) ---")
# Because we defined relationship(), SQLAlchemy loads the posts automatically behind the scenes
user_record = session.query(User).filter_by(name="Alex").first()
for post in user_record.posts:
    print(f"Author: {user_record.name} | Post Title: {post.title}")


print("\n--- APPROACH B: Explicit SQL JOIN (Best for filtering by post attributes) ---")
# This creates an explicit SQL 'JOIN' query between users and posts
results = session.query(User, Post).join(Post, User.id == Post.user_id).all()

for user, post in results:
    print(f"User: {user.name} wrote the post: '{post.title}'")

session.close()
