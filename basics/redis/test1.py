import redis

# Connect to your local Redis instance
# decode_responses=True automatically converts bytes to Python strings
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# =====================================================================
# 📦 HASHES (H) — Object Structuring
# =====================================================================
print("--- HASHES ---")

# 1. Store and Update a User Profile (HSET)
user_data = {
    "name": "Marcus",
    "email": "marcus@dev.com",
    "role": "editor",
    "status": "active"
}
r.hset("user:2088", mapping=user_data)

# 2. Fetch Only a Single Attribute (HGET)
email = r.hget("user:2088", "email")
print(f"User Email: {email}")  # Output: marcus@dev.com

# 3. Increment an Internal Field Value (HINCRBY)
# Adds 1 to the 'item:sku-99' field inside the cart hash
r.hincrby("cart:user:2088", "item:sku-99", 1)

# 4. Pull the Entire Object (HGETALL)
# Returns a native Python dictionary
full_profile = r.hgetall("user:2088")
print(f"Full Profile Dict: {full_profile}")


# =====================================================================
# 🎒 SETS (S) — Unique Unordered Collections
# =====================================================================
print("\n--- SETS ---")

# 1. Add Tags to a Blog Post (SADD)
# Notice "redis" is duplicated; Redis automatically ignores the second one
r.sadd("post:104:tags", "databases", "redis", "backend", "redis")

# 2. Check Group Membership (SISMEMBER)
# Returns a boolean (True/False) in Python
is_tester = r.sismember("group:beta_testers", "user:2088")
print(f"Is user a beta tester?: {is_tester}")

# 3. Find Mutual Connections (SINTER)
# Setup mock friend lists first
r.sadd("user:1001:friends", "alice", "bob", "marcus")
r.sadd("user:2088:friends", "charlie", "bob", "marcus")

mutual_friends = r.sinter("user:1001:friends", "user:2088:friends")
print(f"Mutual Friends (Set): {mutual_friends}")  # {'bob', 'marcus'}

# 4. Fetch All Unique Members (SMEMBERS)
all_tags = r.smembers("post:104:tags")
print(f"All Unique Tags: {all_tags}")


# =====================================================================
# 🏆 SORTED SETS (Z) — Unique Ordered Collections
# =====================================================================
print("\n--- SORTED SETS ---")

# 1. Insert Gaming Leaderboard Scores (ZADD)
# Python expects a dictionary mapping {member: score}
scores = {
    "Player_One": 4500,
    "SkyWalker": 8200,
    "PixelMage": 6100
}
r.zadd("leaderboard:gaming", mapping=scores)

# 2. Get the Top 3 High Scores (ZREVRANGE)
# withscores=True returns a list of tuples: [('member', score), ...]
top_players = r.zrevrange("leaderboard:gaming", 0, 2, withscores=True)
print(f"Top 3 Players: {top_players}")

# 3. Increment an Existing Score (ZINCRBY)
# Increases PixelMage's score by 500
new_score = r.zincrby("leaderboard:gaming", 3500, "PixelMage")
print(f"PixelMage New Score: {new_score}")
top_players = r.zrevrange("leaderboard:gaming", 0, 2, withscores=True)
print(f"Top 3 Players: {top_players}")

# 4. Remove a Banned User (ZREM)
print(f"Banned Player_One")
r.zrem("leaderboard:gaming", "Player_One")
top_players = r.zrevrange("leaderboard:gaming", 0, 2, withscores=True)
print(f"Top 3 Players: {top_players}")

