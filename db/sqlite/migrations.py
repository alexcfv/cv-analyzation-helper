from db.sqlite.connection import get_db

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                uuid TEXT PRIMARY KEY,
                profile_source TEXT,
                profile JSON
            )
        """)
