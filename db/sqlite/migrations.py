from db.connection import get_db

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                id TEXT PRIMARY KEY,
                profile_source_id TEXT,
                profile JSON
            )
        """)
