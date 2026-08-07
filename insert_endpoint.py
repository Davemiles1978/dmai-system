#!/usr/bin/env python3
"""
Insert the /api/admin/create-insights-table endpoint into dmai_core_complete.py
"""

from pathlib import Path

FILE = Path("dmai_core_complete.py")

# Read the file
with open(FILE, "r") as f:
    content = f.read()

# Check if endpoint already exists
if "/api/admin/create-insights-table" in content:
    print("Endpoint already exists")
    exit(0)

# Find a good insertion point (before the if __name__ block)
insert_marker = 'if __name__ == "__main__":'
insert_point = content.find(insert_marker)

if insert_point == -1:
    print("Could not find insertion point")
    exit(1)

# Build the new endpoint
new_endpoint = '''
@app.route("/api/admin/create-insights-table", methods=["POST"])
def api_admin_create_insights_table():
    """Create the insights table and other missing tables."""
    if not _require_auth():
        return jsonify({"error": "Unauthorised"}), 401
    import sqlite3
    from pathlib import Path
    db_path = Path("data/dmai_knowledge.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    try:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_text TEXT NOT NULL,
                entity_type TEXT,
                entities TEXT,
                relationship TEXT,
                source_topic TEXT,
                target_topic TEXT,
                confidence REAL DEFAULT 0.5,
                source_title TEXT,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS capabilities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT,
                description TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        return jsonify({"ok": True, "message": "Tables created"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
'''

# Insert the endpoint before the marker
new_content = content[:insert_point] + new_endpoint + '\n\n' + content[insert_point:]

# Write the file
with open(FILE, "w") as f:
    f.write(new_content)

print("✅ Added /api/admin/create-insights-table endpoint")
