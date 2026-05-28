import sqlite3
import json

db_path = 'data/dmai_knowledge.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all topics from database
cursor.execute('SELECT topic, name, stage, category, content FROM syllabus_content')
topics = cursor.fetchall()
conn.close()

print(f"Found {len(topics)} topics in database")

# Update the endpoint to include database lookup
with open('dmai_smart_endpoint.py', 'r') as f:
    content = f.read()

# Add database lookup function
db_lookup = '''
def get_from_database(question):
    """Look up topic in database"""
    import sqlite3
    conn = sqlite3.connect('data/dmai_knowledge.db')
    cursor = conn.cursor()
    question_lower = question.lower().strip()
    cursor.execute('SELECT topic, stage, category, content FROM syllabus_content WHERE topic = ? OR topic LIKE ?', 
                   (question_lower, f'%{question_lower}%'))
    result = cursor.fetchone()
    conn.close()
    return result
'''

# Insert after imports
if 'get_from_database' not in content:
    lines = content.split('\n')
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.startswith('smart_bp = Blueprint'):
            insert_pos = i + 2
            break
    lines.insert(insert_pos, db_lookup)
    content = '\n'.join(lines)

# Update the ask function to check database
old_check = "if matched_info:"
new_check = '''        # First check database
        db_result = get_from_database(question)
        if db_result:
            topic, stage, category, detailed_content = db_result
            return jsonify({
                "answer": detailed_content,
                "topic": topic,
                "stage": stage,
                "category": category,
                "mastery": "100%",
                "status": "success",
                "syllabus": True,
                "source": "database"
            })
        
        if matched_info:'''

content = content.replace(old_check, new_check)

with open('dmai_smart_endpoint.py', 'w') as f:
    f.write(content)

print("Updated endpoint to prioritize database content")
