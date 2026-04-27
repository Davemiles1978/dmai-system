#!/usr/bin/env python3
"""Insert genealogy API routes into dmai_core_complete.py"""

with open('dmai_core_complete.py', 'r') as f:
    lines = f.readlines()

# Find insertion point: after syllabus_status route, before 3D BRAIN section
insert_at = None
for i, line in enumerate(lines):
    if "# 3D BRAIN NETWORK VISUALIZATION" in line and i > 6980:
        insert_at = i
        break

if not insert_at:
    print("ERROR: Could not find insertion point")
    exit(1)

genealogy_route = '''
        # ============================================================================
        # AI GENEALOGY API - Track AI system versions, predict next capabilities
        # ============================================================================
        
        @self.app.route('/api/genealogy/systems', methods=['GET'])
        def genealogy_systems():
            """List all tracked AI systems with version counts"""
            try:
                import sqlite3
                conn = sqlite3.connect(str(self.evolution.si_core.sqlite.db_path))
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT s.name, s.organization, s.category, COUNT(v.id) as version_count, "
                    "s.first_release_date, s.description "
                    "FROM ai_systems s "
                    "LEFT JOIN system_versions v ON s.id = v.system_id "
                    "GROUP BY s.id ORDER BY s.name"
                )
                systems = []
                for row in cursor.fetchall():
                    systems.append({
                        "name": row[0], "organization": row[1], "category": row[2],
                        "versions": row[3], "first_release": row[4], "description": row[5]
                    })
                conn.close()
                return jsonify({"systems": systems, "total": len(systems)})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/genealogy/versions/<system_name>', methods=['GET'])
        def genealogy_versions(system_name):
            """Get version timeline for a specific AI system"""
            try:
                import sqlite3, json
                conn = sqlite3.connect(str(self.evolution.si_core.sqlite.db_path))
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT v.version_name, v.release_date, v.architecture, "
                    "v.context_window, v.modalities, v.key_additions "
                    "FROM system_versions v JOIN ai_systems s ON v.system_id = s.id "
                    "WHERE LOWER(s.name) LIKE ? ORDER BY v.release_date",
                    (f"%{system_name.lower()}%",)
                )
                versions = []
                for row in cursor.fetchall():
                    versions.append({
                        "version": row[0], "release_date": row[1], "architecture": row[2],
                        "context_window": row[3],
                        "modalities": json.loads(row[4]) if row[4] else [],
                        "key_additions": json.loads(row[5]) if row[5] else []
                    })
                conn.close()
                return jsonify({"system": system_name, "versions": versions, "count": len(versions)})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/genealogy/convergence', methods=['GET'])
        def genealogy_convergence():
            """Analyze convergence patterns across all tracked AI systems"""
            try:
                import sqlite3, json
                from collections import Counter
                conn = sqlite3.connect(str(self.evolution.si_core.sqlite.db_path))
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT s.name, v.version_name, v.key_additions, v.modalities, "
                    "v.context_window, v.release_date "
                    "FROM system_versions v JOIN ai_systems s ON v.system_id = s.id "
                    "WHERE v.release_date = ("
                    "SELECT MAX(v2.release_date) FROM system_versions v2 "
                    "WHERE v2.system_id = v.system_id)"
                )
                capabilities_counter = Counter()
                modality_counter = Counter()
                systems_latest = []
                for row in cursor.fetchall():
                    key_adds = json.loads(row[2]) if row[2] else []
                    mods = json.loads(row[3]) if row[3] else []
                    for ka in key_adds:
                        capabilities_counter[ka] += 1
                    for m in mods:
                        modality_counter[m] += 1
                    systems_latest.append({
                        "name": row[0], "latest_version": row[1],
                        "context_window": row[4], "release_date": row[5]
                    })
                conn.close()
                convergence = [cap for cap, count in capabilities_counter.most_common() if count >= 4]
                divergence = [cap for cap, count in capabilities_counter.most_common() if count == 1]
                contexts = [s["context_window"] for s in systems_latest if s["context_window"]]
                return jsonify({
                    "systems": systems_latest,
                    "convergence_zone": convergence,
                    "divergence_zone": divergence,
                    "shared_modalities": [m for m, c in modality_counter.most_common() if c >= 4],
                    "avg_context_window": sum(contexts) // len(contexts) if contexts else 0
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
'''

lines.insert(insert_at, genealogy_route)

with open('dmai_core_complete.py', 'w') as f:
    f.writelines(lines)

print(f"Genealogy API routes inserted at line {insert_at + 1}")
