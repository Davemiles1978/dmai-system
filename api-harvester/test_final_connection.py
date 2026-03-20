import psycopg2

print('Testing PostgreSQL connection...')

# Use external hostname format
host = 'oregon-postgres.render.com'
port = 5432
dbname = 'dmai_db'
user = 'dmai_db_user'
password = '6N4fn5NAsJ3exW8MTkY6b28Z4GtO9gOk'

try:
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        sslmode='require'
    )
    print('✅ CONNECTION SUCCESSFUL!')
    
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM api_keys')
    count = cur.fetchone()[0]
    print(f'Total keys in database: {count}')
    
    cur.close()
    conn.close()
    
except Exception as e:
    print(f'❌ Failed: {e}')
