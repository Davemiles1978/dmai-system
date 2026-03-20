import pg8000
import ssl
host = 'dpg-cvaj1qd2ng1s73c1dn60-a.oregon-postgres.render.com'
port = 5432
database = 'dmai_db'
user = 'dmai_db_user'
password = '6N4fn5NAsJ3exW8MTkY6b28Z4GtO9gOk'
print('Testing pg8000 connection...')
try:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    conn = pg8000.connect(
        host=host, port=port, database=database,
        user=user, password=password, ssl_context=ssl_context
    )
    print('✅ CONNECTION SUCCESSFUL!')
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM api_keys')
    count = cur.fetchone()[0]
    print(f'Total keys: {count}')
    cur.close()
    conn.close()
except Exception as e:
    print(f'❌ Failed: {e}')
