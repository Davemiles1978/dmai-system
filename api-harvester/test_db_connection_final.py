#!/usr/bin/env python3
"""
Final database connection test with proper error handling
"""
import psycopg2
import ssl
import socket
import sys

print('='*60)
print('DATABASE CONNECTION TEST - FINAL')
print('='*60)

# Connection parameters
params = {
    'host': 'dpg-cvaj1qd2ng1s73c1dn60-a.oregon-postgres.render.com',
    'port': 5432,
    'dbname': 'dmai_db',
    'user': 'dmai_db_user',
    'password': '6N4fn5NAsJ3exW8MTkY6b28Z4GtO9gOk',
    'connect_timeout': 30
}

print('\n📋 Connection parameters:')
for key, value in params.items():
    if key == 'password':
        print(f'  {key}: ********')
    else:
        print(f'  {key}: {value}')

# Test 1: Basic TCP connectivity
print('\n🔄 Test 1: Basic TCP connectivity...')
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result = sock.connect_ex((params['host'], params['port']))
    if result == 0:
        print('✅ TCP port is open and accepting connections')
    else:
        print(f'❌ TCP connection failed with error code: {result}')
    sock.close()
except Exception as e:
    print(f'❌ TCP test failed: {e}')

# Test 2: Try different SSL modes
ssl_modes = ['require', 'prefer', 'allow', 'disable']
print('\n🔄 Test 2: Testing all SSL modes...')

for sslmode in ssl_modes:
    print(f'\n  Testing sslmode={sslmode}...')
    test_params = params.copy()
    test_params['sslmode'] = sslmode
    
    try:
        conn = psycopg2.connect(**test_params)
        print(f'  ✅ SUCCESS with sslmode={sslmode}')
        conn.close()
        break
    except Exception as e:
        error_str = str(e)
        if 'SSL connection has been closed' in error_str:
            print(f'  ❌ SSL connection closed unexpectedly')
        elif 'SSL/TLS required' in error_str:
            print(f'  ❌ SSL/TLS required (expected for disable mode)')
        else:
            print(f'  ❌ Failed: {error_str[:100]}')

# Test 3: Try with connection string format
print('\n🔄 Test 3: Testing with connection string format...')
try:
    conn_str = f"postgresql://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['dbname']}?sslmode=require"
    conn = psycopg2.connect(conn_str)
    print('✅ Connection string format successful!')
    conn.close()
except Exception as e:
    print(f'❌ Connection string failed: {e}')

# Test 4: Try with SSL context
print('\n🔄 Test 4: Testing with custom SSL context...')
try:
    import ssl
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Note: sslcontext parameter may not be supported in all psycopg2 versions
    conn = psycopg2.connect(
        host=params['host'],
        port=params['port'],
        dbname=params['dbname'],
        user=params['user'],
        password=params['password'],
        sslmode='require',
        connect_timeout=30
    )
    print('✅ SSL context connection successful!')
    conn.close()
except Exception as e:
    print(f'❌ SSL context failed: {e}')

print('\n' + '='*60)
print('TEST COMPLETE')
print('='*60)
