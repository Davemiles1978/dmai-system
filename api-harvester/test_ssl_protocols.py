#!/usr/bin/env python3
"""
Test different SSL protocol versions with Render PostgreSQL
"""
import psycopg2
import ssl
import socket
import sys

print('='*60)
print('SSL PROTOCOL VERSION TEST')
print('='*60)
print(f'Python version: {sys.version}')
print(f'SSL version: {ssl.OPENSSL_VERSION}')
print('='*60)

# Connection parameters
host = 'dpg-cvaj1qd2ng1s73c1dn60-a.oregon-postgres.render.com'
port = 5432
user = 'dmai_db_user'
password = '6N4fn5NAsJ3exW8MTkY6b28Z4GtO9gOk'
dbname = 'dmai_db'

print(f'\n📋 Target: {host}:{port}')
print(f'📋 Database: {dbname}')
print(f'📋 User: {user}')

# Method 1: Test raw SSL handshake with different protocols
print('\n' + '='*60)
print('METHOD 1: Raw SSL Handshake Test')
print('='*60)

# Test different SSL/TLS versions
protocol_versions = [
    ('SSLv2', getattr(ssl, 'PROTOCOL_SSLv2', None)),
    ('SSLv3', getattr(ssl, 'PROTOCOL_SSLv3', None)),
    ('TLSv1', getattr(ssl, 'PROTOCOL_TLSv1', None)),
    ('TLSv1.1', getattr(ssl, 'PROTOCOL_TLSv1_1', None)),
    ('TLSv1.2', getattr(ssl, 'PROTOCOL_TLSv1_2', None)),
    ('TLS', ssl.PROTOCOL_TLS)
]

for protocol_name, protocol in protocol_versions:
    if protocol is None:
        print(f'\n🔍 {protocol_name}: Not available in this Python version')
        continue
        
    print(f'\n🔍 Testing {protocol_name}...')
    try:
        # Create custom SSL context
        context = ssl.SSLContext(protocol)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        # Create socket and connect
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((host, port))
        print(f'  ✅ TCP connection established')
        
        # Wrap socket with SSL
        ssl_sock = context.wrap_socket(sock, server_hostname=host)
        print(f'  ✅ SSL handshake successful with {protocol_name}')
        print(f'  📋 Cipher: {ssl_sock.cipher()}')
        
        # Get certificate info
        cert = ssl_sock.getpeercert()
        if cert:
            print(f'  📋 Certificate issued to: {cert.get("subject", [])}')
        
        ssl_sock.close()
        
    except ssl.SSLError as e:
        print(f'  ❌ SSL Error: {e}')
    except socket.timeout:
        print(f'  ❌ Connection timeout')
    except Exception as e:
        print(f'  ❌ Error: {type(e).__name__}: {e}')

# Method 2: Test psycopg2 with different SSL parameters
print('\n' + '='*60)
print('METHOD 2: psycopg2 Connection Test')
print('='*60)

ssl_modes = ['require', 'prefer', 'allow', 'verify-ca', 'verify-full']
ssl_params = [
    {'sslmode': 'require'},
    {'sslmode': 'require', 'sslcompression': '0'},
    {'sslmode': 'require', 'target_session_attrs': 'read-write'},
    {'sslmode': 'require', 'connect_timeout': '30'},
]

for i, params in enumerate(ssl_params):
    print(f'\n🔍 Test {i+1} with params: {params}')
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            **params
        )
        print(f'  ✅ Connection successful!')
        
        # Get PostgreSQL version
        cur = conn.cursor()
        cur.execute('SELECT version()')
        version = cur.fetchone()[0]
        print(f'  📋 PostgreSQL: {version[:60]}...')
        cur.close()
        conn.close()
        break  # Stop if successful
        
    except Exception as e:
        print(f'  ❌ Failed: {type(e).__name__}: {e}')

# Method 3: Test with connection string
print('\n' + '='*60)
print('METHOD 3: Connection String Test')
print('='*60)

try:
    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
    conn = psycopg2.connect(conn_str)
    print('✅ Connection string successful!')
    conn.close()
except Exception as e:
    print(f'❌ Connection string failed: {e}')

# Method 4: Test with explicit SSL context (if supported)
print('\n' + '='*60)
print('METHOD 4: Custom SSL Context Test')
print('='*60)

try:
    # Try different SSL context creation methods
    for method in ['default', 'protocol']:
        print(f'\n🔍 Testing method: {method}')
        try:
            if method == 'default':
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            else:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            
            # Set cipher list
            context.set_ciphers('DEFAULT@SECLEVEL=1')
            
            conn = psycopg2.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password,
                sslmode='require',
                sslcontext=context
            )
            print(f'  ✅ Connection successful with {method} context!')
            conn.close()
            break
            
        except Exception as e:
            print(f'  ❌ {method} context failed: {e}')
            
except Exception as e:
    print(f'❌ All SSL context methods failed: {e}')

print('\n' + '='*60)
print('SSL PROTOCOL TEST COMPLETE')
print('='*60)
print('\n📝 If all tests fail, try connecting with openssl:')
print(f'openssl s_client -connect {host}:{port} -servername {host}')
