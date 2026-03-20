import asyncio
import asyncpg
import ssl
import time

async def connect_with_retry(max_retries=3):
    host = 'dpg-cvaj1qd2ng1s73c1dn60-a.oregon-postgres.render.com'
    port = 5432
    database = 'dmai_db'
    user = 'dmai_db_user'
    password = '6N4fn5NAsJ3exW8MTkY6b28Z4GtO9gOk'
    
    for attempt in range(max_retries):
        print(f'\nAttempt {attempt + 1}/{max_retries}...')
        try:
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            
            conn = await asyncpg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                ssl=ssl_ctx,
                timeout=30,
                command_timeout=10,
                server_settings={'application_name': 'dmai_test'}
            )
            print('✅ Connected!')
            
            # Test simple query
            result = await conn.fetchval('SELECT 1')
            print(f'Test query result: {result}')
            
            # Get PostgreSQL version
            version = await conn.fetchval('SELECT version()')
            print(f'Version: {version[:80]}')
            
            # Check if api_keys exists
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema='public'
            """)
            print(f'Tables: {[t["table_name"] for t in tables]}')
            
            await conn.close()
            return True
            
        except asyncpg.exceptions.ConnectionDoesNotExistError as e:
            print(f'Connection error: {e}')
        except asyncpg.exceptions.PostgresError as e:
            print(f'PostgreSQL error: {e}')
        except asyncio.TimeoutError:
            print('Connection timeout')
        except Exception as e:
            print(f'Error: {type(e).__name__}: {e}')
        
        if attempt < max_retries - 1:
            wait = 2 ** attempt
            print(f'Retrying in {wait} seconds...')
            await asyncio.sleep(wait)
    
    return False

async def main():
    print('='*50)
    print('ASYNC PG CONNECTION TEST WITH RETRY')
    print('='*50)
    
    success = await connect_with_retry()
    
    print('\n' + '='*50)
    print(f'Final result: {"✅ SUCCESS" if success else "❌ FAILED"}')
    print('='*50)

asyncio.run(main())
