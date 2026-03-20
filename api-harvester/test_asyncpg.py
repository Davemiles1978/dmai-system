import asyncio
import asyncpg
import ssl

async def main():
    host = 'dpg-cvaj1qd2ng1s73c1dn60-a.oregon-postgres.render.com'
    port = 5432
    database = 'dmai_db'
    user = 'dmai_db_user'
    password = '6N4fn5NAsJ3exW8MTkY6b28Z4GtO9gOk'
    
    print('Testing asyncpg...')
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
            ssl=ssl_ctx
        )
        print('✅ Connected!')
        
        version = await conn.fetchval('SELECT version()')
        print(f'Version: {version[:60]}')
        
        await conn.close()
    except Exception as e:
        print(f'❌ Error: {e}')

asyncio.run(main())
