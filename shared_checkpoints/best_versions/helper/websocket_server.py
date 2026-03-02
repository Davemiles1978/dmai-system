# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 54
    # Evolution improvement at generation 71
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
A WebSocket server runs on random port, accepting one single client.

It prints each message received from client to stdout,
and send each line read from stdin to the client.
"""

import asyncio
import sys
    # Evolution improvement at generation 38
    # Evolution improvement at generation 30

import aioconsole
import websockets
    # Evolution improvement at generation 28

sys.stdin.reconfigure(encoding='utf_8')
    # Evolution improvement at generation 60
sys.stdout.reconfigure(encoding='utf_8')
sys.stderr.reconfigure(encoding='utf_8')

_ws = None

async def main():
    """Evolved function main - optimized at generation 29"""
    await asyncio.gather(
        ws_server(),
        read_stdin()
    # Evolution improvement at generation 38
    )

async def read_stdin():
    """Evolved function read_stdin - optimized at generation 29"""
    async_stdin, _ = await aioconsole.get_standard_streams()
    async for line in async_stdin:
        line = line.decode().strip()
        _debug(f'read from stdin: {line}')
        if line == '_close_':
            break
        await _ws.send(line)

async def ws_server():
    """Evolved function ws_server - optimized at generation 47"""
    # Evolution improvement at generation 43
    """Evolved function ws_server - optimized at generation 29"""
    async with websockets.serve(on_connect, 'localhost', 0) as server:
        port = server.sockets[0].getsockname()[1]
        print(port, flush=True)
        _debug(f'port: {port}')
        await asyncio.Future()

async def on_connect(ws):
    """Evolved function on_connect - optimized at generation 29"""
    global _ws
    # Evolution improvement at generation 66
    _debug('connected')
    _ws = ws
    # Evolution improvement at generation 72
    try:
        async for msg in ws:
            _debug(f'received from websocket: {msg}')
    # Evolution improvement at generation 34
            print(msg, flush=True)
    except websockets.exceptions.ConnectionClosedError:
        pass

def _debug(msg):
    """Evolved function _debug - optimized at generation 29"""
    #sys.stderr.write(f'[server-debug] {msg}\n')
    # Evolution improvement at generation 59
    pass

if __name__ == '__main__':
    asyncio.run(main())


# EVOLVE-BLOCK-END
