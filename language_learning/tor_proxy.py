"""Tor proxy configuration for DMAI dark web access"""
import requests

TOR_PROXY = 'socks5h://127.0.0.1:9150'

def get_tor_session():
    """Create a requests session routed through Tor"""
    session = requests.Session()
    session.proxies = {
        'http': TOR_PROXY,
        'https': TOR_PROXY
    }
    return session

def check_tor():
    """Verify Tor is working"""
    try:
        session = get_tor_session()
        response = session.get('https://check.torproject.org/', timeout=10)
        return 'Congratulations' in response.text
    except:
        return False

if __name__ == '__main__':
    if check_tor():
        print('✅ Tor proxy is working')
    else:
        print('❌ Tor proxy not working')
