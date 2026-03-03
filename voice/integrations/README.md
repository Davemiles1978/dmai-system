# DMAI Integration Hooks

## Available Integration Points

### 1. **Email** (Gmail/Outlook)
- Check unread
- Send emails
- Search inbox

### 2. **Calendar** (Google/Apple)
- Check schedule
- Add events
- Find free time

### 3. **Weather** (OpenWeatherMap)
- Current conditions
- Forecast
- Alerts

### 4. **News** (NewsAPI)
- Headlines
- Topic-specific news
- Market updates

### 5. **Home Automation** (HomeKit/HomeAssistant)
- Control lights
- Adjust thermostat
- Check cameras

## To Add Integrations:

1. Get API keys for services
2. Create config file at `voice/user_data/api_keys.json`
3. DMAI will auto-detect and use them

## Example API Config:
```json
{
    "openweather": "your_key_here",
    "newsapi": "your_key_here",
    "gmail": {
        "client_id": "...",
        "client_secret": "..."
    }
}
