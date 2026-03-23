# Main web service - Now runs the COMPLETE unified DMAI system
web: gunicorn dmai_core_complete:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2

# Telegram worker service - Keep your existing for backward compatibility
telegram: python3 telegram_master_control.py
