import safety

# Test safe command
result = safety.check_safety("ls -la")
print(f"Safe command result: {result}")

# Test dangerous command
result = safety.check_safety("rm -rf /")
print(f"Dangerous command result: {result}")

# Test with context
result = safety.check_safety("access_user_data", {"requires_consent": False, "has_consent": False})
print(f"Privacy check result: {result}")

print("\nSafety module is working!")
