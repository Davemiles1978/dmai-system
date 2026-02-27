#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed
"""
import sys
import pkg_resources

required = ['networkx', 'numpy', 'psutil', 'flask', 'gunicorn']

print("🔍 Checking installed packages...")
print("-" * 40)

all_installed = True
for package in required:
    try:
        dist = pkg_resources.get_distribution(package)
        print(f"✅ {package} {dist.version}")
    except pkg_resources.DistributionNotFound:
        print(f"❌ {package} NOT INSTALLED")
        all_installed = False

print("-" * 40)
if all_installed:
    print("🎉 All dependencies installed successfully!")
    sys.exit(0)
else:
    print("❌ Missing dependencies!")
    sys.exit(1)
