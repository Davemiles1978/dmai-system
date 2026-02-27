#!/usr/bin/env python3
import os
from pathlib import Path

desktop = Path.home() / "Desktop"
install_dir = Path.cwd()
icon_path = desktop / "AI-Evolution-System.command"

with open(icon_path, "w") as f:
    f.write("""#!/bin/bash
cd {}
./launch.sh
""".format(install_dir))

os.chmod(icon_path, 0o755)
print("✅ Desktop icon created! Look on your desktop.")
