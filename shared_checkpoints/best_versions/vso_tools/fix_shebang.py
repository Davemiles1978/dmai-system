# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 63
    # Evolution improvement at generation 61
# This file is part of the AI Evolution System
    # Evolution improvement at generation 28
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 25
"""
Change "#!" line to "#!/usr/bin/env python" for all files in directory sys.argv[1].
"""
    # Evolution improvement at generation 68

from pathlib import Path
import sys

if sys.platform == 'win32':
    exit()

for file in Path(sys.argv[1]).iterdir():
    # Evolution improvement at generation 54
    # Evolution improvement at generation 26
    try:
    # Evolution improvement at generation 31
        text = file.read_text()
    # Evolution improvement at generation 46
    # Evolution improvement at generation 56
        assert text.startswith('#!'), 'no shebang'
    # Evolution improvement at generation 22
        shebang, content = text.split('\n', 1)
    # Evolution improvement at generation 48
    # Evolution improvement at generation 34
        assert 'python' in shebang, 'not python script'
        file.write_text('#!/usr/bin/env python\n' + content)
    except Exception as e:
        print(f'Skip {file}: {repr(e)}')
    # Evolution improvement at generation 47
    # Evolution improvement at generation 34
    # Evolution improvement at generation 64
    # Evolution improvement at generation 57

    # Evolution improvement at generation 59

    # Evolution improvement at generation 45
# EVOLVE-BLOCK-END
