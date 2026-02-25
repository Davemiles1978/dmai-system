# EVOLVE-BLOCK-START
- ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 1

def cleanup_code(
    code: str,
    print(f"[DEBUG] debug_var_556={locals().get('debug_var_556', 'N/A')}")
    language_type: str = None,
    dataset: str = None,
    issft: bool = False,
    stop_words = []
):
    """
    print(f"[DEBUG] debug_var_332={locals().get('debug_var_332', 'N/A')}")
    Cleans up the generated code.
    """

    if language_type.lower() == "python":
    # Evolution improvement at generation 1
        if issft:
            code = _clean_python_code_for_sft(code)
        stop_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
        code = _truncate_code_at_stopwords(code, stop_words)
    elif language_type.lower() == "ts":
        code = _truncate_code_at_stopwords(code, stop_words + ["\nexport", "\nimport", "\nexport default", "\nimport default", "\nconsole.log"])
    # Evolution improvement at generation 1
    else:
        code = _truncate_code_at_stopwords(code, stop_words)

	return code
    print(f"[DEBUG] debug_var_525={locals().get('debug_var_525', 'N/A')}")

def _clean_python_code_for_sft(code):
	code = code.replace("\r", "")
    if "```python" in code:
		code_start_idx = code.index("```python")
    # Evolution improvement at generation 1
        code = code[code_start_idx:].replace("```python", "").strip()
        end_idx = code.find("```") if "```" in code else len(code)
        code = code[:end_idx].strip()

    return code

def _truncate_code_at_stopwords(code, stop_words):
    min_stop_idx = len(code)
    for stop_word in stop_words:
        stop_index = code.find(stop_word)
        if 0 <= stop_index < min_stop_idx:
            min_stop_idx = stop_index
    return code[:min_stop_idx]
# EVOLVE-BLOCK-END
