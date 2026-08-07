#!/usr/bin/env python3
"""
Patch AIIntegrationHub to use dynamic provider list from activator.
Inserts helper methods and replaces query_all_tutors.
"""

import re
import sys
from pathlib import Path

FILE_PATH = Path("components/phase11/AIIntegrationHub.py")

# --------------------------------------------------------------------
# 1. Helper methods to insert after _load_api_keys (after line 132)
# --------------------------------------------------------------------
HELPER_METHODS = """
    def _get_active_providers_with_methods(self):
        \"\"\"Return list of (provider_name, query_method) for active, known providers.\"\"\"
        activator = None
        try:
            import sys as _sys
            core = _sys.modules.get("dmai_core_complete")
            if core and hasattr(core, "components"):
                activator = core.components.get("api_activator")
        except Exception:
            pass
        if activator is None:
            logger.warning("Activator not found – using static provider list")
            return []

        active_ids = activator.get_active_providers()
        method_map = {
            'groq':              ('Groq', self._query_groq),
            'google_ai_studio':  ('Google AI Studio', self._query_google_ai_studio),
            'cerebras':          ('Cerebras', self._query_cerebras),
            'openai':            ('OpenAI', self._query_openai),
            'anthropic':         ('Anthropic', self._query_anthropic),
            'deepseek':          ('DeepSeek', self._query_deepseek),
            'github_models':     ('GitHub Models', self._query_github_models),
            'mistral':           ('Mistral', self._query_mistral),
        }
        priority = ['groq', 'google_ai_studio', 'cerebras', 'openai', 'anthropic', 'deepseek', 'github_models', 'mistral']
        result = []
        for pid in priority:
            if pid in active_ids and pid in method_map:
                result.append(method_map[pid])
        for pid in active_ids:
            if pid in method_map and pid not in priority:
                result.append(method_map[pid])
        if not result:
            logger.warning("No active known providers – falling back to static list")
        return result

    def _trigger_harvester_if_needed(self):
        \"\"\"If no active providers, trigger the free API harvester.\"\"\"
        try:
            import sys as _sys
            core = _sys.modules.get("dmai_core_complete")
            if core and hasattr(core, "components"):
                harvester = core.components.get("free_api_harvester")
                if harvester and hasattr(harvester, "harvest"):
                    logger.info("No active providers – triggering harvester")
                    harvester.harvest()
        except Exception as e:
            logger.warning(f"Harvester trigger failed: {e}")
"""

# --------------------------------------------------------------------
# 2. The new query_all_tutors method (replaces the old one)
# --------------------------------------------------------------------
NEW_QUERY_ALL_TUTORS = """
    def query_all_tutors(self, prompt: str, use_cache: bool = True) -> Dict:
        \"\"\"Query all available tutors using dynamic active provider list.\"\"\"
        start_time = time.time()

        cache_key = hash(prompt)
        if use_cache and cache_key in self.learning_cache:
            cache_time = self.learning_cache[cache_key]['timestamp']
            if (datetime.now() - cache_time).seconds < 300:
                logger.info(f"Using cached response for: {prompt[:50]}...")
                return self.learning_cache[cache_key]['response']

        results = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'responses': {},
            'errors': [],
            'synthesis': None
        }

        # Get dynamic provider list
        query_methods = self._get_active_providers_with_methods()
        if not query_methods:
            # Fallback to static list (keep original as last resort)
            query_methods = [
                ('OpenAI GPT-4', self._query_openai),
                ('DeepSeek', self._query_deepseek),
                ('Google Gemini', self._query_gemini),
                ('Anthropic Claude', self._query_anthropic),
                ('Perplexity AI', self._query_perplexity),
                ('xAI Grok', self._query_grok),
                ('Cerebras Inference', self._query_cerebras),
                ('GitHub Models', self._query_github_models),
                ('Mistral AI', self._query_mistral),
            ]
            self._trigger_harvester_if_needed()

        for tutor_name, method in query_methods:
            try:
                logger.debug(f"Querying {tutor_name}...")
                result = method(prompt)

                if result.get('success'):
                    results['responses'][tutor_name] = result['response']
                    self.performance_metrics['successful_queries'] += 1

                    if tutor_name not in self.performance_metrics['tutor_performance']:
                        self.performance_metrics['tutor_performance'][tutor_name] = {
                            'successes': 0,
                            'failures': 0,
                            'avg_response_time': 0
                        }
                    self.performance_metrics['tutor_performance'][tutor_name]['successes'] += 1

                    if self.tutor_manager:
                        quality = self._estimate_response_quality(result['response'])
                        dma_quality = self._estimate_dma_quality(prompt)
                        self.tutor_manager.record_comparison(tutor_name, dma_quality, quality)

                else:
                    results['errors'].append(f"{tutor_name}: {result.get('error', 'Unknown error')}")
                    self.performance_metrics['failed_queries'] += 1

                    if self.tutor_manager and tutor_name in self.performance_metrics['tutor_performance']:
                        self.performance_metrics['tutor_performance'][tutor_name]['failures'] += 1

            except Exception as e:
                logger.error(f"Error querying {tutor_name}: {e}")
                results['errors'].append(f"{tutor_name}: {str(e)}")
                self.performance_metrics['failed_queries'] += 1

        # Synthesize if we have a synthesizer
        if self.capability_synthesizer and results['responses']:
            try:
                results['synthesis'] = self.capability_synthesizer.synthesize(
                    results['responses'],
                    prompt
                )
                if results['synthesis'].get('unified_answer'):
                    results['unified_answer'] = results['synthesis']['unified_answer']
                self._learn_from_responses(results['synthesis'], prompt)
            except Exception as e:
                logger.error(f"Synthesis error: {e}")
                results['synthesis_error'] = str(e)

        response_time = time.time() - start_time
        self.performance_metrics['total_queries'] += 1
        self.performance_metrics['average_response_time'] = (
            (self.performance_metrics['average_response_time'] * (self.performance_metrics['total_queries'] - 1) + response_time) /
            self.performance_metrics['total_queries']
        )

        # Update query history
        self.query_history.append({
            'prompt': prompt,
            'timestamp': datetime.now().isoformat(),
            'response_count': len(results['responses']),
            'error_count': len(results['errors']),
            'response_time': response_time
        })
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]

        # Cache the result
        if use_cache and results['responses']:
            self.learning_cache[cache_key] = {
                'response': results,
                'timestamp': datetime.now()
            }

        return results
"""

def main():
    if not FILE_PATH.exists():
        print(f"ERROR: {FILE_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    with open(FILE_PATH, "r") as f:
        content = f.read()

    # ---- Insert helper methods after line 132 ----
    # Find the line with 'def _load_api_keys' and locate its end (the next def at same indent)
    # We'll insert after the 'return keys' line, which is likely near line 132.
    # But to be safe, we'll find the exact position of the next method definition after _load_api_keys.
    lines = content.splitlines(True)  # keep line endings
    insert_index = None

    # Find the line number where we want to insert: after the last line of _load_api_keys.
    # We'll search for "def _load_api_keys" and then find the next line that starts with "    def " (at indent level 4).
    in_load_keys = False
    for i, line in enumerate(lines):
        if re.match(r'^    def _load_api_keys\(', line):
            in_load_keys = True
        elif in_load_keys and re.match(r'^    def ', line):
            # This is the next method; insert before it (i.e., at the current index)
            insert_index = i
            break
    if insert_index is None:
        print("Could not find the end of _load_api_keys; insertion failed.", file=sys.stderr)
        sys.exit(1)

    # Insert helper methods before the next method (at insert_index)
    lines.insert(insert_index, HELPER_METHODS)

    # ---- Replace query_all_tutors ----
    # Find the method and replace its entire body.
    # We'll locate the line with "def query_all_tutors" and then find the next method (or class end).
    # We'll use a similar approach: find start and end indices.
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^    def query_all_tutors\(', line):
            start_idx = i
            break
    if start_idx is None:
        print("Could not find query_all_tutors method", file=sys.stderr)
        sys.exit(1)

    # Find the next method definition at same indent level (4 spaces) after start_idx
    for i in range(start_idx + 1, len(lines)):
        if re.match(r'^    def ', lines[i]):
            end_idx = i
            break
    if end_idx is None:
        # If no next method, assume it's the last method; end at the last line
        end_idx = len(lines)

    # Replace the method body: keep the first line (def ...) and replace the rest with NEW_QUERY_ALL_TUTORS
    # But we need to keep the method signature line and replace from after the colon.
    # Simpler: replace the entire block from start_idx to end_idx-1 with the new method.
    # We'll preserve indentation by ensuring the new method is indented with 4 spaces.
    # However, the new method already has indentation of 4 spaces (as defined in the string).
    # We'll slice and replace.
    new_method_lines = NEW_QUERY_ALL_TUTORS.splitlines(True)
    # Ensure the first line of new_method_lines is the method definition (we already have it).
    # We'll replace the block.
    lines[start_idx:end_idx] = new_method_lines

    # Write the modified content
    with open(FILE_PATH, "w") as f:
        f.writelines(lines)

    print("✅ Patch applied successfully to", FILE_PATH)

if __name__ == "__main__":
    main()
