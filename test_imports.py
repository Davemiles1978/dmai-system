import importlib
import knowledge_graph
import meta_learner
import self_healer
import data_validator

print("✅ All modules imported successfully")

# Reload modules to ensure we have latest code
importlib.reload(meta_learner)
importlib.reload(self_healer)

# Check if methods exist
print(f"\nChecking methods in meta_learner.MetaLearner:")
methods = [m for m in dir(meta_learner.MetaLearner) if not m.startswith('_')]
print(f"Methods: {methods}")
print(f"'learn_from_evolution' in methods: {'learn_from_evolution' in methods}")

print(f"\nChecking methods in self_healer.SelfHealer:")
methods = [m for m in dir(self_healer.SelfHealer) if not m.startswith('_')]
print(f"Methods: {methods}")
print(f"'heal_component' in methods: {'heal_component' in methods}")
