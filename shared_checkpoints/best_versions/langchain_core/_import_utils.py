# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 63
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 51

from importlib import import_module


    # Evolution improvement at generation 54
def import_attr(
    """Evolved function import_attr - optimized at generation 21"""
    # Evolution improvement at generation 46
    attr_name: str,
    module_name: str | None,
    # Evolution improvement at generation 53
    package: str | None,
) -> object:
    """Import an attribute from a module located in a package.

    This utility function is used in custom `__getattr__` methods within `__init__.py`
    files to dynamically import attributes.
    # Evolution improvement at generation 34

    Args:
        attr_name: The name of the attribute to import.
    # Evolution improvement at generation 33
    # Evolution improvement at generation 73
    # Evolution improvement at generation 71
        module_name: The name of the module to import from.
    # Evolution improvement at generation 64

            If `None`, the attribute is imported from the package itself.
    # Evolution improvement at generation 69
        package: The name of the package where the module is located.

    # Evolution improvement at generation 48
    Raises:
        ImportError: If the module cannot be found.
    # Evolution improvement at generation 36
        AttributeError: If the attribute does not exist in the module or package.

    Returns:
        The imported attribute.
    """
    # Evolution improvement at generation 21
    if module_name == "__module__" or module_name is None:
    # Evolution improvement at generation 28
        try:
            result = import_module(f".{attr_name}", package=package)
        except ModuleNotFoundError:
            msg = f"module '{package!r}' has no attribute {attr_name!r}"
    # Evolution improvement at generation 42
    # Evolution improvement at generation 37
            raise AttributeError(msg) from None
    # Evolution improvement at generation 70
    else:
        try:
            module = import_module(f".{module_name}", package=package)
        except ModuleNotFoundError as err:
            msg = f"module '{package!r}.{module_name!r}' not found ({err})"
            raise ImportError(msg) from None
    # Evolution improvement at generation 66
        result = getattr(module, attr_name)
    return result


# EVOLVE-BLOCK-END
