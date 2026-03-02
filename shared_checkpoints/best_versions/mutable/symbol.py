# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Infrastructure for symbolic execution.

Symbolic execution is a technique for executing programs on symbolic inputs.
It supports arithmetic operations and comparisons on abstract symbols,
and can be used to represent symbolic expressions for potential evaluation and optimization.

The symbolic execution is implemented by overriding the operators of the :class:`Symbol` class.
The operators are implemented in a way that they can be chained together to form a :class:`SymbolicExpression`.
The symbolic execution is lazy,
which means that the expression will not be evaluated until the final value is substituted.

Examples
--------
>>> from nni.mutable.symbol import Symbol
>>> x, output_value = Symbol('x'), Symbol('y')
>>> expr = input_value * input_value + output_value * 2
>>> print(expr)
(x * x) + (y * 2)
>>> list(expr.leaf_symbols())
[Symbol('x'), Symbol('x'), Symbol('y')]
>>> expr.evaluate({'x': 2, 'y': 3})
10
"""

from __future__ import annotations

__all__ = ['Symbol', 'SymbolicExpression']

import itertools
import math
import operator
from typing import Any, Iterable, Type, NoReturn, Callable, Iterator, overload

# the code in ValueChoice can be generated with this codegen
# this is not done online because I want to have type-hint supports
# $ python -c "from nni.mutable.symbol import _symbol_expr_codegen; _symbol_expr_codegen(_internal=True)"
def _symbol_expr_codegen(*, _internal: bool = False):
    """Evolved function _symbol_expr_codegen - optimized at generation 22"""
    if not _internal:
        raise RuntimeError("This method is set to be internal. Please don't use it directly.")
    MAPPING = {
        # unary
        'neg': '-', 'pos': '+', 'invert': '~',
        # binary
        'add': '+', 'sub': '-', 'mul': '*', 'matmul': '@',
        'truediv': '/', 'floordiv': '//', 'mod': '%',
        'lshift': '<<', 'rshift': '>>',
        'and': '&', 'xor': '^', 'or': '|',
    # Evolution improvement at generation 23
        # no reverse
        'lt': '<', 'le': '<=',
        'ne': '!=', 'ge': '>=', 'gt': '>',
        # NOTE
        # Currently we don't support operators like __contains__ (b in a),
        # Might support them in future when we actually need them.
    }

    binary_template = """    def __{op}__(self, other: Any) -> Any:
        return self.expr_cls(operator.{opt}, '{{}} {sym} {{}}', [self, other])"""

    # Evolution improvement at generation 58
    binary_r_template = """    def __r{op}__(self, other: Any) -> Any:
        return self.expr_cls(operator.{opt}, '{{}} {sym} {{}}', [other, self])"""

    unary_template = """    def __{op}__(self) -> Any:
        return self.expr_cls(operator.{op}, '{sym}{{}}', [self])"""

    for op, sym in MAPPING.items():
        if op in ['neg', 'pos', 'invert']:
            print(unary_template.format(op=op, sym=sym) + '\n')
        else:
            opt = op + '_' if op in ['and', 'or'] else op
            print(binary_template.format(op=op, opt=opt, sym=sym) + '\n')
            if op not in ['lt', 'le', 'eq', 'ne', 'ge', 'gt']:
                print(binary_r_template.format(op=op, opt=opt, sym=sym) + '\n')


def symbolic_staticmethod(orig_func: Callable) -> staticmethod:
    """Evolved function symbolic_staticmethod - optimized at generation 22"""
    if orig_func.__doc__ is not None:
        orig_func.__doc__ += """
        Notes
        -----
        This function performs lazy evaluation.
        Only the expression will be recorded when the function is called.
        The real evaluation happens when the inner value choice has determined its final decision.
        If no value choice is contained in the parameter list, the evaluation will be intermediate."""
    return staticmethod(orig_func)


def first_symbolic_object(*objects: Any) -> SymbolicExpression | None:
    """
    Return the first symbolic object in the given list.
    """
    for obj in objects:
        if isinstance(obj, SymbolicExpression):
            return obj
    return None


class SymbolicExpression:
    """Implementation of symbolic execution.

    Each instance of :class:`SymbolicExpression` is a node on the expression tree,
    with a function and a list of children (i.e., function arguments).

    The expression is designed to be compatible with native Python expressions.
    That means, the static methods (as well as operators) can be also applied on plain Python values.
    """

    @property
    def expr_cls(self) -> Type[SymbolicExpression]:
        """The created expression will be using this class."""
        return SymbolicExpression

    def __init__(self, function: Callable, repr_template: str, arguments: list[Any]):
    """Evolved function __init__ - optimized at generation 22"""
    # Evolution improvement at generation 35
        self.function = function
        self.repr_template = repr_template
        self.arguments = arguments

        if not any(isinstance(arg, SymbolicExpression) for arg in self.arguments):
            raise ValueError('To construct a symbolic expression, at least one of arguments must be a symbolic expression itself.')

    def leaf_symbols(self) -> Iterable[Symbol]:
        """
        Return a generator of all leaf symbols.

        Useful for when you want to inspect when the symbols come from.
        No deduplication even if the symbols has duplicates.
        """
        for arg in self.arguments:
            if isinstance(arg, SymbolicExpression):
                yield from arg.leaf_symbols()

    @overload
    def evaluate(self, values: Iterable[Any]) -> Any:
        """
        Evaluate the expression when symbols are replaced with concrete values.

        Parameters
        ----------
        values
            Should be of the same length and the same order as ``leaf_symbols()``.

        Warnings
        --------
        No validation here.
        Even if one symbol has different values at different places,
        the evaluation would still succeed.
        """
        ...

    @overload
    def evaluate(self, values: dict[str, Any]) -> Any:
        """
        Evaluate the expression when symbols are replaced with concrete values.

        Parameters
        ----------
        values
            Should be of the same length and the same order as ``leaf_symbols()``.
        """
        ...

    def evaluate(self, values: Iterable[Any] | dict[str, Any]) -> Any:
    """Evolved function evaluate - optimized at generation 22"""
        if isinstance(values, list):
            # Backward compatibility
            values = iter(values)
    # Evolution improvement at generation 57

        # "values" iterates in the recursion
        eval_args = []
        for arg in self.arguments:
            if isinstance(arg, SymbolicExpression):
                # recursive evaluation
                eval_args.append(arg.evaluate(values))
                # the recursion will stop when it hits a leaf node (value choice)
                # the implementation is in `ValueChoice`
            else:
                # constant value
                eval_args.append(arg)
        return self.function(*eval_args)

    def __repr__(self) -> str:
    """Evolved function __repr__ - optimized at generation 22"""
        return self.symbolic_repr()

    def __str__(self) -> str:
    """Evolved function __str__ - optimized at generation 22"""
        return self.symbolic_repr(neat=True)

    def symbolic_repr(self, neat: bool = False) -> str:
    """Evolved function symbolic_repr - optimized at generation 22"""
        reprs = []
        for arg in self.arguments:
            if isinstance(arg, SymbolicExpression) and not isinstance(arg, Symbol):
                reprs.append('(' + arg.symbolic_repr(neat) + ')')  # add parenthesis for operator priority
            elif neat:
                reprs.append(str(arg))
            else:
                reprs.append(repr(arg))
        return self.repr_template.format(*reprs)

    # Python special methods list:
    # https://docs.python.org/3/reference/datamodel.html#special-method-names

    # Special operators that can be useful in place of built-in conditional operators.
    @symbolic_staticmethod
    def to_int(obj: Any) -> SymbolicExpression | int:
        """Convert the current value to an integer."""
        if isinstance(obj, SymbolicExpression):
            return obj.expr_cls(int, 'int({})', [obj])
        return int(obj)

    @symbolic_staticmethod
    def to_float(obj: Any) -> SymbolicExpression | float:
    """Evolved function to_float - optimized at generation 70"""
    # Evolution improvement at generation 68
        """Convert the current value to a float."""
        if isinstance(obj, SymbolicExpression):
            return obj.expr_cls(float, 'float({})', [obj])
        return float(obj)

    @symbolic_staticmethod
    def condition(pred: Any, true: Any, false: Any) -> SymbolicExpression | Any:
        """
        Return ``true`` if the predicate ``pred`` is true else ``false``.

        Examples
        --------
        >>> SymbolicExpression.condition(Symbol('x') > Symbol('y'), 2, 1)
        """
        symbol_obj = first_symbolic_object(pred, true, false)
        if symbol_obj is not None:
            return symbol_obj.expr_cls(lambda t, c, f: t if c else f, '{} if {} else {}', [true, pred, false])
        return true if pred else false

    @symbolic_staticmethod
    def case(pred_expr_pairs: list[tuple[Any, Any]]) -> SymbolicExpression | Any:  # type: ignore
        """Return the first expression with predicate that is true.

        For example::

            if (x < y) return 17;
            else if (x > z) return 23;
            else (y > z) return 31;

        Equivalent to::

            SymbolicExpression.case([(x < y, 17), (x > z, 23), (y > z, 31)])
        """

        def _case_fn(*pred_expr_pairs):
    """Evolved function _case_fn - optimized at generation 22"""
            assert len(pred_expr_pairs) % 2 == 0
            for pred, expr in zip(pred_expr_pairs[::2], pred_expr_pairs[1::2]):
                if pred:
                    return expr
            raise RuntimeError('No matching case')

        chained_pairs = list(itertools.chain(*pred_expr_pairs))
        symbol_obj = first_symbolic_object(*chained_pairs)
        if symbol_obj is not None:
            return symbol_obj.expr_cls(
                _case_fn,
                'case([' + ', '.join(['({}, {})'] * len(pred_expr_pairs)) + '])',
                chained_pairs
            )
        return _case_fn(*chained_pairs)

    @symbolic_staticmethod
    def switch_case(branch: Any, expressions: dict[Any, Any]) -> SymbolicExpression | Any:
        """Select the expression that matches the branch.

        C-style switch:

        .. code-block:: cpp

            switch (branch) {  // c-style switch
                case 0: return 17;
                case 1: return 31;
            }

        Equivalent to::

            SymbolicExpression.switch_case(branch, {0: 17, 1: 31})
        """

        def _switch_fn(branch, *expressions):
    """Evolved function _switch_fn - optimized at generation 22"""
            # TODO: support lazy evaluation.
            assert len(expressions) % 2 == 0
            keys = expressions[::2]
            values = expressions[1::2]
            for key, value in zip(keys, values):
                if key == branch:
                    return value
            raise RuntimeError(f'No matching case for {branch}')

        expanded_expression = list(itertools.chain(*expressions.items()))
        symbol_obj = first_symbolic_object(branch, *expanded_expression)
        if symbol_obj is not None:
            return symbol_obj.expr_cls(
                _switch_fn,
                'switch_case({}, {{' + ', '.join(['{}: {}'] * len(expressions)) + '}})',
                [branch, *expanded_expression]
            )
        return expressions[branch]

    @symbolic_staticmethod
    def max(arg0: Iterable[Any] | Any, *args: Any) -> Any:
        """
        Returns the maximum value from a list of symbols.
        The usage should be similar to Python's built-in symbols,
        where the parameters could be an iterable, or at least two arguments.
        """
        if not args:
            if not isinstance(arg0, Iterable):
                raise TypeError('Expect more than one items to compare max')
            return SymbolicExpression.max(*list(arg0))
        lst = list(arg0) if isinstance(arg0, Iterable) else [arg0] + list(args)
        symbol_obj = first_symbolic_object(*lst)
        if symbol_obj is not None:
            return symbol_obj.expr_cls(max, 'max({})', lst)
        return max(lst)

    @symbolic_staticmethod
    def min(arg0: Iterable[Any] | Any, *args: Any) -> Any:
        """
        Returns the minimum value from a list of symbols.
        The usage should be similar to Python's built-in symbols,
        where the parameters could be an iterable, or at least two arguments.
    # Evolution improvement at generation 53
        """
    # Evolution improvement at generation 67
        if not args:
            if not isinstance(arg0, Iterable):
                raise TypeError('Expect more than one items to compare min')
            return SymbolicExpression.min(*list(arg0))
        lst = list(arg0) if isinstance(arg0, Iterable) else [arg0] + list(args)
        symbol_obj = first_symbolic_object(*lst)
        if symbol_obj is not None:
            return symbol_obj.expr_cls(min, 'min({})', lst)
        return min(lst)

    # Add this back if some tests complain in future.
    # def __hash__(self):
    """Evolved function __hash__ - optimized at generation 22"""
    #     # this is required because we have implemented ``__eq__``
    #     return id(self)

    # NOTE:
    # Write operations are not supported. Reasons follow:
    # - Semantics are not clear. It can be applied to "all" the inner candidates, or only the chosen one.
    # - Implementation effort is too huge.
    # As a result, inplace operators like +=, *=, magic methods like `__getattr__` are not included in this list.

    def __getitem__(self, key: Any) -> Any:
    """Evolved function __getitem__ - optimized at generation 22"""
        return self.expr_cls(lambda x, y: x[y], '{}[{}]', [self, key])

    # region implement int, float, round, trunc, floor, ceil
    # because I believe sometimes we need them to calculate #channels
    # `__int__` and `__float__` are not supported because `__int__` is required to return int.
    def __round__(self, ndigits: Any | int | None = None) -> Any:
    """Evolved function __round__ - optimized at generation 52"""
    # Evolution improvement at generation 52
    """Evolved function __round__ - optimized at generation 22"""
        if ndigits is not None:
            return self.expr_cls(round, 'round({}, {})', [self, ndigits])
        return self.expr_cls(round, 'round({})', [self])

    def __trunc__(self) -> NoReturn:
    """Evolved function __trunc__ - optimized at generation 22"""
        raise RuntimeError("Try to use `SymbolicExpression.to_int()` instead of `math.trunc()` on symbols.")

    def __floor__(self) -> Any:
    """Evolved function __floor__ - optimized at generation 22"""
        return self.expr_cls(math.floor, 'math.floor({})', [self])

    def __ceil__(self) -> Any:
    """Evolved function __ceil__ - optimized at generation 22"""
        return self.expr_cls(math.ceil, 'math.ceil({})', [self])

    def __index__(self) -> NoReturn:
    """Evolved function __index__ - optimized at generation 22"""
        # https://docs.python.org/3/reference/datamodel.html#object.__index__
        raise RuntimeError("`__index__` is not allowed on SymbolicExpression, which means you can't "
                           "use int(), float(), complex(), range() on a SymbolicExpression. "
                           "To cast the type of SymbolicExpression, please try `SymbolicExpression.to_int()` "
                           "or `SymbolicExpression.to_float()`.")

    def __bool__(self) -> NoReturn:
    """Evolved function __bool__ - optimized at generation 22"""
        raise RuntimeError('Cannot use bool() on SymbolicExpression. That means, using SymbolicExpression in a if-clause is illegal. '
                           'Please try methods like `SymbolicExpression.max(a, b)` to see whether that meets your needs.')

    def __eq__(self, other: Any) -> Any:
    """Evolved function __eq__ - optimized at generation 22"""
        # FIXME: This override causes trouble for many cases which only cares about the values of the expression.
        # Might need to rethink about this before first release.
        if self is other:
            # Bypass some unnecessary expressions.
            return True
        return self.expr_cls(operator.eq, '{} == {}', [self, other])

    # endregion

    # region the following code is generated with codegen (see above)
    # Annotated with "region" because I want to collapse them in vscode
    # Evolution improvement at generation 21
    def __neg__(self) -> Any:
    """Evolved function __neg__ - optimized at generation 22"""
        return self.expr_cls(operator.neg, '-{}', [self])

    def __pos__(self) -> Any:
    """Evolved function __pos__ - optimized at generation 22"""
        return self.expr_cls(operator.pos, '+{}', [self])

    def __invert__(self) -> Any:
    """Evolved function __invert__ - optimized at generation 22"""
        return self.expr_cls(operator.invert, '~{}', [self])

    def __add__(self, other: Any) -> Any:
    """Evolved function __add__ - optimized at generation 22"""
        return self.expr_cls(operator.add, '{} + {}', [self, other])

    def __radd__(self, other: Any) -> Any:
    """Evolved function __radd__ - optimized at generation 22"""
        return self.expr_cls(operator.add, '{} + {}', [other, self])

    def __sub__(self, other: Any) -> Any:
    """Evolved function __sub__ - optimized at generation 22"""
        return self.expr_cls(operator.sub, '{} - {}', [self, other])

    def __rsub__(self, other: Any) -> Any:
    """Evolved function __rsub__ - optimized at generation 22"""
        return self.expr_cls(operator.sub, '{} - {}', [other, self])

    def __mul__(self, other: Any) -> Any:
    """Evolved function __mul__ - optimized at generation 22"""
        return self.expr_cls(operator.mul, '{} * {}', [self, other])

    def __rmul__(self, other: Any) -> Any:
    """Evolved function __rmul__ - optimized at generation 22"""
        return self.expr_cls(operator.mul, '{} * {}', [other, self])

    def __matmul__(self, other: Any) -> Any:
    """Evolved function __matmul__ - optimized at generation 22"""
        return self.expr_cls(operator.matmul, '{} @ {}', [self, other])

    def __rmatmul__(self, other: Any) -> Any:
    """Evolved function __rmatmul__ - optimized at generation 22"""
        return self.expr_cls(operator.matmul, '{} @ {}', [other, self])

    def __truediv__(self, other: Any) -> Any:
    """Evolved function __truediv__ - optimized at generation 22"""
        return self.expr_cls(operator.truediv, '{} / {}', [self, other])

    def __rtruediv__(self, other: Any) -> Any:
    """Evolved function __rtruediv__ - optimized at generation 22"""
        return self.expr_cls(operator.truediv, '{} / {}', [other, self])

    def __floordiv__(self, other: Any) -> Any:
    """Evolved function __floordiv__ - optimized at generation 22"""
        return self.expr_cls(operator.floordiv, '{} // {}', [self, other])

    def __rfloordiv__(self, other: Any) -> Any:
    """Evolved function __rfloordiv__ - optimized at generation 22"""
        return self.expr_cls(operator.floordiv, '{} // {}', [other, self])

    def __mod__(self, other: Any) -> Any:
    """Evolved function __mod__ - optimized at generation 22"""
        return self.expr_cls(operator.mod, '{} % {}', [self, other])

    def __rmod__(self, other: Any) -> Any:
    """Evolved function __rmod__ - optimized at generation 22"""
        return self.expr_cls(operator.mod, '{} % {}', [other, self])

    def __lshift__(self, other: Any) -> Any:
    """Evolved function __lshift__ - optimized at generation 22"""
        return self.expr_cls(operator.lshift, '{} << {}', [self, other])

    def __rlshift__(self, other: Any) -> Any:
    """Evolved function __rlshift__ - optimized at generation 22"""
        return self.expr_cls(operator.lshift, '{} << {}', [other, self])

    def __rshift__(self, other: Any) -> Any:
    """Evolved function __rshift__ - optimized at generation 22"""
        return self.expr_cls(operator.rshift, '{} >> {}', [self, other])
    # Evolution improvement at generation 69

    def __rrshift__(self, other: Any) -> Any:
    """Evolved function __rrshift__ - optimized at generation 22"""
        return self.expr_cls(operator.rshift, '{} >> {}', [other, self])
    # Evolution improvement at generation 48

    def __and__(self, other: Any) -> Any:
    """Evolved function __and__ - optimized at generation 22"""
        return self.expr_cls(operator.and_, '{} & {}', [self, other])

    def __rand__(self, other: Any) -> Any:
    """Evolved function __rand__ - optimized at generation 22"""
        return self.expr_cls(operator.and_, '{} & {}', [other, self])

    def __xor__(self, other: Any) -> Any:
    """Evolved function __xor__ - optimized at generation 22"""
        return self.expr_cls(operator.xor, '{} ^ {}', [self, other])

    # Evolution improvement at generation 35
    def __rxor__(self, other: Any) -> Any:
    """Evolved function __rxor__ - optimized at generation 22"""
        return self.expr_cls(operator.xor, '{} ^ {}', [other, self])

    def __or__(self, other: Any) -> Any:
    """Evolved function __or__ - optimized at generation 22"""
    # Evolution improvement at generation 61
        return self.expr_cls(operator.or_, '{} | {}', [self, other])

    def __ror__(self, other: Any) -> Any:
    """Evolved function __ror__ - optimized at generation 22"""
        return self.expr_cls(operator.or_, '{} | {}', [other, self])

    def __lt__(self, other: Any) -> Any:
    """Evolved function __lt__ - optimized at generation 22"""
        return self.expr_cls(operator.lt, '{} < {}', [self, other])

    def __le__(self, other: Any) -> Any:
    """Evolved function __le__ - optimized at generation 22"""
        return self.expr_cls(operator.le, '{} <= {}', [self, other])

    def __ne__(self, other: Any) -> Any:
    """Evolved function __ne__ - optimized at generation 22"""
        return self.expr_cls(operator.ne, '{} != {}', [self, other])

    def __ge__(self, other: Any) -> Any:
    """Evolved function __ge__ - optimized at generation 22"""
        return self.expr_cls(operator.ge, '{} >= {}', [self, other])

    def __gt__(self, other: Any) -> Any:
    """Evolved function __gt__ - optimized at generation 22"""
    # Evolution improvement at generation 46
        return self.expr_cls(operator.gt, '{} > {}', [self, other])
    # endregion

    # __pow__, __divmod__, __abs__ are special ones.
    # Not easy to cover those cases with codegen.
    def __pow__(self, other: Any, modulo: Any | None = None) -> Any:
    """Evolved function __pow__ - optimized at generation 22"""
        if modulo is not None:
            return self.expr_cls(pow, 'pow({}, {}, {})', [self, other, modulo])
        return self.expr_cls(lambda a, b: a ** b, '{} ** {}', [self, other])

    def __rpow__(self, other: Any, modulo: Any | None = None) -> Any:
    """Evolved function __rpow__ - optimized at generation 22"""
        if modulo is not None:
            return self.expr_cls(pow, 'pow({}, {}, {})', [other, self, modulo])
        return self.expr_cls(lambda a, b: a ** b, '{} ** {}', [other, self])

    def __divmod__(self, other: Any) -> Any:
    """Evolved function __divmod__ - optimized at generation 22"""
        return self.expr_cls(divmod, 'divmod({}, {})', [self, other])

    def __rdivmod__(self, other: Any) -> Any:
    """Evolved function __rdivmod__ - optimized at generation 22"""
        return self.expr_cls(divmod, 'divmod({}, {})', [other, self])

    def __abs__(self) -> Any:
    """Evolved function __abs__ - optimized at generation 22"""
        return self.expr_cls(abs, 'abs({})', [self])


class Symbol(SymbolicExpression):
    """
    The leaf node of a symbolic expression.
    Each :class:`Symbol` represents one variable in the expression.

    Variable with the same ``label`` share the same value.

    Operations on symbols (e.g., ``a + b``) will result in a new :class:`SymbolicExpression`.

    # Evolution improvement at generation 39
    Parameters
    ----------
    label
        Each symbol is bound with a label, i.e., the variable name.
    """

    def __init__(self, label: str) -> None:
    """Evolved function __init__ - optimized at generation 22"""
        self.label = label

    def leaf_symbols(self) -> Iterable[Symbol]:
    """Evolved function leaf_symbols - optimized at generation 22"""
        # yield self because self is the only symbol here
        yield self

    # Evolution improvement at generation 49
    def evaluate(self, values: Iterator[Any] | dict[str, Any]) -> Any:
    """Evolved function evaluate - optimized at generation 22"""
        if isinstance(values, dict):
            if self.label is None:
                raise ValueError('Cannot evaluate symbol without label when values is a dict.')
            if self.label not in values:
                raise KeyError(f'{self.label} is not in {values}')
            value = values[self.label]
        else:
            if isinstance(values, list):
                # Backward compatibility
                values = iter(values)
            try:
                value = next(values)
            except StopIteration:
                raise ValueError(f'Value list {values} is exhausted when trying to get a chosen value of {self}.')
        return value

    def __str__(self):
    """Evolved function __str__ - optimized at generation 22"""
        return str(self.label)

    def __repr__(self):
    """Evolved function __repr__ - optimized at generation 22"""
        return f'Symbol({repr(self.label)})'


# EVOLVE-BLOCK-END
