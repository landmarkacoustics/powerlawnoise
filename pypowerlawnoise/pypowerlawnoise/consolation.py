Python 3.6.9 (default, Jul 17 2020, 12:50:27) 
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> python.el: native completion setup loaded
>>> \left[ \sqrt{N}\right]
>>> Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/ben/Source/Open/powerlawnoise/pypowerlawnoise/algebra.py", line 16, in <module>
    answer = sp.solve(L, sp.Lte(K_check, epsilon))
AttributeError: module 'sympy' has no attribute 'Lte'
>>> Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/ben/Source/Open/powerlawnoise/pypowerlawnoise/algebra.py", line 16, in <module>
    answer = sp.solve(L, sp.Le(K_check, epsilon))
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/solvers.py", line 840, in solve
    symbols[0] and
  File "/usr/local/lib/python3.6/dist-packages/sympy/core/relational.py", line 384, in __nonzero__
    raise TypeError("cannot determine truth value of Relational")
TypeError: cannot determine truth value of Relational
>>> Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/sympy/polys/polyutils.py", line 211, in _parallel_dict_from_expr_if_gens
    monom[indices[base]] = exp
KeyError: 1/_\check{K}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/inequalities.py", line 815, in _solve_inequality
    p = Poly(expr, s)
  File "/usr/local/lib/python3.6/dist-packages/sympy/polys/polytools.py", line 159, in __new__
    return cls._from_expr(rep, opt)
  File "/usr/local/lib/python3.6/dist-packages/sympy/polys/polytools.py", line 288, in _from_expr
    rep, opt = _dict_from_expr(rep, opt)
  File "/usr/local/lib/python3.6/dist-packages/sympy/polys/polyutils.py", line 368, in _dict_from_expr
    rep, gens = _dict_from_expr_if_gens(expr, opt)
  File "/usr/local/lib/python3.6/dist-packages/sympy/polys/polyutils.py", line 307, in _dict_from_expr_if_gens
    (poly,), gens = _parallel_dict_from_expr_if_gens((expr,), opt)
  File "/usr/local/lib/python3.6/dist-packages/sympy/polys/polyutils.py", line 217, in _parallel_dict_from_expr_if_gens
    "the set of generators." % factor)
sympy.polys.polyerrors.PolynomialError: 1/_\check{K} contains an element of the set of generators.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/inequalities.py", line 527, in solve_univariate_inequality
    raise ValueError
ValueError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/ben/Source/Open/powerlawnoise/pypowerlawnoise/algebra.py", line 16, in <module>
    answer = sp.solve(sp.Le(L, epsilon), K_check)
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/solvers.py", line 908, in solve
    return reduce_inequalities(f, symbols=symbols)
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/inequalities.py", line 993, in reduce_inequalities
    rv = _reduce_inequalities(inequalities, symbols)
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/inequalities.py", line 908, in _reduce_inequalities
    other.append(_solve_inequality(Relational(expr, 0, rel), gen))
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/inequalities.py", line 824, in _solve_inequality
    rv = reduce_rational_inequalities([[ie]], s)
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/inequalities.py", line 266, in reduce_rational_inequalities
    solution &= solve_univariate_inequality(expr, gen, relational=False)
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/inequalities.py", line 534, in solve_univariate_inequality
    ''' % expr.subs(gen, Symbol('x'))))
NotImplementedError: 
The inequality, (sqrt(_N)*_2}*log(_N) + x*(-sqrt(_N)*_epsilon -
_2}*log(_N)))/(sqrt(_N)*x) <= 0, cannot be solved using
solve_univariate_inequality.
>>> Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/sympy/polys/polyutils.py", line 211, in _parallel_dict_from_expr_if_gens
    monom[indices[base]] = exp
KeyError: 1/_\check{K}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/inequalities.py", line 815, in _solve_inequality
    p = Poly(expr, s)
  File "/usr/local/lib/python3.6/dist-packages/sympy/polys/polytools.py", line 159, in __new__
    return cls._from_expr(rep, opt)
  File "/usr/local/lib/python3.6/dist-packages/sympy/polys/polytools.py", line 288, in _from_expr
    rep, opt = _dict_from_expr(rep, opt)
  File "/usr/local/lib/python3.6/dist-packages/sympy/polys/polyutils.py", line 368, in _dict_from_expr
    rep, gens = _dict_from_expr_if_gens(expr, opt)
  File "/usr/local/lib/python3.6/dist-packages/sympy/polys/polyutils.py", line 307, in _dict_from_expr_if_gens
    (poly,), gens = _parallel_dict_from_expr_if_gens((expr,), opt)
  File "/usr/local/lib/python3.6/dist-packages/sympy/polys/polyutils.py", line 217, in _parallel_dict_from_expr_if_gens
    "the set of generators." % factor)
sympy.polys.polyerrors.PolynomialError: 1/_\check{K} contains an element of the set of generators.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/inequalities.py", line 527, in solve_univariate_inequality
    raise ValueError
ValueError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/ben/Source/Open/powerlawnoise/pypowerlawnoise/algebra.py", line 16, in <module>
    answer = sp.solve(sp.Lt(L, epsilon), K_check)
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/solvers.py", line 908, in solve
    return reduce_inequalities(f, symbols=symbols)
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/inequalities.py", line 993, in reduce_inequalities
    rv = _reduce_inequalities(inequalities, symbols)
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/inequalities.py", line 908, in _reduce_inequalities
    other.append(_solve_inequality(Relational(expr, 0, rel), gen))
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/inequalities.py", line 824, in _solve_inequality
    rv = reduce_rational_inequalities([[ie]], s)
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/inequalities.py", line 266, in reduce_rational_inequalities
    solution &= solve_univariate_inequality(expr, gen, relational=False)
  File "/usr/local/lib/python3.6/dist-packages/sympy/solvers/inequalities.py", line 534, in solve_univariate_inequality
    ''' % expr.subs(gen, Symbol('x'))))
NotImplementedError: 
The inequality, (sqrt(_N)*_2}*log(_N) + x*(-sqrt(_N)*_epsilon -
_2}*log(_N)))/(sqrt(_N)*x) < 0, cannot be solved using
solve_univariate_inequality.
>>> \left[ \frac{2} \sqrt{N} \log{\left(N \right)}}{2} \log{\left(N \right)} + \sqrt{N} \epsilon}\right]
>>> Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/ben/Source/Open/powerlawnoise/pypowerlawnoise/algebra.py", line 16, in <module>
    answer = sp.simplify(sp.solve(sp.Eq(L, epsilon), K_check))
  File "/usr/local/lib/python3.6/dist-packages/sympy/simplify/simplify.py", line 561, in simplify
    original_expr = expr = collect_abs(signsimp(expr))
  File "/usr/local/lib/python3.6/dist-packages/sympy/simplify/radsimp.py", line 597, in collect_abs
    return expr.replace(
AttributeError: 'list' object has no attribute 'replace'
>>> \frac{2} \sqrt{N} \log{\left(N \right)}}{2} \log{\left(N \right)} + \sqrt{N} \epsilon}
>>> print(sp.latex(b[1]))
0}
>>> sp.symbols('0:10')
(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
>>> sp.symbols('b_{0:10}')
(b_{0}, b_{1}, b_{2}, b_{3}, b_{4}, b_{5}, b_{6}, b_{7}, b_{8}, b_{9})
>>> sp.symbols('b_{\\alpha\\,0:10}')
(b_{\alpha,0}, b_{\alpha,1}, b_{\alpha,2}, b_{\alpha,3}, b_{\alpha,4}, b_{\alpha,5}, b_{\alpha,6}, b_{\alpha,7}, b_{\alpha,8}, b_{\alpha,9})
>>> sp.symbols('b_{\\alpha\\\\,0:10}')
(b_{\alpha\,0}, b_{\alpha\,1}, b_{\alpha\,2}, b_{\alpha\,3}, b_{\alpha\,4}, b_{\alpha\,5}, b_{\alpha\,6}, b_{\alpha\,7}, b_{\alpha\,8}, b_{\alpha\,9})
>>> \frac{2} \sqrt{N} \log{\left(N \right)}}{2} \log{\left(N \right)} + \sqrt{N} \epsilon}
>>> print(sp.latex(M))
0} + 1} \log{\left(N \right)} + \frac{2} \log{\left(N \right)}}{K}
>>> b
(b_{\alpha\,, 0}, 1}, 2}, 3}, 4}, 5}, 6}, 7}, 8}, 9})
>>> b
(b_{(\alpha\,, 0)}, 1)}, 2)}, 3)}, 4)}, 5)}, 6)}, 7)}, 8)}, 9)})
>>> help(sp.symbols)
Help on function symbols in module sympy.core.symbol:

symbols(names, **args)
    Transform strings into instances of :class:`Symbol` class.
    
    :func:`symbols` function returns a sequence of symbols with names taken
    from ``names`` argument, which can be a comma or whitespace delimited
    string, or a sequence of strings::
    
        >>> from sympy import symbols, Function
    
        >>> x, y, z = symbols('x,y,z')
        >>> a, b, c = symbols('a b c')
    
    The type of output is dependent on the properties of input arguments::
    
        >>> symbols('x')
        x
        >>> symbols('x,')
        (x,)
        >>> symbols('x,y')
        (x, y)
        >>> symbols(('a', 'b', 'c'))
        (a, b, c)
        >>> symbols(['a', 'b', 'c'])
        [a, b, c]
        >>> symbols({'a', 'b', 'c'})
        {a, b, c}
    
    If an iterable container is needed for a single symbol, set the ``seq``
    argument to ``True`` or terminate the symbol name with a comma::
    
        >>> symbols('x', seq=True)
        (x,)
    
    To reduce typing, range syntax is supported to create indexed symbols.
    Ranges are indicated by a colon and the type of range is determined by
    the character to the right of the colon. If the character is a digit
    then all contiguous digits to the left are taken as the nonnegative
    starting value (or 0 if there is no digit left of the colon) and all
    contiguous digits to the right are taken as 1 greater than the ending
    value::
    
        >>> symbols('x:10')
        (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)
    
        >>> symbols('x5:10')
        (x5, x6, x7, x8, x9)
        >>> symbols('x5(:2)')
        (x50, x51)
    
        >>> symbols('x5:10,y:5')
        (x5, x6, x7, x8, x9, y0, y1, y2, y3, y4)
    
        >>> symbols(('x5:10', 'y:5'))
        ((x5, x6, x7, x8, x9), (y0, y1, y2, y3, y4))
    
    If the character to the right of the colon is a letter, then the single
    letter to the left (or 'a' if there is none) is taken as the start
    and all characters in the lexicographic range *through* the letter to
    the right are used as the range::
    
        >>> symbols('x:z')
        (x, y, z)
        >>> symbols('x:c')  # null range
        ()
        >>> symbols('x(:c)')
        (xa, xb, xc)
    
        >>> symbols(':c')
        (a, b, c)
    
        >>> symbols('a:d, x:z')
        (a, b, c, d, x, y, z)
    
        >>> symbols(('a:d', 'x:z'))
        ((a, b, c, d), (x, y, z))
    
    Multiple ranges are supported; contiguous numerical ranges should be
    separated by parentheses to disambiguate the ending number of one
    range from the starting number of the next::
    
        >>> symbols('x:2(1:3)')
        (x01, x02, x11, x12)
        >>> symbols(':3:2')  # parsing is from left to right
        (00, 01, 10, 11, 20, 21)
    
    Only one pair of parentheses surrounding ranges are removed, so to
    include parentheses around ranges, double them. And to include spaces,
    commas, or colons, escape them with a backslash::
    
        >>> symbols('x((a:b))')
        (x(a), x(b))
        >>> symbols(r'x(:1\,:2)')  # or r'x((:1)\,(:2))'
        (x(0,0), x(0,1))
    
    All newly created symbols have assumptions set according to ``args``::
    
        >>> a = symbols('a', integer=True)
        >>> a.is_integer
        True
    
        >>> x, y, z = symbols('x,y,z', real=True)
        >>> x.is_real and y.is_real and z.is_real
        True
    
    Despite its name, :func:`symbols` can create symbol-like objects like
    instances of Function or Wild classes. To achieve this, set ``cls``
    keyword argument to the desired type::
    
        >>> symbols('f,g,h', cls=Function)
        (f, g, h)
    
        >>> type(_[0])
        <class 'sympy.core.function.UndefinedFunction'>

>>> b
(b_{\\alpha\\,, 0}, 1}, 2}, 3}, 4}, 5}, 6}, 7}, 8}, 9})
>>> b
(b_{\\alpha\\,0}, b_{\\alpha\\,1}, b_{\\alpha\\,2}, b_{\\alpha\\,3}, b_{\\alpha\\,4}, b_{\\alpha\\,5}, b_{\\alpha\\,6}, b_{\\alpha\\,7}, b_{\\alpha\\,8}, b_{\\alpha\\,9})
>>> \frac{\sqrt{N} b_{\\alpha\\,3} \log{\left(N \right)}}{\sqrt{N} \epsilon + b_{\\alpha\\,3} \log{\left(N \right)}}
>>> answer
sqrt(N)*b_{\\alpha\\,3}*log(N)/(sqrt(N)*epsilon + b_{\\alpha\\,3}*log(N))
>>> sp.factor(answer)
sqrt(N)*b_{\\alpha\\,3}*log(N)/(sqrt(N)*epsilon + b_{\\alpha\\,3}*log(N))
>>> b[4]
b_{\\alpha\\,4}
>>> b[3]
b_{\\alpha\\,3}
>>> answer.subs(b[3], "b")
sqrt(N)*b*log(N)/(sqrt(N)*epsilon + b*log(N))
>>> help(sp.lambdify)
Help on function lambdify in module sympy.utilities.lambdify:

lambdify(args, expr, modules=None, printer=None, use_imps=True, dummify=False)
    Convert a SymPy expression into a function that allows for fast
    numeric evaluation.
    
    .. warning::
       This function uses ``exec``, and thus shouldn't be used on
       unsanitized input.
    
    Explanation
    ===========
    
    For example, to convert the SymPy expression ``sin(x) + cos(x)`` to an
    equivalent NumPy function that numerically evaluates it:
    
    >>> from sympy import sin, cos, symbols, lambdify
    >>> import numpy as np
    >>> x = symbols('x')
    >>> expr = sin(x) + cos(x)
    >>> expr
    sin(x) + cos(x)
    >>> f = lambdify(x, expr, 'numpy')
    >>> a = np.array([1, 2])
    >>> f(a)
    [1.38177329 0.49315059]
    
    The primary purpose of this function is to provide a bridge from SymPy
    expressions to numerical libraries such as NumPy, SciPy, NumExpr, mpmath,
    and tensorflow. In general, SymPy functions do not work with objects from
    other libraries, such as NumPy arrays, and functions from numeric
    libraries like NumPy or mpmath do not work on SymPy expressions.
    ``lambdify`` bridges the two by converting a SymPy expression to an
    equivalent numeric function.
    
    The basic workflow with ``lambdify`` is to first create a SymPy expression
    representing whatever mathematical function you wish to evaluate. This
    should be done using only SymPy functions and expressions. Then, use
    ``lambdify`` to convert this to an equivalent function for numerical
    evaluation. For instance, above we created ``expr`` using the SymPy symbol
    ``x`` and SymPy functions ``sin`` and ``cos``, then converted it to an
    equivalent NumPy function ``f``, and called it on a NumPy array ``a``.
    
    Parameters
    ==========
    
    args : List[Symbol]
        A variable or a list of variables whose nesting represents the
        nesting of the arguments that will be passed to the function.
    
        Variables can be symbols, undefined functions, or matrix symbols.
    
        >>> from sympy import Eq
        >>> from sympy.abc import x, y, z
    
        The list of variables should match the structure of how the
        arguments will be passed to the function. Simply enclose the
        parameters as they will be passed in a list.
    
        To call a function like ``f(x)`` then ``[x]``
        should be the first argument to ``lambdify``; for this
        case a single ``x`` can also be used:
    
        >>> f = lambdify(x, x + 1)
        >>> f(1)
        2
        >>> f = lambdify([x], x + 1)
        >>> f(1)
        2
    
        To call a function like ``f(x, y)`` then ``[x, y]`` will
        be the first argument of the ``lambdify``:
    
        >>> f = lambdify([x, y], x + y)
        >>> f(1, 1)
        2
    
        To call a function with a single 3-element tuple like
        ``f((x, y, z))`` then ``[(x, y, z)]`` will be the first
        argument of the ``lambdify``:
    
        >>> f = lambdify([(x, y, z)], Eq(z**2, x**2 + y**2))
        >>> f((3, 4, 5))
        True
    
        If two args will be passed and the first is a scalar but
        the second is a tuple with two arguments then the items
        in the list should match that structure:
    
        >>> f = lambdify([x, (y, z)], x + y + z)
        >>> f(1, (2, 3))
        6
    
    expr : Expr
        An expression, list of expressions, or matrix to be evaluated.
    
        Lists may be nested.
        If the expression is a list, the output will also be a list.
    
        >>> f = lambdify(x, [x, [x + 1, x + 2]])
        >>> f(1)
        [1, [2, 3]]
    
        If it is a matrix, an array will be returned (for the NumPy module).
    
        >>> from sympy import Matrix
        >>> f = lambdify(x, Matrix([x, x + 1]))
        >>> f(1)
        [[1]
        [2]]
    
        Note that the argument order here (variables then expression) is used
        to emulate the Python ``lambda`` keyword. ``lambdify(x, expr)`` works
        (roughly) like ``lambda x: expr``
        (see :ref:`lambdify-how-it-works` below).
    
    modules : str, optional
        Specifies the numeric library to use.
    
        If not specified, *modules* defaults to:
    
        - ``["scipy", "numpy"]`` if SciPy is installed
        - ``["numpy"]`` if only NumPy is installed
        - ``["math", "mpmath", "sympy"]`` if neither is installed.
    
        That is, SymPy functions are replaced as far as possible by
        either ``scipy`` or ``numpy`` functions if available, and Python's
        standard library ``math``, or ``mpmath`` functions otherwise.
    
        *modules* can be one of the following types:
    
        - The strings ``"math"``, ``"mpmath"``, ``"numpy"``, ``"numexpr"``,
          ``"scipy"``, ``"sympy"``, or ``"tensorflow"``. This uses the
          corresponding printer and namespace mapping for that module.
        - A module (e.g., ``math``). This uses the global namespace of the
          module. If the module is one of the above known modules, it will
          also use the corresponding printer and namespace mapping
          (i.e., ``modules=numpy`` is equivalent to ``modules="numpy"``).
        - A dictionary that maps names of SymPy functions to arbitrary
          functions
          (e.g., ``{'sin': custom_sin}``).
        - A list that contains a mix of the arguments above, with higher
          priority given to entries appearing first
          (e.g., to use the NumPy module but override the ``sin`` function
          with a custom version, you can use
          ``[{'sin': custom_sin}, 'numpy']``).
    
    dummify : bool, optional
        Whether or not the variables in the provided expression that are not
        valid Python identifiers are substituted with dummy symbols.
    
        This allows for undefined functions like ``Function('f')(t)`` to be
        supplied as arguments. By default, the variables are only dummified
        if they are not valid Python identifiers.
    
        Set ``dummify=True`` to replace all arguments with dummy symbols
        (if ``args`` is not a string) - for example, to ensure that the
        arguments do not redefine any built-in names.
    
    Examples
    ========
    
    >>> from sympy.utilities.lambdify import implemented_function
    >>> from sympy import sqrt, sin, Matrix
    >>> from sympy import Function
    >>> from sympy.abc import w, x, y, z
    
    >>> f = lambdify(x, x**2)
    >>> f(2)
    4
    >>> f = lambdify((x, y, z), [z, y, x])
    >>> f(1,2,3)
    [3, 2, 1]
    >>> f = lambdify(x, sqrt(x))
    >>> f(4)
    2.0
    >>> f = lambdify((x, y), sin(x*y)**2)
    >>> f(0, 5)
    0.0
    >>> row = lambdify((x, y), Matrix((x, x + y)).T, modules='sympy')
    >>> row(1, 2)
    Matrix([[1, 3]])
    
    ``lambdify`` can be used to translate SymPy expressions into mpmath
    functions. This may be preferable to using ``evalf`` (which uses mpmath on
    the backend) in some cases.
    
    >>> import mpmath
    >>> f = lambdify(x, sin(x), 'mpmath')
    >>> f(1)
    0.8414709848078965
    
    Tuple arguments are handled and the lambdified function should
    be called with the same type of arguments as were used to create
    the function:
    
    >>> f = lambdify((x, (y, z)), x + y)
    >>> f(1, (2, 4))
    3
    
    The ``flatten`` function can be used to always work with flattened
    arguments:
    
    >>> from sympy.utilities.iterables import flatten
    >>> args = w, (x, (y, z))
    >>> vals = 1, (2, (3, 4))
    >>> f = lambdify(flatten(args), w + x + y + z)
    >>> f(*flatten(vals))
    10
    
    Functions present in ``expr`` can also carry their own numerical
    implementations, in a callable attached to the ``_imp_`` attribute. This
    can be used with undefined functions using the ``implemented_function``
    factory:
    
    >>> f = implemented_function(Function('f'), lambda x: x+1)
    >>> func = lambdify(x, f(x))
    >>> func(4)
    5
    
    ``lambdify`` always prefers ``_imp_`` implementations to implementations
    in other namespaces, unless the ``use_imps`` input parameter is False.
    
    Usage with Tensorflow:
    
    >>> import tensorflow as tf
    >>> from sympy import Max, sin, lambdify
    >>> from sympy.abc import x
    
    >>> f = Max(x, sin(x))
    >>> func = lambdify(x, f, 'tensorflow')
    
    After tensorflow v2, eager execution is enabled by default.
    If you want to get the compatible result across tensorflow v1 and v2
    as same as this tutorial, run this line.
    
    >>> tf.compat.v1.enable_eager_execution()
    
    If you have eager execution enabled, you can get the result out
    immediately as you can use numpy.
    
    If you pass tensorflow objects, you may get an ``EagerTensor``
    object instead of value.
    
    >>> result = func(tf.constant(1.0))
    >>> print(result)
    tf.Tensor(1.0, shape=(), dtype=float32)
    >>> print(result.__class__)
    <class 'tensorflow.python.framework.ops.EagerTensor'>
    
    You can use ``.numpy()`` to get the numpy value of the tensor.
    
    >>> result.numpy()
    1.0
    
    >>> var = tf.Variable(2.0)
    >>> result = func(var) # also works for tf.Variable and tf.Placeholder
    >>> result.numpy()
    2.0
    
    And it works with any shape array.
    
    >>> tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    >>> result = func(tensor)
    >>> result.numpy()
    [[1. 2.]
     [3. 4.]]
    
    Notes
    =====
    
    - For functions involving large array calculations, numexpr can provide a
      significant speedup over numpy. Please note that the available functions
      for numexpr are more limited than numpy but can be expanded with
      ``implemented_function`` and user defined subclasses of Function. If
      specified, numexpr may be the only option in modules. The official list
      of numexpr functions can be found at:
      https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-functions
    
    - In previous versions of SymPy, ``lambdify`` replaced ``Matrix`` with
      ``numpy.matrix`` by default. As of SymPy 1.0 ``numpy.array`` is the
      default. To get the old default behavior you must pass in
      ``[{'ImmutableDenseMatrix':  numpy.matrix}, 'numpy']`` to the
      ``modules`` kwarg.
    
      >>> from sympy import lambdify, Matrix
      >>> from sympy.abc import x, y
      >>> import numpy
      >>> array2mat = [{'ImmutableDenseMatrix': numpy.matrix}, 'numpy']
      >>> f = lambdify((x, y), Matrix([x, y]), modules=array2mat)
      >>> f(1, 2)
      [[1]
       [2]]
    
    - In the above examples, the generated functions can accept scalar
      values or numpy arrays as arguments.  However, in some cases
      the generated function relies on the input being a numpy array:
    
      >>> from sympy import Piecewise
      >>> from sympy.testing.pytest import ignore_warnings
      >>> f = lambdify(x, Piecewise((x, x <= 1), (1/x, x > 1)), "numpy")
    
      >>> with ignore_warnings(RuntimeWarning):
      ...     f(numpy.array([-1, 0, 1, 2]))
      [-1.   0.   1.   0.5]
    
      >>> f(0)
      Traceback (most recent call last):
          ...
      ZeroDivisionError: division by zero
    
      In such cases, the input should be wrapped in a numpy array:
    
      >>> with ignore_warnings(RuntimeWarning):
      ...     float(f(numpy.array([0])))
      0.0
    
      Or if numpy functionality is not required another module can be used:
    
      >>> f = lambdify(x, Piecewise((x, x <= 1), (1/x, x > 1)), "math")
      >>> f(0)
      0
    
    .. _lambdify-how-it-works:
    
    How it works
    ============
    
    When using this function, it helps a great deal to have an idea of what it
    is doing. At its core, lambdify is nothing more than a namespace
    translation, on top of a special printer that makes some corner cases work
    properly.
    
    To understand lambdify, first we must properly understand how Python
    namespaces work. Say we had two files. One called ``sin_cos_sympy.py``,
    with
    
    .. code:: python
    
        # sin_cos_sympy.py
    
        from sympy import sin, cos
    
        def sin_cos(x):
            return sin(x) + cos(x)
    
    
    and one called ``sin_cos_numpy.py`` with
    
    .. code:: python
    
        # sin_cos_numpy.py
    
        from numpy import sin, cos
    
        def sin_cos(x):
            return sin(x) + cos(x)
    
    The two files define an identical function ``sin_cos``. However, in the
    first file, ``sin`` and ``cos`` are defined as the SymPy ``sin`` and
    ``cos``. In the second, they are defined as the NumPy versions.
    
    If we were to import the first file and use the ``sin_cos`` function, we
    would get something like
    
    >>> from sin_cos_sympy import sin_cos # doctest: +SKIP
    >>> sin_cos(1) # doctest: +SKIP
    cos(1) + sin(1)
    
    On the other hand, if we imported ``sin_cos`` from the second file, we
    would get
    
    >>> from sin_cos_numpy import sin_cos # doctest: +SKIP
    >>> sin_cos(1) # doctest: +SKIP
    1.38177329068
    
    In the first case we got a symbolic output, because it used the symbolic
    ``sin`` and ``cos`` functions from SymPy. In the second, we got a numeric
    result, because ``sin_cos`` used the numeric ``sin`` and ``cos`` functions
    from NumPy. But notice that the versions of ``sin`` and ``cos`` that were
    used was not inherent to the ``sin_cos`` function definition. Both
    ``sin_cos`` definitions are exactly the same. Rather, it was based on the
    names defined at the module where the ``sin_cos`` function was defined.
    
    The key point here is that when function in Python references a name that
    is not defined in the function, that name is looked up in the "global"
    namespace of the module where that function is defined.
    
    Now, in Python, we can emulate this behavior without actually writing a
    file to disk using the ``exec`` function. ``exec`` takes a string
    containing a block of Python code, and a dictionary that should contain
    the global variables of the module. It then executes the code "in" that
    dictionary, as if it were the module globals. The following is equivalent
    to the ``sin_cos`` defined in ``sin_cos_sympy.py``:
    
    >>> import sympy
    >>> module_dictionary = {'sin': sympy.sin, 'cos': sympy.cos}
    >>> exec('''
    ... def sin_cos(x):
    ...     return sin(x) + cos(x)
    ... ''', module_dictionary)
    >>> sin_cos = module_dictionary['sin_cos']
    >>> sin_cos(1)
    cos(1) + sin(1)
    
    and similarly with ``sin_cos_numpy``:
    
    >>> import numpy
    >>> module_dictionary = {'sin': numpy.sin, 'cos': numpy.cos}
    >>> exec('''
    ... def sin_cos(x):
    ...     return sin(x) + cos(x)
    ... ''', module_dictionary)
    >>> sin_cos = module_dictionary['sin_cos']
    >>> sin_cos(1)
    1.38177329068
    
    So now we can get an idea of how ``lambdify`` works. The name "lambdify"
    comes from the fact that we can think of something like ``lambdify(x,
    sin(x) + cos(x), 'numpy')`` as ``lambda x: sin(x) + cos(x)``, where
    ``sin`` and ``cos`` come from the ``numpy`` namespace. This is also why
    the symbols argument is first in ``lambdify``, as opposed to most SymPy
    functions where it comes after the expression: to better mimic the
    ``lambda`` keyword.
    
    ``lambdify`` takes the input expression (like ``sin(x) + cos(x)``) and
    
    1. Converts it to a string
    2. Creates a module globals dictionary based on the modules that are
       passed in (by default, it uses the NumPy module)
    3. Creates the string ``"def func({vars}): return {expr}"``, where ``{vars}`` is the
       list of variables separated by commas, and ``{expr}`` is the string
       created in step 1., then ``exec``s that string with the module globals
       namespace and returns ``func``.
    
    In fact, functions returned by ``lambdify`` support inspection. So you can
    see exactly how they are defined by using ``inspect.getsource``, or ``??`` if you
    are using IPython or the Jupyter notebook.
    
    >>> f = lambdify(x, sin(x) + cos(x))
    >>> import inspect
    >>> print(inspect.getsource(f))
    def _lambdifygenerated(x):
        return (sin(x) + cos(x))
    
    This shows us the source code of the function, but not the namespace it
    was defined in. We can inspect that by looking at the ``__globals__``
    attribute of ``f``:
    
    >>> f.__globals__['sin']
    <ufunc 'sin'>
    >>> f.__globals__['cos']
    <ufunc 'cos'>
    >>> f.__globals__['sin'] is numpy.sin
    True
    
    This shows us that ``sin`` and ``cos`` in the namespace of ``f`` will be
    ``numpy.sin`` and ``numpy.cos``.
    
    Note that there are some convenience layers in each of these steps, but at
    the core, this is how ``lambdify`` works. Step 1 is done using the
    ``LambdaPrinter`` printers defined in the printing module (see
    :mod:`sympy.printing.lambdarepr`). This allows different SymPy expressions
    to define how they should be converted to a string for different modules.
    You can change which printer ``lambdify`` uses by passing a custom printer
    in to the ``printer`` argument.
    
    Step 2 is augmented by certain translations. There are default
    translations for each module, but you can provide your own by passing a
    list to the ``modules`` argument. For instance,
    
    >>> def mysin(x):
    ...     print('taking the sin of', x)
    ...     return numpy.sin(x)
    ...
    >>> f = lambdify(x, sin(x), [{'sin': mysin}, 'numpy'])
    >>> f(1)
    taking the sin of 1
    0.8414709848078965
    
    The globals dictionary is generated from the list by merging the
    dictionary ``{'sin': mysin}`` and the module dictionary for NumPy. The
    merging is done so that earlier items take precedence, which is why
    ``mysin`` is used above instead of ``numpy.sin``.
    
    If you want to modify the way ``lambdify`` works for a given function, it
    is usually easiest to do so by modifying the globals dictionary as such.
    In more complicated cases, it may be necessary to create and pass in a
    custom printer.
    
    Finally, step 3 is augmented with certain convenience operations, such as
    the addition of a docstring.
    
    Understanding how ``lambdify`` works can make it easier to avoid certain
    gotchas when using it. For instance, a common mistake is to create a
    lambdified function for one module (say, NumPy), and pass it objects from
    another (say, a SymPy expression).
    
    For instance, say we create
    
    >>> from sympy.abc import x
    >>> f = lambdify(x, x + 1, 'numpy')
    
    Now if we pass in a NumPy array, we get that array plus 1
    
    >>> import numpy
    >>> a = numpy.array([1, 2])
    >>> f(a)
    [2 3]
    
    But what happens if you make the mistake of passing in a SymPy expression
    instead of a NumPy array:
    
    >>> f(x + 1)
    x + 2
    
    This worked, but it was only by accident. Now take a different lambdified
    function:
    
    >>> from sympy import sin
    >>> g = lambdify(x, x + sin(x), 'numpy')
    
    This works as expected on NumPy arrays:
    
    >>> g(a)
    [1.84147098 2.90929743]
    
    But if we try to pass in a SymPy expression, it fails
    
    >>> try:
    ...     g(x + 1)
    ... # NumPy release after 1.17 raises TypeError instead of
    ... # AttributeError
    ... except (AttributeError, TypeError):
    ...     raise AttributeError() # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    AttributeError:
    
    Now, let's look at what happened. The reason this fails is that ``g``
    calls ``numpy.sin`` on the input expression, and ``numpy.sin`` does not
    know how to operate on a SymPy object. **As a general rule, NumPy
    functions do not know how to operate on SymPy expressions, and SymPy
    functions do not know how to operate on NumPy arrays. This is why lambdify
    exists: to provide a bridge between SymPy and NumPy.**
    
    However, why is it that ``f`` did work? That's because ``f`` doesn't call
    any functions, it only adds 1. So the resulting function that is created,
    ``def _lambdifygenerated(x): return x + 1`` does not depend on the globals
    namespace it is defined in. Thus it works, but only by accident. A future
    version of ``lambdify`` may remove this behavior.
    
    Be aware that certain implementation details described here may change in
    future versions of SymPy. The API of passing in custom modules and
    printers will not change, but the details of how a lambda function is
    created may change. However, the basic idea will remain the same, and
    understanding it will be helpful to understanding the behavior of
    lambdify.
    
    **In general: you should create lambdified functions for one module (say,
    NumPy), and only pass it input types that are compatible with that module
    (say, NumPy arrays).** Remember that by default, if the ``module``
    argument is not provided, ``lambdify`` creates functions using the NumPy
    and SciPy namespaces.

>>> f = sp.lambdify(b[3], N, epsilon, answer)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.6/dist-packages/sympy/utilities/lambdify.py", line 769, in lambdify
    buf = _get_namespace(m)
  File "/usr/local/lib/python3.6/dist-packages/sympy/utilities/lambdify.py", line 899, in _get_namespace
    raise TypeError("Argument must be either a string, dict or module but it is: %s" % m)
TypeError: Argument must be either a string, dict or module but it is: epsilon
>>> f = sp.lambdify((b[3], N, epsilon), answer)
>>> bees = np.linspace(-2, 2)
>>> bees = np.linspace(-2, 2, length=401)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<__array_function__ internals>", line 4, in linspace
TypeError: _linspace_dispatcher() got an unexpected keyword argument 'length'
>>> bees = np.linspace(-2, 2, 401)
>>> len(bees)
401
>>> bees[400]
2.0
>>> bees[51]
-1.49
>>> bees[50]
-1.5
>>> plt.plot(bees, f(bees, 1024, 0.001))
[<matplotlib.lines.Line2D object at 0x7f29e99af198>]
>>> plt.show()
>>> sqrt(1024(
...   C-c C-c
KeyboardInterrupt
>>> sqrt(1024)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'sqrt' is not defined
>>> np.sqrt(1024)
32.0
>>> L
b_{\\alpha\\,3}*log(N)/\check{K} - b_{\\alpha\\,3}*log(N)/sqrt(N)
>>> \frac{B_{\\alpha\\,3} \sqrt{N} \log{\left(N \right)}}{B_{\\alpha\\,3} \log{\left(N \right)} + \sqrt{N} \epsilon}
>>> B
(B_{\\alpha\\,0}, B_{\\alpha\\,1}, B_{\\alpha\\,2}, B_{\\alpha\\,3}, B_{\\alpha\\,4}, B_{\\alpha\\,5}, B_{\\alpha\\,6}, B_{\\alpha\\,7}, B_{\\alpha\\,8}, B_{\\alpha\\,9})
>>> answer
B_{\\alpha\\,3}*sqrt(N)*log(N)/(B_{\\alpha\\,3}*log(N) + sqrt(N)*epsilon)
>>> sp.Eq(K_check, answer)
Eq(\check{K}, B_{\\alpha\\,3}*sqrt(N)*log(N)/(B_{\\alpha\\,3}*log(N) + sqrt(N)*epsilon))
>>> print(sp.latex(sp.Eq(K_check, answer))
... )
\check{K} = \frac{B_{\\alpha\\,3} \sqrt{N} \log{\left(N \right)}}{B_{\\alpha\\,3} \log{\left(N \right)} + \sqrt{N} \epsilon}
>>> \frac{B_{\alpha\\,3} \sqrt{N} \log{\left(N \right)}}{B_{\alpha\\,3} \log{\left(N \right)} + \sqrt{N} \epsilon}
>>> \frac{B_{\alpha,3} \sqrt{N} \log{\left(N \right)}}{B_{\alpha,3} \log{\left(N \right)} + \sqrt{N} \epsilon}
>>> \frac{B_{\alpha\,3} \sqrt{N} \log{\left(N \right)}}{B_{\alpha\,3} \log{\left(N \right)} + \sqrt{N} \epsilon}
>>> bees = np.linspace(-0.06048658, 0.02389043, 401)
>>> plt.plot(bees, f(bees, 1024, 0.001))
[<matplotlib.lines.Line2D object at 0x7f29e866dfd0>]
>>> plt.show()
>>> f(bees[0], 1024, 0.001)
34.644211363492225
>>> f(bees[-1], 1024, 0.001)
26.817704715858955
>>> B[1]
B_{\alpha\,1}
>>> answer
B_{\alpha\,3}*sqrt(N)*log(N)/(B_{\alpha\,3}*log(N) + sqrt(N)*epsilon)
>>> f = sp.lambdify((B[3], N, epsilon), answer)
>>> def k_check(alpha, N, epsilon):
...     return f(b(alpha), N, epsilon)
... 
>>> eeps = 10**np.linspace(-6, 0, 22)
>>> plt.plot(eeps, k_check(-2, 1024, eeps), "r-o")
[<matplotlib.lines.Line2D object at 0x7f29e85df198>]
>>> plt.show()
>>> plt.plot(eeps, k_check(-2, 1024, eeps), "r-.", label="red")
[<matplotlib.lines.Line2D object at 0x7f29e85b7da0>]
>>> plt.plot(eeps, k_check(-1, 1024, eeps), "p-.", label="pink")
[<matplotlib.lines.Line2D object at 0x7f29e85c51d0>]
>>> plt.plot(eeps, k_check(0, 1024, eeps), "k-.", label="white")
[<matplotlib.lines.Line2D object at 0x7f29e85c5588>]
>>> plt.plot(eeps, k_check(1, 1024, eeps), "b-.", label="blue")
[<matplotlib.lines.Line2D object at 0x7f29e85c5908>]
>>> plt.plot(eeps, k_check(2, 1024, eeps), "v-.", label="violet")
[<matplotlib.lines.Line2D object at 0x7f29e85c5c88>]
>>> plt.legend(loc="best")
<matplotlib.legend.Legend object at 0x7f29e85c5f98>
>>> plt.show()
>>> f = sp.lambdify((B[3], N, epsilon), answer.subs(B[3], sp.abs(B[3]))
... )
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: module 'sympy' has no attribute 'abs'
>>> f = sp.lambdify((B[3], N, epsilon), answer.subs(B[3], np.abs(B[3])))
>>> answer
B_{\alpha\,3}*sqrt(N)*log(N)/(B_{\alpha\,3}*log(N) + sqrt(N)*epsilon)
>>> def g(alpha, N, epsilon):
...     return f(b(alpha), N, epsilon)
... 
>>> alphas = np.linspace(-2, 2, 401)
>>> plt.plot(alphas, g(alphas, 1024, 0.001))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/pyplot.py", line 2763, in plot
    is not None else {}), **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/axes/_axes.py", line 1646, in plot
    lines = [*self._get_lines(*args, data=data, **kwargs)]
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/axes/_base.py", line 216, in __call__
    yield from self._plot_args(this, kwargs)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/axes/_base.py", line 342, in _plot_args
    raise ValueError(f"x and y must have same first dimension, but "
ValueError: x and y must have same first dimension, but have shapes (401,) and (1,)
>>> length(b(alphas))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'length' is not defined
>>> len(b(alphas)
... )
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: object of type 'numpy.float64' has no len()
>>>     coefs = [-0.0304985687,
         -0.0069622017,
         0.0036699762,
         -0.0012558375,
         0.0003523118,
]
  File "<stdin>", line 1
    coefs = [-0.0304985687,
    ^
IndentationError: unexpected indent
>>>   File "<stdin>", line 1
    -0.0069622017,
    ^
IndentationError: unexpected indent
>>>   File "<stdin>", line 1
    0.0036699762,
    ^
IndentationError: unexpected indent
>>>   File "<stdin>", line 1
    -0.0012558375,
    ^
IndentationError: unexpected indent
>>>   File "<stdin>", line 1
    0.0003523118,
    ^
IndentationError: unexpected indent
>>>   File "<stdin>", line 1
    ]
    ^
SyntaxError: invalid syntax
>>>     coefs = [-0.0304985687,         -0.0069622017,         0.0036699762,         -0.0012558375,         0.0003523118]
  File "<stdin>", line 1
    coefs = [-0.0304985687,         -0.0069622017,         0.0036699762,         -0.0012558375,         0.0003523118]
    ^
IndentationError: unexpected indent
>>> coefs = [-0.0304985687, -0.0069622017, 0.0036699762, -0.0012558375, 0.0003523118]
>>> powers = [1, 2, 3, 5, 7]
>>> b = np.array([a * alphas**p for a, p in zip(coefs, powers)])
>>> length(b)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'length' is not defined
>>> len(b)
5
>>> b.shape
(5, 401)
>>> len(np.sum(b, 1))
5
>>> len(np.sum(b))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: object of type 'numpy.float64' has no len()
>>> len(np.sum(b, 0))
401
>>> b = np.sum([a * alphas**p for a, p in zip(coefs, powers)], 0)
>>> len(b)
401
>>> plt.plot(alphas, g(alphas, 1024, 0.001))
[<matplotlib.lines.Line2D object at 0x7f29e8536ba8>]
>>> plt.show()
>>> plt.plot(alphas, g(alphas, 1024, 0.001), label=0.001)
[<matplotlib.lines.Line2D object at 0x7f29e8662d30>]
>>> plt.plot(alphas, g(alphas, 1024, 0.01), label=0.01)
[<matplotlib.lines.Line2D object at 0x7f29e8662ac8>]
>>> plt.plot(alphas, g(alphas, 1024, 0.1), label=0.1)
[<matplotlib.lines.Line2D object at 0x7f29e8662358>]
>>> plt.legend(loc="best", title=f'$epsilon$')
<matplotlib.legend.Legend object at 0x7f29e8662668>
>>> plt.xlabel(f'$alpha$')
Text(0.5, 0, '$alpha$')
>>> plt.ylabel(f'$\check{K}$')
Text(0, 0.5, '$\\checkK$')
>>> plt.show()
Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/mathtext.py", line 2581, in parse
    result = self._expression.parseString(s)
  File "/usr/local/lib/python3.6/dist-packages/pyparsing.py", line 1955, in parseString
    raise exc
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/mathtext.py", line 2752, in unknown_symbol
    raise ParseFatalException(s, loc, "Unknown symbol: %s" % c)
pyparsing.ParseFatalException: Unknown symbol: \checkK, found '\'  (at char 0), (line:1, col:1)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/backends/backend_gtk3.py", line 304, in idle_draw
    self.draw()
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/backends/backend_gtk3agg.py", line 79, in draw
    self._render_figure(allocation.width, allocation.height)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/backends/backend_gtk3agg.py", line 24, in _render_figure
    backend_agg.FigureCanvasAgg.draw(self)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/backends/backend_agg.py", line 393, in draw
    self.figure.draw(self.renderer)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/artist.py", line 38, in draw_wrapper
    return draw(artist, renderer, *args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/figure.py", line 1736, in draw
    renderer, self, artists, self.suppressComposite)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/image.py", line 137, in _draw_list_compositing_images
    a.draw(renderer)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/artist.py", line 38, in draw_wrapper
    return draw(artist, renderer, *args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/axes/_base.py", line 2630, in draw
    mimage._draw_list_compositing_images(renderer, self, artists)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/image.py", line 137, in _draw_list_compositing_images
    a.draw(renderer)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/artist.py", line 38, in draw_wrapper
    return draw(artist, renderer, *args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/axis.py", line 1241, in draw
    self.label.draw(renderer)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/artist.py", line 38, in draw_wrapper
    return draw(artist, renderer, *args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/text.py", line 685, in draw
    bbox, info, descent = textobj._get_layout(renderer)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/text.py", line 300, in _get_layout
    clean_line, self._fontproperties, ismath=ismath)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/backends/backend_agg.py", line 209, in get_text_width_height_descent
    self.mathtext_parser.parse(s, self.dpi, prop)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/mathtext.py", line 3354, in parse
    box = self._parser.parse(s, font_output, fontsize, dpi)
  File "/usr/local/lib/python3.6/dist-packages/matplotlib/mathtext.py", line 2586, in parse
    str(err)]))
ValueError: 
\checkK
^
Unknown symbol: \checkK, found '\'  (at char 0), (line:1, col:1)
>>> plt.xlabel(r'$alpha$')
Text(0.5, 0, '$alpha$')
>>> plt.ylabel(r'$\check{K}$')
Text(0, 0.5, '$\\check{K}$')
>>> 10**-1::-3
  File "<stdin>", line 1
    10**-1::-3
           ^
SyntaxError: invalid syntax
>>> 10**-1:-3
  File "<stdin>", line 1
SyntaxError: illegal target for annotation
>>> 10**(-1:-3)
  File "<stdin>", line 1
    10**(-1:-3)
           ^
SyntaxError: invalid syntax
>>> 10**[-1:-3]
  File "<stdin>", line 1
    10**[-1:-3]
           ^
SyntaxError: invalid syntax
>>> 10**[-1:-3:-1]
  File "<stdin>", line 1
    10**[-1:-3:-1]
           ^
SyntaxError: invalid syntax
>>> -1:-1:-3
  File "<stdin>", line 1
    -1:-1:-3
         ^
SyntaxError: invalid syntax
>>> -1:-3:-1
  File "<stdin>", line 1
    -1:-3:-1
         ^
SyntaxError: invalid syntax
>>> 10**np.arange(-1, -3, -1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: Integers to negative integer powers are not allowed.
>>> 10.0**np.arange(-1, -3, -1)
array([0.1 , 0.01])
>>> 10.0**np.arange(-1, -4, -1)
array([0.1  , 0.01 , 0.001])
>>> for e in 10.0**np.arange(-1, -4, -1):
...     plt.plot(alphas, g(alphas, 1024, e), label=e)
... 
[<matplotlib.lines.Line2D object at 0x7f29e85f1da0>]
[<matplotlib.lines.Line2D object at 0x7f29e85f1c50>]
[<matplotlib.lines.Line2D object at 0x7f29e85f1d30>]
>>> plt.legend(loc="best", title=f'$epsilon$')
<matplotlib.legend.Legend object at 0x7f29e85f1860>
>>> plt.show()
>>> for e in 10.0**np.arange(-1, -4, -1):
    plt.plot(alphas, g(alphas, 1024, e), label=e)
... ... 
[<matplotlib.lines.Line2D object at 0x7f29e85bfac8>]
[<matplotlib.lines.Line2D object at 0x7f29e85bfbe0>]
[<matplotlib.lines.Line2D object at 0x7f29e85bff60>]
>>> plt.xlabel(r'$\alpha$')
Text(0.5, 0, '$\\alpha$')
>>> plt.ylabel(r'$\check\{K\}$')
Text(0, 0.5, '$\\check\\{K\\}$')
>>> plt.show()
>>> g(-2, 1024, e)
6.250222448095048
>>> b(0, 1024, 0.01)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: b() takes 1 positional argument but 3 were given
>>> b(-2)
-0.0011205894000000036
>>> b(-2.1)
-0.012808767503233391
>>> b(-1.99)
-0.00014937999682618913
>>> len(B)
10
>>> sum([B[i] * alpha**i for i in powers])
B_{\alpha\,1}*alpha + B_{\alpha\,2}*alpha**2 + B_{\alpha\,3}*alpha**3 + B_{\alpha\,5}*alpha**5 + B_{\alpha\,7}*alpha**7
>>> sp.factor(sum([B[i] * alpha**i for i in powers]))
alpha*(B_{\alpha\,1} + B_{\alpha\,2}*alpha + B_{\alpha\,3}*alpha**2 + B_{\alpha\,5}*alpha**4 + B_{\alpha\,7}*alpha**6)
>>> sp.expand(sum([B[i] * alpha**i for i in powers]))
B_{\alpha\,1}*alpha + B_{\alpha\,2}*alpha**2 + B_{\alpha\,3}*alpha**3 + B_{\alpha\,5}*alpha**5 + B_{\alpha\,7}*alpha**7
>>> alpha
alpha
>>> befit = sum([B[i] * alpha**i for i in powers])
>>> befit
B_{\alpha\,1}*alpha + B_{\alpha\,2}*alpha**2 + B_{\alpha\,3}*alpha**3 + B_{\alpha\,5}*alpha**5 + B_{\alpha\,7}*alpha**7
>>> befit / alpha
(B_{\alpha\,1}*alpha + B_{\alpha\,2}*alpha**2 + B_{\alpha\,3}*alpha**3 + B_{\alpha\,5}*alpha**5 + B_{\alpha\,7}*alpha**7)/alpha
>>> sp.simplify(befit / alpha)
B_{\alpha\,1} + B_{\alpha\,2}*alpha + B_{\alpha\,3}*alpha**2 + B_{\alpha\,5}*alpha**4 + B_{\alpha\,7}*alpha**6
>>> sp.simplify(befit / (alpha + 2))
alpha*(B_{\alpha\,1} + B_{\alpha\,2}*alpha + B_{\alpha\,3}*alpha**2 + B_{\alpha\,5}*alpha**4 + B_{\alpha\,7}*alpha**6)/(alpha + 2)
>>> sp.simplify(befit / (alpha**2 + 2*alpha))
(B_{\alpha\,1} + B_{\alpha\,2}*alpha + B_{\alpha\,3}*alpha**2 + B_{\alpha\,5}*alpha**4 + B_{\alpha\,7}*alpha**6)/(alpha + 2)
>>> powers
[1, 2, 3, 5, 7]
>>> coefs
[-0.0304985687, -0.0069622017, 0.0036699762, -0.0012558375, 0.0003523118]
>>> tmp = b(alphas)
>>> plot(alphas[1:], np.diff(tmp))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'plot' is not defined
>>> plt.plot(alphas[1:], np.diff(tmp)); plt.show()
[<matplotlib.lines.Line2D object at 0x7f2a114a0518>]
>>> plt.plot(alphas[1:], np.diff(tmp)); plt.grid(); plt.show()
[<matplotlib.lines.Line2D object at 0x7f29e991cb38>]
>>> plt.plot(alphas, tmp)
[<matplotlib.lines.Line2D object at 0x7f29e8421588>]
>>> plt.plot(alphas[1:], np.diff(tmp))
[<matplotlib.lines.Line2D object at 0x7f29e83e6240>]
>>> plt.plot(alphas[2:], np.diff(np.diff(tmp)))
[<matplotlib.lines.Line2D object at 0x7f29e84219e8>]
>>> plt.grid()
>>> plt.show()
>>> plt.plot(alphas, tmp)
[<matplotlib.lines.Line2D object at 0x7f29e8336518>]
>>> plt.plot(alphas[1:], np.diff(tmp)/np.diff(alphas[:2]))
[<matplotlib.lines.Line2D object at 0x7f29e8412c88>]
>>> plt.plot(alphas[2:], np.diff(np.diff(tmp))/np.diff(alphas[:2])**2)
[<matplotlib.lines.Line2D object at 0x7f29e8336978>]
>>> plt.show()
>>> plt.hlines(0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: hlines() missing 2 required positional arguments: 'xmin' and 'xmax'
>>> plt.hlines(0, -2, 2)
<matplotlib.collections.LineCollection object at 0x7f29e831cc50>
>>> plt.show()
>>> plt.grid()
>>> plt.hlines(0, -2, 2)
<matplotlib.collections.LineCollection object at 0x7f29e8288208>
>>> plt.plot(alphas, tmp)
[<matplotlib.lines.Line2D object at 0x7f29e8288198>]
>>> plt.plot(alphas[1:], np.diff(tmp)/np.diff(alphas[:2]))
[<matplotlib.lines.Line2D object at 0x7f29e82884a8>]
>>> plt.plot(alphas[2:], np.diff(np.diff(tmp))/np.diff(alphas[:2])**2)
[<matplotlib.lines.Line2D object at 0x7f29e8288908>]
>>> plt.show()
>>> da
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'da' is not defined
>>> da = np.diff(alphas[:2])
>>> da
array([0.01])
>>> da = 0.01
>>> for i in range(5):
...     plt.plot(alphas[i:], np.diff(tmp, i)/da**i)
... 
[<matplotlib.lines.Line2D object at 0x7f29e8259128>]
[<matplotlib.lines.Line2D object at 0x7f29e8259278>]
[<matplotlib.lines.Line2D object at 0x7f29e824a6d8>]
[<matplotlib.lines.Line2D object at 0x7f29e8252eb8>]
[<matplotlib.lines.Line2D object at 0x7f29e8288240>]
>>> plt.show()
>>> plt.grid()
>>> plt.hlines(0, -2, 2)
<matplotlib.collections.LineCollection object at 0x7f29e83dcf60>
>>> for i in range(5):
    plt.plot(alphas[i:], np.diff(tmp, i)/da**i)
... ... 
[<matplotlib.lines.Line2D object at 0x7f29e83dc5c0>]
[<matplotlib.lines.Line2D object at 0x7f29e83dc390>]
[<matplotlib.lines.Line2D object at 0x7f29e83dc470>]
[<matplotlib.lines.Line2D object at 0x7f29e9938da0>]
[<matplotlib.lines.Line2D object at 0x7f29e9938518>]
>>> plt.ylim([-0.1, 0.1])
(-0.1, 0.1)
>>> plt.show()
>>> plt.grid()
>>> plt.hlines(0, -2, 2)
<matplotlib.collections.LineCollection object at 0x7f2a114f11d0>
>>> for i in range(5):
    plt.plot(alphas[i:]+i*da/2, np.diff(tmp, i)/da**i)
... ... 
[<matplotlib.lines.Line2D object at 0x7f2a114f1400>]
[<matplotlib.lines.Line2D object at 0x7f2a114f1780>]
[<matplotlib.lines.Line2D object at 0x7f2a114f1ef0>]
[<matplotlib.lines.Line2D object at 0x7f2a114f1048>]
[<matplotlib.lines.Line2D object at 0x7f29e8656400>]
>>> plt.ylim([-0.1, 0.1])
(-0.1, 0.1)
>>> plt.show()
>>> plt.grid()
>>> plt.hlines(0, -2, 2)
<matplotlib.collections.LineCollection object at 0x7f29e8265c50>
>>> for i in range(5):
    plt.plot(alphas[i:]+i*da/2, np.diff(tmp, i)/da**i)
... ...   C-c C-c
KeyboardInterrupt
>>> for i in range(5):
    plt.plot(alphas[i:]-i*da/2, np.diff(tmp, i)/da**i)
... ... 
[<matplotlib.lines.Line2D object at 0x7f29e8265be0>]
[<matplotlib.lines.Line2D object at 0x7f29e8265fd0>]
[<matplotlib.lines.Line2D object at 0x7f29e8226390>]
[<matplotlib.lines.Line2D object at 0x7f29e8226710>]
[<matplotlib.lines.Line2D object at 0x7f29e8226a90>]
>>> plt.ylim([-0.1, 0.1])
(-0.1, 0.1)
>>> plt.show()
>>> foo = (alpha - B[0]) * alpha * (alpha + B[0])
>>> plot(alphas, sp.lambdify((alpha, B[0]), foo)(alphas, 5/3))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'plot' is not defined
>>> plt.plot(alphas, sp.lambdify((alpha, B[0]), foo)(alphas, 5/3))
[<matplotlib.lines.Line2D object at 0x7f29e84d8208>]
>>> plt.show()
>>> bar = sp.integrate(foo, alpha)
>>> bar
-B_{\alpha\,0}**2*alpha**2/2 + alpha**4/4
>>> plt.plot(alphas, sp.lambdify((alpha, B[0]), bar)(alphas, 5/3))
[<matplotlib.lines.Line2D object at 0x7f29e8686ef0>]
>>> plt.show()
>>> bar = sp.integrate(foo, alpha) + B[1]
>>> plt.plot(alphas, sp.lambdify((alpha, B[0], B[1]), bar)(alphas, 5/3, 5))
[<matplotlib.lines.Line2D object at 0x7f29e820c978>]
>>> plt.show()
>>> sp.simplify(bar)
-B_{\alpha\,0}**2*alpha**2/2 + B_{\alpha\,1} + alpha**4/4
>>> \frac{\sqrt{N} b_{\alpha\,3} \log{\left(N \right)}}{\sqrt{N} \epsilon + b_{\alpha\,3} \log{\left(N \right)}}
>>> plt.show()
>>> Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/ben/Source/Open/powerlawnoise/pypowerlawnoise/algebra.py", line 46, in <module>
    befit = sp.integrate(one)
  File "/usr/local/lib/python3.6/dist-packages/sympy/integrals/integrals.py", line 1542, in integrate
    integral = Integral(*args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/sympy/integrals/integrals.py", line 87, in __new__
    obj = AddWithLimits.__new__(cls, function, *symbols, **assumptions)
  File "/usr/local/lib/python3.6/dist-packages/sympy/concrete/expr_with_limits.py", line 494, in __new__
    pre = _common_new(cls, function, *symbols, **assumptions)
  File "/usr/local/lib/python3.6/dist-packages/sympy/concrete/expr_with_limits.py", line 58, in _common_new
    "specify dummy variables for %s" % function)
ValueError: specify dummy variables for alpha**6/120 - alpha**4*b_{\alpha\,4}**2/24 + alpha**2*b_{\alpha\,3}/2 + alpha*b_{\alpha\,2} + b_{\alpha\,1}
>>> sp.simplify(befit)
alpha*(alpha**6 - 7*alpha**4*b_{\alpha\,4}**2 + 140*alpha**2*b_{\alpha\,3} + 420*alpha*b_{\alpha\,2} + 840*b_{\alpha\,1})/840
>>> sp.expand(befit)
alpha**7/840 - alpha**5*b_{\alpha\,4}**2/120 + alpha**3*b_{\alpha\,3}/6 + alpha**2*b_{\alpha\,2}/2 + alpha*b_{\alpha\,1}
>>> sp.solve(befit.subs(alpha, -2), b[1])
[b_{\alpha\,2} - 2*b_{\alpha\,3}/3 + 2*b_{\alpha\,4}**2/15 - 8/105]
>>> sp.solve(befit.subs(alpha, -2), b[4])
[-sqrt(1470*b_{\alpha\,1} - 1470*b_{\alpha\,2} + 980*b_{\alpha\,3} + 112)/14, sqrt(1470*b_{\alpha\,1} - 1470*b_{\alpha\,2} + 980*b_{\alpha\,3} + 112)/14]
>>> sp.solve(befit.subs(alpha, -2), b[3])
[-3*b_{\alpha\,1}/2 + 3*b_{\alpha\,2}/2 + b_{\alpha\,4}**2/5 - 4/35]
>>> sp.solve(befit.subs(alpha, -2), b[2])
[b_{\alpha\,1} + 2*b_{\alpha\,3}/3 - 2*b_{\alpha\,4}**2/15 + 8/105]
>>> bee_one = sp.solve(befit.subs(alpha, -2), b[1])[0]
>>> sp.simplify(befit.subs(b[1], bee_one))
alpha*(alpha**6 - 7*alpha**4*b_{\alpha\,4}**2 + 140*alpha**2*b_{\alpha\,3} + 420*alpha*b_{\alpha\,2} + 840*b_{\alpha\,2} - 560*b_{\alpha\,3} + 112*b_{\alpha\,4}**2 - 64)/840
>>> maybe_better = sp.simplify(befit.subs(b[1], bee_one))
>>> sp.solve(maybe_better.subs(alpha, 0), b[4])
[]
>>> sp.solve(maybe_better.subs(alpha, 0), b[3])
[]
>>> sp.solve(maybe_better.subs(alpha, 0), b[2])
[]
>>> sp.solve(maybe_better.subs(alpha, 0), b[1])
[]
>>> maybe_better.subs(alpha, 0)
0
>>> befit
alpha**7/840 - alpha**5*b_{\alpha\,4}**2/120 + alpha**3*b_{\alpha\,3}/6 + alpha**2*b_{\alpha\,2}/2 + alpha*b_{\alpha\,1}
>>> bee_four = sp.solve(befit.subs(alpha, -2), b[4])
>>> bee_four
[-sqrt(1470*b_{\alpha\,1} - 1470*b_{\alpha\,2} + 980*b_{\alpha\,3} + 112)/14, sqrt(1470*b_{\alpha\,1} - 1470*b_{\alpha\,2} + 980*b_{\alpha\,3} + 112)/14]
>>> len(bee_four)
2
>>> befit.subs(b[4], bee_four[0])
alpha**7/840 - alpha**5*(15*b_{\alpha\,1}/2 - 15*b_{\alpha\,2}/2 + 5*b_{\alpha\,3} + 4/7)/120 + alpha**3*b_{\alpha\,3}/6 + alpha**2*b_{\alpha\,2}/2 + alpha*b_{\alpha\,1}
>>> befit.subs(b[4], bee_four[0]).simplify()
alpha*(2*alpha**6 + alpha**4*(-105*b_{\alpha\,1} + 105*b_{\alpha\,2} - 70*b_{\alpha\,3} - 8) + 280*alpha**2*b_{\alpha\,3} + 840*alpha*b_{\alpha\,2} + 1680*b_{\alpha\,1})/1680
>>> befit.subs(b[4], bee_four[1]).simplify()
alpha*(2*alpha**6 + alpha**4*(-105*b_{\alpha\,1} + 105*b_{\alpha\,2} - 70*b_{\alpha\,3} - 8) + 280*alpha**2*b_{\alpha\,3} + 840*alpha*b_{\alpha\,2} + 1680*b_{\alpha\,1})/1680
>>> maybe_better
alpha*(alpha**6 - 7*alpha**4*b_{\alpha\,4}**2 + 140*alpha**2*b_{\alpha\,3} + 420*alpha*b_{\alpha\,2} + 840*b_{\alpha\,2} - 560*b_{\alpha\,3} + 112*b_{\alpha\,4}**2 - 64)/840
>>> plt.show()
>>> 
>>> befit
0.00119047619047619*alpha**7 - 0.00416666666666667*alpha**5 + 0.166666666666667*alpha**3*b_{\alpha\,3} + 0.5*alpha**2*b_{\alpha\,2} + 1.0*alpha*b_{\alpha\,1}
>>> befit.simplify()
alpha*(0.00119047619047619*alpha**6 - 0.00416666666666667*alpha**4 + 0.166666666666667*alpha**2*b_{\alpha\,3} + 0.5*alpha*b_{\alpha\,2} + 1.0*b_{\alpha\,1})
>>> befit.simplify()
alpha*(alpha**6/840 - alpha**4/240 + alpha**2*b_{\alpha\,3}/6 + alpha*b_{\alpha\,2}/2 + b_{\alpha\,1})
>>> bee_one = sp.solve(befit.subs(alpha, -2), b[1])[0]
>>> bee_one
b_{\alpha\,2} - 2*b_{\alpha\,3}/3 - 1/105
>>> bee_two = sp.solve(befit.subs(alpha, -2), b[1])
>>> bee_two
[b_{\alpha\,2} - 2*b_{\alpha\,3}/3 - 1/105]
>>> bee_two = sp.solve(befit.subs(alpha, -2), b[2])
>>> bee_two
[b_{\alpha\,1} + 2*b_{\alpha\,3}/3 + 1/105]
>>> bee_three = sp.solve(befit.subs(alpha, -2), b[3])
>>> bee_three
[-3*b_{\alpha\,1}/2 + 3*b_{\alpha\,2}/2 - 1/70]
>>> bee_one = sp.solve(befit.subs(alpha, -2), b[1])[0]
>>> bee_two = sp.solve(befit.subs(alpha, -2), b[2])[0]
>>> bee_three = sp.solve(befit.subs(alpha, -2), b[3])[0]
>>> bee_two - b[1])
  File "<stdin>", line 1
    bee_two - b[1])
                  ^
SyntaxError: invalid syntax
>>> bee_two - b[1]
2*b_{\alpha\,3}/3 + 1/105
>>> befit.subs(b[3], bee_three)
alpha**7/840 - alpha**5/240 + alpha**3*(-3*b_{\alpha\,1}/2 + 3*b_{\alpha\,2}/2 - 1/70)/6 + alpha**2*b_{\alpha\,2}/2 + alpha*b_{\alpha\,1}
>>> befit.subs(b[3], bee_three).simplify()
alpha*(alpha**6/840 - alpha**4/240 + alpha**2*(-105*b_{\alpha\,1} + 105*b_{\alpha\,2} - 1)/420 + alpha*b_{\alpha\,2}/2 + b_{\alpha\,1})
>>> alpha*(alpha**6 - 7*alpha**4*b_{\alpha\,4}**2 + alpha**2*(-210*b_{\alpha\,1} + 210*b_{\alpha\,2} + 28*b_{\alpha\,4}**2 - 16) + 420*alpha*b_{\alpha\,2} + 840*b_{\alpha\,1})/840
>>> maybe_better
alpha*(alpha**6 - 7*alpha**4*b_{\alpha\,4}**2 + alpha**2*(-210*b_{\alpha\,1} + 210*b_{\alpha\,2} + 28*b_{\alpha\,4}**2 - 16) + 420*alpha*b_{\alpha\,2} + 840*b_{\alpha\,1})/840
>>> alpha*(alpha**6 - 7*alpha**4*b_{\alpha\,4}**2 + 140*alpha**2*b_{\alpha\,3} + 4*alpha*(105*b_{\alpha\,1} + 70*b_{\alpha\,3} - 14*b_{\alpha\,4}**2 + 8) + 840*b_{\alpha\,1})/840
>>> alpha*(alpha**6/840 + alpha**5*(-b_{\alpha\,4} + b_{\alpha\,5})/360 - alpha**4*b_{\alpha\,4}*b_{\alpha\,5}/120 + alpha**2*b_{\alpha\,3}/6 + alpha*(315*b_{\alpha\,1} + 210*b_{\alpha\,3} - 42*b_{\alpha\,4}*b_{\alpha\,5} + 28*b_{\alpha\,4} - 28*b_{\alpha\,5} + 24)/630 + b_{\alpha\,1})
>>> alpha*(alpha**6 - 7*alpha**4*b_{\alpha\,4}**2 + 140*alpha**2*b_{\alpha\,3} + 4*alpha*(105*b_{\alpha\,1} + 70*b_{\alpha\,3} - 14*b_{\alpha\,4}**2 + 8) + 840*b_{\alpha\,1})/840
>>> tmp = sum([b[i]*alpha**i for i in powers])
>>> tmp
alpha**7*b_{\alpha\,7} + alpha**5*b_{\alpha\,5} + alpha**3*b_{\alpha\,3} + alpha**2*b_{\alpha\,2} + alpha*b_{\alpha\,1}
>>> foo = sp.solve(tmp.subs(alpha, -2), b[1])
>>> foo
[2*b_{\alpha\,2} - 4*b_{\alpha\,3} - 16*b_{\alpha\,5} - 64*b_{\alpha\,7}]
>>> sp.solve(tmp.subs(alpha, -2), b[7])
[-b_{\alpha\,1}/64 + b_{\alpha\,2}/32 - b_{\alpha\,3}/16 - b_{\alpha\,5}/4]
>>> foo = tmp.subs(b[7], sp.solve(tmp.subs(alpha, -2), b[7])[0])
>>> foo.simplify()
alpha*(alpha**6*(-b_{\alpha\,1} + 2*b_{\alpha\,2} - 4*b_{\alpha\,3} - 16*b_{\alpha\,5})/64 + alpha**4*b_{\alpha\,5} + alpha**2*b_{\alpha\,3} + alpha*b_{\alpha\,2} + b_{\alpha\,1})
>>> foo.subs(b[5], 0).expand()
-alpha**7*b_{\alpha\,1}/64 + alpha**7*b_{\alpha\,2}/32 - alpha**7*b_{\alpha\,3}/16 + alpha**3*b_{\alpha\,3} + alpha**2*b_{\alpha\,2} + alpha*b_{\alpha\,1}
>>> foo.subs(b[5], 0).simplify()
alpha*(alpha**6*(-b_{\alpha\,1} + 2*b_{\alpha\,2} - 4*b_{\alpha\,3})/64 + alpha**2*b_{\alpha\,3} + alpha*b_{\alpha\,2} + b_{\alpha\,1})
>>> foo.subs(b[5], 0).simplify().factor()
-alpha*(alpha + 2)*(alpha**5*b_{\alpha\,1} - 2*alpha**5*b_{\alpha\,2} + 4*alpha**5*b_{\alpha\,3} - 2*alpha**4*b_{\alpha\,1} + 4*alpha**4*b_{\alpha\,2} - 8*alpha**4*b_{\alpha\,3} + 4*alpha**3*b_{\alpha\,1} - 8*alpha**3*b_{\alpha\,2} + 16*alpha**3*b_{\alpha\,3} - 8*alpha**2*b_{\alpha\,1} + 16*alpha**2*b_{\alpha\,2} - 32*alpha**2*b_{\alpha\,3} + 16*alpha*b_{\alpha\,1} - 32*alpha*b_{\alpha\,2} - 32*b_{\alpha\,1})/64
>>> maybe_better.factor()
alpha*(alpha + 2)*(alpha**5 - 2*alpha**4 - 7*alpha**3*b_{\alpha\,4}**2 + 4*alpha**3 + 14*alpha**2*b_{\alpha\,4}**2 - 8*alpha**2 + 140*alpha*b_{\alpha\,3} - 28*alpha*b_{\alpha\,4}**2 + 16*alpha + 420*b_{\alpha\,1})/840
>>> tmp.subs(alpha, 0)
0
>>> tmp.subs(alpha, -2)
-2*b_{\alpha\,1} + 4*b_{\alpha\,2} - 8*b_{\alpha\,3} - 32*b_{\alpha\,5} - 128*b_{\alpha\,7}
>>> tmp.subs(alpha, -2).solve(b[1])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Add' object has no attribute 'solve'
>>> sp.solve(tmp.subs(alpha, -2), b[1])
[2*b_{\alpha\,2} - 4*b_{\alpha\,3} - 16*b_{\alpha\,5} - 64*b_{\alpha\,7}]
>>> bar = tmp.subs(b[1], sp.solve(tmp.subs(alpha, -2), b[1])[0])
>>> bar.simplify()
alpha*(alpha**6*b_{\alpha\,7} + alpha**4*b_{\alpha\,5} + alpha**2*b_{\alpha\,3} + alpha*b_{\alpha\,2} + 2*b_{\alpha\,2} - 4*b_{\alpha\,3} - 16*b_{\alpha\,5} - 64*b_{\alpha\,7})
>>> bar.factor()
alpha*(alpha + 2)*(alpha**5*b_{\alpha\,7} - 2*alpha**4*b_{\alpha\,7} + alpha**3*b_{\alpha\,5} + 4*alpha**3*b_{\alpha\,7} - 2*alpha**2*b_{\alpha\,5} - 8*alpha**2*b_{\alpha\,7} + alpha*b_{\alpha\,3} + 4*alpha*b_{\alpha\,5} + 16*alpha*b_{\alpha\,7} + b_{\alpha\,2} - 2*b_{\alpha\,3} - 8*b_{\alpha\,5} - 32*b_{\alpha\,7})
>>> factor(alpha**6-64)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'factor' is not defined
>>> sp.factor(alpha**6-64)
(alpha - 2)*(alpha + 2)*(alpha**2 - 2*alpha + 4)*(alpha**2 + 2*alpha + 4)
>>> bar.expand()
alpha**7*b_{\alpha\,7} + alpha**5*b_{\alpha\,5} + alpha**3*b_{\alpha\,3} + alpha**2*b_{\alpha\,2} + 2*alpha*b_{\alpha\,2} - 4*alpha*b_{\alpha\,3} - 16*alpha*b_{\alpha\,5} - 64*alpha*b_{\alpha\,7}
>>> bar/(alpha+2)
(alpha**7*b_{\alpha\,7} + alpha**5*b_{\alpha\,5} + alpha**3*b_{\alpha\,3} + alpha**2*b_{\alpha\,2} + alpha*(2*b_{\alpha\,2} - 4*b_{\alpha\,3} - 16*b_{\alpha\,5} - 64*b_{\alpha\,7}))/(alpha + 2)
>>> bar/(alpha+2)/alpha
(alpha**7*b_{\alpha\,7} + alpha**5*b_{\alpha\,5} + alpha**3*b_{\alpha\,3} + alpha**2*b_{\alpha\,2} + alpha*(2*b_{\alpha\,2} - 4*b_{\alpha\,3} - 16*b_{\alpha\,5} - 64*b_{\alpha\,7}))/(alpha*(alpha + 2))
>>> (bar/(alpha+2)/alpha).simplify
<bound method Basic.simplify of (alpha**7*b_{\alpha\,7} + alpha**5*b_{\alpha\,5} + alpha**3*b_{\alpha\,3} + alpha**2*b_{\alpha\,2} + alpha*(2*b_{\alpha\,2} - 4*b_{\alpha\,3} - 16*b_{\alpha\,5} - 64*b_{\alpha\,7}))/(alpha*(alpha + 2))>
>>> (bar/(alpha+2)/alpha).simplify()
(alpha**6*b_{\alpha\,7} + alpha**4*b_{\alpha\,5} + alpha**2*b_{\alpha\,3} + alpha*b_{\alpha\,2} + 2*b_{\alpha\,2} - 4*b_{\alpha\,3} - 16*b_{\alpha\,5} - 64*b_{\alpha\,7})/(alpha + 2)
>>> (bar.factor()/(alpha+2)/alpha).simplify()
alpha**5*b_{\alpha\,7} - 2*alpha**4*b_{\alpha\,7} + alpha**3*b_{\alpha\,5} + 4*alpha**3*b_{\alpha\,7} - 2*alpha**2*b_{\alpha\,5} - 8*alpha**2*b_{\alpha\,7} + alpha*b_{\alpha\,3} + 4*alpha*b_{\alpha\,5} + 16*alpha*b_{\alpha\,7} + b_{\alpha\,2} - 2*b_{\alpha\,3} - 8*b_{\alpha\,5} - 32*b_{\alpha\,7}
>>> bar
alpha**7*b_{\alpha\,7} + alpha**5*b_{\alpha\,5} + alpha**3*b_{\alpha\,3} + alpha**2*b_{\alpha\,2} + alpha*(2*b_{\alpha\,2} - 4*b_{\alpha\,3} - 16*b_{\alpha\,5} - 64*b_{\alpha\,7})
>>> bar / (alpha-2)
(alpha**7*b_{\alpha\,7} + alpha**5*b_{\alpha\,5} + alpha**3*b_{\alpha\,3} + alpha**2*b_{\alpha\,2} + alpha*(2*b_{\alpha\,2} - 4*b_{\alpha\,3} - 16*b_{\alpha\,5} - 64*b_{\alpha\,7}))/(alpha - 2)
>>> bar / (alpha+2)
(alpha**7*b_{\alpha\,7} + alpha**5*b_{\alpha\,5} + alpha**3*b_{\alpha\,3} + alpha**2*b_{\alpha\,2} + alpha*(2*b_{\alpha\,2} - 4*b_{\alpha\,3} - 16*b_{\alpha\,5} - 64*b_{\alpha\,7}))/(alpha + 2)
>>> (alpha**6 - (-2)**6)
alpha**6 - 64
>>> (alpha**6 - (-2)**6)/(alpha-2)
(alpha**6 - 64)/(alpha - 2)
>>> (alpha**6 - (-2)**6)/(alpha+2)
(alpha**6 - 64)/(alpha + 2)
>>> sp.simplify((alpha**6 - (-2)**6)/(alpha+2))
(alpha**6 - 64)/(alpha + 2)
>>> sp.simplify((alpha**6 - (-2)**6)/(alpha-2))
(alpha**6 - 64)/(alpha - 2)
>>> [sp.simplify(sp.factor(alpha**i - (-2)**i) / (alpha + 2)) for i in (1, 2, 4, 6)]
[1, alpha - 2, (alpha - 2)*(alpha**2 + 4), (alpha - 2)*(alpha**2 - 2*alpha + 4)*(alpha**2 + 2*alpha + 4)]
>>> sp.factor(alpha**2 - 2*alpha + 4)
alpha**2 - 2*alpha + 4
>>> i
4
>>> j
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'j' is not defined
>>> 3j
3j
>>> 3i
  File "<stdin>", line 1
    3i
     ^
SyntaxError: invalid syntax
>>> 3i**2
  File "<stdin>", line 1
    3i**2
     ^
SyntaxError: invalid syntax
>>> 3j**2
(-9+0j)
>>> (alpha + 2j)*(alpha - 2j)
(alpha - 2.0*I)*(alpha + 2.0*I)
>>> sp.simplify((alpha-2*I)*(alpha+2*I))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'I' is not defined
>>> sp.simplify((alpha-2*sp.I)*(alpha+2*sp.I))
alpha**2 + 4
>>> sp.factor((alpha**2 - 2*alpha + 4)/(alpha-2*sp.I))
-(alpha**2 - 2*alpha + 4)/(-alpha + 2*I)
>>> sp.factor((alpha**2 - 2*alpha + 4)/(alpha+2*sp.I))
(alpha**2 - 2*alpha + 4)/(alpha + 2*I)
>>> sp.factor((alpha**2 + 2*alpha + 4)
... )
alpha**2 + 2*alpha + 4
>>> (alpha**2 + 2*alpha + 4)/(alpha+2*sp.I)
(alpha**2 + 2*alpha + 4)/(alpha + 2*I)
>>> sp.expand((alpha**2 + 2*alpha + 4)/(alpha+2*sp.I))
alpha**2/(alpha + 2*I) + 2*alpha/(alpha + 2*I) + 4/(alpha + 2*I)
>>> sp.factor((alpha**2 + 2*alpha + 4)/(alpha+2*sp.I))
(alpha**2 + 2*alpha + 4)/(alpha + 2*I)
>>> sp.solve(alpha**2 + 2*alpha + 4)
[-1 - sqrt(3)*I, -1 + sqrt(3)*I]
>>> sqrt(3)/2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'sqrt' is not defined
>>> np.sqrt(3)/2
0.8660254037844386
>>> (np.sqrt(2)+np.sqrt(3))/2
1.5731321849709863
>>> (np.sqrt(2)+np.sqrt(3))/4
0.7865660924854931
>>> np.sqrt(5)/2
1.118033988749895
>>> [(b[i], bc) for i, bc in enumerate(bee_coefs)]
[(b_{\alpha\,0}, -0.0069256872), (b_{\alpha\,1}, 0.0029379664), (b_{\alpha\,2}, -0.0007799595), (b_{\alpha\,3}, 0.0002673209)]
>>> [(b[i], bc) for i, bc in zip([2, 3, 5, 7], bee_coefs)]
[(b_{\alpha\,2}, -0.0069256872), (b_{\alpha\,3}, 0.0029379664), (b_{\alpha\,5}, -0.0007799595), (b_{\alpha\,7}, 0.0002673209)]
>>> f = sp.lambdify(alpha, def_better.subs([(b[i], bc) for i, bc in zip([2, 3, 5, 7], bee_coefs)]))
>>> plt.plot(alphas, f(alphas))
[<matplotlib.lines.Line2D object at 0x7f29e9965cc0>]
>>> plt.show()
>>> \frac{\sqrt{N} b_{\alpha\,3} \log{\left(N \right)}}{\sqrt{N} \epsilon + b_{\alpha\,3} \log{\left(N \right)}}
>>> plt.show()
>>> \frac{\sqrt{N} b_{\alpha\,3} \log{\left(N \right)}}{\sqrt{N} \epsilon + b_{\alpha\,3} \log{\left(N \right)}}
>>> 

Process Python finished
Python 3.6.9 (default, Jul 17 2020, 12:50:27) 
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> python.el: native completion setup loaded
>>> Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/ben/Source/Open/powerlawnoise/pypowerlawnoise/algebra.py", line 12, in <module>
    answer = sp.simplify(sp.solve(sp.Eq(M, epsilon), K_check)[0])
IndexError: list index out of range
>>> M
-B*log(N)/sqrt(N) + B*log(N)/K
>>> 

Process Python finished
Python 3.6.9 (default, Jul 17 2020, 12:50:27) 
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> python.el: native completion setup loaded
>>> \frac{B \sqrt{N} \log{\left(N \right)}}{B \log{\left(N \right)} + \sqrt{N} \epsilon}
>>> plt.show(
... )
>>> \frac{B \sqrt{N} \log{\left(N \right)}}{B \log{\left(N \right)} + \sqrt{N} \epsilon}
>>> plt.plot(alphas, kay_check(alphas, 1024, 0.001)/sqrt(1024))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'sqrt' is not defined
>>> for n in 2**np.arange(6, 13):
...     plt.plot(alphas, kay_check(alphas, n, 0.001)/np.sqrt(n), label=n)
... 
[<matplotlib.lines.Line2D object at 0x7fe732f0c0b8>]
[<matplotlib.lines.Line2D object at 0x7fe732f0c1d0>]
[<matplotlib.lines.Line2D object at 0x7fe732f0c550>]
[<matplotlib.lines.Line2D object at 0x7fe732f0c8d0>]
[<matplotlib.lines.Line2D object at 0x7fe732f0cc50>]
[<matplotlib.lines.Line2D object at 0x7fe732f0cfd0>]
[<matplotlib.lines.Line2D object at 0x7fe732f18390>]
>>> plt.show()
>>> for n in 2**np.arange(6, 13):
...     plt.plot(alphas, kay_check(alphas, n, 0.001)/np.sqrt(n), label=n)
... 
[<matplotlib.lines.Line2D object at 0x7fe72fde0ac8>]
[<matplotlib.lines.Line2D object at 0x7fe72fde0be0>]
[<matplotlib.lines.Line2D object at 0x7fe72fde0f60>]
[<matplotlib.lines.Line2D object at 0x7fe72fdef320>]
[<matplotlib.lines.Line2D object at 0x7fe72fdef6a0>]
[<matplotlib.lines.Line2D object at 0x7fe72fdefa20>]
[<matplotlib.lines.Line2D object at 0x7fe72fdefda0>]
>>> plt.ylim=c(-2, 2)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'c' is not defined
>>> plt.ylim=(-2, 2)
>>> del plt.ylim
>>> plt.ylim(-2 ,2)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: module 'matplotlib.pyplot' has no attribute 'ylim'
>>> 

Process Python finished
Python 3.6.9 (default, Jul 17 2020, 12:50:27) 
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> python.el: native completion setup loaded
>>> Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/ben/Source/Open/powerlawnoise/pypowerlawnoise/algebra.py", line 18, in <module>
    print(sp.latex(answer))
NameError: name 'answer' is not defined
>>> \frac{B \sqrt{N} \log{\left(N \right)}}{B \log{\left(N \right)} + \sqrt{N} \epsilon}
>>> for n in 2**np.arange(6, 13):
...     plt.plot(alphas, kay_check(alphas, n, 0.001)/np.sqrt(n), label=n)
... 
[<matplotlib.lines.Line2D object at 0x7f4118050128>]
[<matplotlib.lines.Line2D object at 0x7f4118050240>]
[<matplotlib.lines.Line2D object at 0x7f41180505c0>]
[<matplotlib.lines.Line2D object at 0x7f4118050940>]
[<matplotlib.lines.Line2D object at 0x7f4118050cc0>]
[<matplotlib.lines.Line2D object at 0x7f411805c080>]
[<matplotlib.lines.Line2D object at 0x7f411805c400>]
>>> plt.ylim(-2, 2)
(-2.0, 2.0)
>>> plt.show()
>>> for n in 2**np.arange(6, 13):
...     plt.plot(alphas, kay_check(alphas, n, 0.001)/np.sqrt(n), label=n)
... 
[<matplotlib.lines.Line2D object at 0x7f411174d400>]
[<matplotlib.lines.Line2D object at 0x7f411174d518>]
[<matplotlib.lines.Line2D object at 0x7f411174d898>]
[<matplotlib.lines.Line2D object at 0x7f411174dc18>]
[<matplotlib.lines.Line2D object at 0x7f411174df98>]
[<matplotlib.lines.Line2D object at 0x7f4111759358>]
[<matplotlib.lines.Line2D object at 0x7f41117596d8>]
>>> plt.ylim(-2, 2)
(-2.0, 2.0)
>>> plt.show()
>>> for n in 2**np.arange(6, 13):
...     plt.plot(alphas, kay_check(alphas, n, 0.001)/np.sqrt(n), label=n)
... 
[<matplotlib.lines.Line2D object at 0x7f41116ce5c0>]
[<matplotlib.lines.Line2D object at 0x7f41116ce6d8>]
[<matplotlib.lines.Line2D object at 0x7f41116cea58>]
[<matplotlib.lines.Line2D object at 0x7f41116cedd8>]
[<matplotlib.lines.Line2D object at 0x7f41116db198>]
[<matplotlib.lines.Line2D object at 0x7f41116db518>]
[<matplotlib.lines.Line2D object at 0x7f41116db898>]
>>> plt.legend(loc="best')
  File "<stdin>", line 1
    plt.legend(loc="best')
                         ^
SyntaxError: EOL while scanning string literal
>>> plt.legend(loc='best')
<matplotlib.legend.Legend object at 0x7f41117124a8>
>>> plt.show()
>>> for n in 2**np.arange(6, 13):
...     plt.plot(alphas, kay_check(alphas, n, 0.001)/np.log(n), label=n)
... 
[<matplotlib.lines.Line2D object at 0x7f41115f1518>]
[<matplotlib.lines.Line2D object at 0x7f41115f1630>]
[<matplotlib.lines.Line2D object at 0x7f41115f19b0>]
[<matplotlib.lines.Line2D object at 0x7f41115f1d30>]
[<matplotlib.lines.Line2D object at 0x7f41116000f0>]
[<matplotlib.lines.Line2D object at 0x7f4111600470>]
[<matplotlib.lines.Line2D object at 0x7f41116007f0>]
>>> plt.legend(loc='best')
<matplotlib.legend.Legend object at 0x7f4111600b70>
>>> plt.show()
>>> for n in 2**np.arange(6, 13):
...     plt.plot(alphas, kay_check(alphas, n, 0.001), label=n)
... 
[<matplotlib.lines.Line2D object at 0x7f4111589dd8>]
[<matplotlib.lines.Line2D object at 0x7f4111589ef0>]
[<matplotlib.lines.Line2D object at 0x7f41115952b0>]
[<matplotlib.lines.Line2D object at 0x7f4111595630>]
[<matplotlib.lines.Line2D object at 0x7f41115959b0>]
[<matplotlib.lines.Line2D object at 0x7f4111595d30>]
[<matplotlib.lines.Line2D object at 0x7f41115a20f0>]
>>> plt.legend(loc='best')
<matplotlib.legend.Legend object at 0x7f41115a2470>
>>> plt.show()
>>> np.sqrt(1024)
32.0
>>> for n in 2**np.arange(6, 13):
...     plt.plot(alphas, np.sqrt(n)*kay_check(alphas, n, 0.001), label=n)
... 
[<matplotlib.lines.Line2D object at 0x7f411156beb8>]
[<matplotlib.lines.Line2D object at 0x7f411156ba58>]
[<matplotlib.lines.Line2D object at 0x7f411156b0f0>]
[<matplotlib.lines.Line2D object at 0x7f411156b160>]
[<matplotlib.lines.Line2D object at 0x7f4111564da0>]
[<matplotlib.lines.Line2D object at 0x7f41115642e8>]
[<matplotlib.lines.Line2D object at 0x7f4111564b38>]
>>> plt.legend(loc="best")
<matplotlib.legend.Legend object at 0x7f41115643c8>
>>> plt.show()
>>> for n in 2**np.arange(6, 13):
...     plt.plot(alphas, kay_check(alphas, n, 0.001)/np.sqrt(n), label=n)
... 
[<matplotlib.lines.Line2D object at 0x7f41117902b0>]
[<matplotlib.lines.Line2D object at 0x7f4111790358>]
[<matplotlib.lines.Line2D object at 0x7f4111790978>]
[<matplotlib.lines.Line2D object at 0x7f4111790f98>]
[<matplotlib.lines.Line2D object at 0x7f4111774c18>]
[<matplotlib.lines.Line2D object at 0x7f41117746a0>]
[<matplotlib.lines.Line2D object at 0x7f41117749b0>]
>>> plt.legend(loc="best")
<matplotlib.legend.Legend object at 0x7f4111774fd0>
>>> plt.show()
>>> for n in 2**np.arange(6, 13):
...     plt.plot(alphas, kay_check(alphas, n, 0.001)/n, label=n)
... 
[<matplotlib.lines.Line2D object at 0x7f41114f35f8>]
[<matplotlib.lines.Line2D object at 0x7f41114f3710>]
[<matplotlib.lines.Line2D object at 0x7f41114f3a90>]
[<matplotlib.lines.Line2D object at 0x7f41114f3e10>]
[<matplotlib.lines.Line2D object at 0x7f41115181d0>]
[<matplotlib.lines.Line2D object at 0x7f4111518550>]
[<matplotlib.lines.Line2D object at 0x7f41115188d0>]
>>> plt.legend(loc="best")
<matplotlib.legend.Legend object at 0x7f4111518c50>
>>> plt.show()
>>> for n in 2**np.arange(6, 13):
...     plt.plot(alphas, kay_check(alphas, n, 0.001)/n**(1/3), label=n)
... 
[<matplotlib.lines.Line2D object at 0x7f41115b2a58>]
[<matplotlib.lines.Line2D object at 0x7f41115b2b70>]
[<matplotlib.lines.Line2D object at 0x7f41115b2ef0>]
[<matplotlib.lines.Line2D object at 0x7f41116862b0>]
[<matplotlib.lines.Line2D object at 0x7f4111686630>]
[<matplotlib.lines.Line2D object at 0x7f41116869b0>]
[<matplotlib.lines.Line2D object at 0x7f4111686d30>]
>>> plt.legend(loc="best")
<matplotlib.legend.Legend object at 0x7f411168d0f0>
>>> plt.show()
>>> for n in 2**np.arange(6, 13):
...     plt.plot(alphas, kay_check(alphas, n, 0.001)/n**(1/2), label=n)
... 
[<matplotlib.lines.Line2D object at 0x7f41115994e0>]
[<matplotlib.lines.Line2D object at 0x7f41115995f8>]
[<matplotlib.lines.Line2D object at 0x7f4111599978>]
[<matplotlib.lines.Line2D object at 0x7f4111599cf8>]
[<matplotlib.lines.Line2D object at 0x7f41115840b8>]
[<matplotlib.lines.Line2D object at 0x7f4111584438>]
[<matplotlib.lines.Line2D object at 0x7f41115847b8>]
>>> plt.legend(loc="best")
<matplotlib.legend.Legend object at 0x7f4111584b38>
>>> plt.show()
>>> for n in 2**np.arange(6, 13):
...     plt.plot(alphas, kay_check(alphas, n, 0.001)/n**(1/exp(1)), label=n)
... 
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
NameError: name 'exp' is not defined
>>>     plt.plot(alphas, kay_check(alphas, n, 0.001)/n**(1/np.exp(1)), label=n)
  File "<stdin>", line 1
    plt.plot(alphas, kay_check(alphas, n, 0.001)/n**(1/np.exp(1)), label=n)
    ^
IndentationError: unexpected indent
>>> for n in 2**np.arange(6, 13):
...     plt.plot(alphas, kay_check(alphas, n, 0.001)/n**(1/np.exp(1)), label=n)
... 
[<matplotlib.lines.Line2D object at 0x7f4118060518>]
[<matplotlib.lines.Line2D object at 0x7f4118060630>]
[<matplotlib.lines.Line2D object at 0x7f41180609b0>]
[<matplotlib.lines.Line2D object at 0x7f4118060d30>]
[<matplotlib.lines.Line2D object at 0x7f411173c0f0>]
[<matplotlib.lines.Line2D object at 0x7f411173c470>]
[<matplotlib.lines.Line2D object at 0x7f411173c7f0>]
>>> plt.legend(loc="best")
<matplotlib.legend.Legend object at 0x7f411173cb70>
>>> plt.show()
>>> for n in 2**np.arange(6, 13):
...   C-c C-c
KeyboardInterrupt
>>> for p in np.linspace(2, 2.8, 0.1):
...     plt.figure()
...     for n in 2**np.arange(6, 13):
...         plt.plot(alphas, kay_check(alphas, n, 0.001)/n**(1/np.exp(1)), label=n)
...     plt.legend(loc="best")
... 
Traceback (most recent call last):
  File "/usr/local/lib/python3.6/dist-packages/numpy/core/function_base.py", line 117, in linspace
    num = operator.index(num)
TypeError: 'float' object cannot be interpreted as an integer

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<__array_function__ internals>", line 6, in linspace
  File "/usr/local/lib/python3.6/dist-packages/numpy/core/function_base.py", line 121, in linspace
    .format(type(num)))
TypeError: object of type <class 'float'> cannot be safely interpreted as an integer.
>>> p
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'p' is not defined
>>> plt.show()
>>> plt.figure()
<Figure size 640x480 with 0 Axes>
>>> plt.show()
>>> plt.show("HI")
__main__:1: MatplotlibDeprecationWarning: Passing the block parameter of show() positionally is deprecated since Matplotlib 3.1; the parameter will become keyword-only in 3.3.
>>> plt.show()
>>> plt.figure("HI")
<Figure size 640x480 with 0 Axes>
>>> plt.show()
>>> np.linspace(2.0, 2.8, 9)
array([2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8])
>>> for p in np.linspace(2, 2.8, 9):
    plt.figure(f'Exponent = {np.around(p, 1)}')
    for n in 2**np.arange(6, 13):
        plt.plot(alphas, kay_check(alphas, n, 0.001)/n**(1/np.exp(1)), label=n)
    plt.legend(loc="best")

... ... ... ... ... <Figure size 640x480 with 0 Axes>
[<matplotlib.lines.Line2D object at 0x7f4119b6d2e8>]
[<matplotlib.lines.Line2D object at 0x7f4119b6d400>]
[<matplotlib.lines.Line2D object at 0x7f4119b6d780>]
[<matplotlib.lines.Line2D object at 0x7f4119b6db00>]
[<matplotlib.lines.Line2D object at 0x7f4119b6de80>]
[<matplotlib.lines.Line2D object at 0x7f4119b79240>]
[<matplotlib.lines.Line2D object at 0x7f4119b795c0>]
<matplotlib.legend.Legend object at 0x7f4119ba9fd0>
<Figure size 640x480 with 0 Axes>
[<matplotlib.lines.Line2D object at 0x7f4111472ef0>]
[<matplotlib.lines.Line2D object at 0x7f4111481048>]
[<matplotlib.lines.Line2D object at 0x7f41114813c8>]
[<matplotlib.lines.Line2D object at 0x7f4111481748>]
[<matplotlib.lines.Line2D object at 0x7f4111481ac8>]
[<matplotlib.lines.Line2D object at 0x7f4111481e48>]
[<matplotlib.lines.Line2D object at 0x7f411148b208>]
<matplotlib.legend.Legend object at 0x7f41114baeb8>
<Figure size 640x480 with 0 Axes>
[<matplotlib.lines.Line2D object at 0x7f4111458b38>]
[<matplotlib.lines.Line2D object at 0x7f4111458c50>]
[<matplotlib.lines.Line2D object at 0x7f4111458fd0>]
[<matplotlib.lines.Line2D object at 0x7f4111468390>]
[<matplotlib.lines.Line2D object at 0x7f4111468710>]
[<matplotlib.lines.Line2D object at 0x7f4111468a90>]
[<matplotlib.lines.Line2D object at 0x7f4111468e10>]
<matplotlib.legend.Legend object at 0x7f41114a4940>
<Figure size 640x480 with 0 Axes>
[<matplotlib.lines.Line2D object at 0x7f41113c3780>]
[<matplotlib.lines.Line2D object at 0x7f41113c3898>]
[<matplotlib.lines.Line2D object at 0x7f41113c3c18>]
[<matplotlib.lines.Line2D object at 0x7f41113c3f98>]
[<matplotlib.lines.Line2D object at 0x7f41113d0358>]
[<matplotlib.lines.Line2D object at 0x7f41113d06d8>]
[<matplotlib.lines.Line2D object at 0x7f41113d0a58>]
<matplotlib.legend.Legend object at 0x7f411140b048>
<Figure size 640x480 with 0 Axes>
[<matplotlib.lines.Line2D object at 0x7f41113ac438>]
[<matplotlib.lines.Line2D object at 0x7f41113ac550>]
[<matplotlib.lines.Line2D object at 0x7f41113ac8d0>]
[<matplotlib.lines.Line2D object at 0x7f41113acc50>]
[<matplotlib.lines.Line2D object at 0x7f41113acfd0>]
[<matplotlib.lines.Line2D object at 0x7f4111338390>]
[<matplotlib.lines.Line2D object at 0x7f4111338710>]
<matplotlib.legend.Legend object at 0x7f41113dc128>
<Figure size 640x480 with 0 Axes>
[<matplotlib.lines.Line2D object at 0x7f411130bfd0>]
[<matplotlib.lines.Line2D object at 0x7f4111317128>]
[<matplotlib.lines.Line2D object at 0x7f41113174a8>]
[<matplotlib.lines.Line2D object at 0x7f4111317828>]
[<matplotlib.lines.Line2D object at 0x7f4111317ba8>]
[<matplotlib.lines.Line2D object at 0x7f4111317f28>]
[<matplotlib.lines.Line2D object at 0x7f41113222e8>]
<matplotlib.legend.Legend object at 0x7f411134feb8>
<Figure size 640x480 with 0 Axes>
[<matplotlib.lines.Line2D object at 0x7f4111271c18>]
[<matplotlib.lines.Line2D object at 0x7f4111271d30>]
[<matplotlib.lines.Line2D object at 0x7f411127f0f0>]
[<matplotlib.lines.Line2D object at 0x7f411127f470>]
[<matplotlib.lines.Line2D object at 0x7f411127f7f0>]
[<matplotlib.lines.Line2D object at 0x7f411127fb70>]
[<matplotlib.lines.Line2D object at 0x7f411127fef0>]
<matplotlib.legend.Legend object at 0x7f41112bb940>
<Figure size 640x480 with 0 Axes>
[<matplotlib.lines.Line2D object at 0x7f41112598d0>]
[<matplotlib.lines.Line2D object at 0x7f41112599e8>]
[<matplotlib.lines.Line2D object at 0x7f4111259d68>]
[<matplotlib.lines.Line2D object at 0x7f411126a128>]
[<matplotlib.lines.Line2D object at 0x7f411126a4a8>]
[<matplotlib.lines.Line2D object at 0x7f411126a828>]
[<matplotlib.lines.Line2D object at 0x7f411126aba8>]
<matplotlib.legend.Legend object at 0x7f41112a5908>
<Figure size 640x480 with 0 Axes>
[<matplotlib.lines.Line2D object at 0x7f41112a5c88>]
[<matplotlib.lines.Line2D object at 0x7f41112342b0>]
[<matplotlib.lines.Line2D object at 0x7f41112349b0>]
[<matplotlib.lines.Line2D object at 0x7f4111234ef0>]
[<matplotlib.lines.Line2D object at 0x7f411124e390>]
[<matplotlib.lines.Line2D object at 0x7f411126aa20>]
[<matplotlib.lines.Line2D object at 0x7f41111f45c0>]
<matplotlib.legend.Legend object at 0x7f41114bab70>
>>> plt.show()
>>> for p in np.linspace(2, 2.8, 9):
    plt.figure(f'Exponent = {np.around(p, 1)}') and None
    for n in 2**np.arange(6, 13):
        plt.plot(alphas, kay_check(alphas, n, 0.001)/n**(1/p), label=n) and None
    plt.legend(loc="best") and None

... ... ... ... ... >>> plt.show()
>>> plt.figure(f'Exponent = {np.around(p, 1)}') and None
for n in 2**np.arange(6, 13):
    plt.plot(alphas, kay_check(alphas, n, 0.001), label=n) and None

plt.legend(loc="best") and None

>>> ... ... >>> >>> >>> plt.show()
>>> plt.figure(f'Exponent = {np.around(p, 1)}') and None
for n in 2**np.arange(6, 13):
    plt.plot(alphas, kay_check(alphas, n, 1/n), label=n) and None

plt.legend(loc="best") and None

>>> ... ... >>> >>> >>> plt.show()
>>> plt.figure(f'Exponent = {np.around(p, 1)}') and None
for n in 2**np.arange(6, 13):
    plt.plot(alphas, kay_check(alphas, n, 1/n)/np.sqrt(n), label=n) and None

plt.legend(loc="best") and None

>>> ... ... >>> >>> >>> plt.show()
>>> plt.figure() and None
for n in 2**np.arange(6, 13):
    plt.plot(alphas, kay_check(alphas, n, 1/np.sqrt(n))/np.sqrt(n), label=n) and None

plt.legend(loc="best") and None
plt.show()
>>> ... ... >>> >>> >>> plt.figure() and None
for n in 2**np.arange(6, 13):
    plt.plot(alphas, kay_check(alphas, n, 0.5/n)/np.sqrt(n), label=n) and None

plt.legend(loc="best") and None
plt.show()
>>> ... ... >>> >>> >>> for p in np.linspace(0, 1, 11):
    plt.figure(f'p = {np.around(p, 1)}') and None
    for n in 2**np.arange(6, 13):
        plt.plot(alphas, kay_check(alphas, n, p/n)/np.sqrt(n), label=n) and None
    plt.legend(loc="best") and None

plt.show()
... ... ... ... ... <string>:2: RuntimeWarning: invalid value encountered in true_divide
<string>:2: RuntimeWarning: invalid value encountered in true_divide
<string>:2: RuntimeWarning: invalid value encountered in true_divide
<string>:2: RuntimeWarning: invalid value encountered in true_divide
<string>:2: RuntimeWarning: invalid value encountered in true_divide
<string>:2: RuntimeWarning: invalid value encountered in true_divide
<string>:2: RuntimeWarning: invalid value encountered in true_divide
>>> plt.figure() and None
for n in 2**np.arange(6, 13):
    plt.plot(alphas, kay_check(alphas, n, 1/n**2)/np.sqrt(n), label=n) and None

plt.legend(loc="best") and None
plt.show()
>>> ... ... >>> >>> >>> for p in np.linspace(0, -2, 11):
    plt.figure(f'p = {np.around(p, 1)}') and None
    for n in 2**np.arange(6, 13):
        plt.plot(alphas, kay_check(alphas, n, n**p)/np.sqrt(n), label=n) and None
    plt.legend(loc="best") and None

plt.show()
... ... ... ... ... >>> >>> for p in np.linspace(-0.2, -0.4, 11):
    plt.figure(f'p = {np.around(p, 1)}') and None
    for n in 2**np.arange(6, 13):
        plt.plot(alphas, kay_check(alphas, n, n**p)/np.sqrt(n), label=n) and None
    plt.legend(loc="best") and None

plt.show()
... ... ... ... ... >>> >>> 
for p in np.linspace(-0.2, -0.4, 11):
    plt.figure(f'p = {np.around(p, 1)}') and None
    for n in 2**np.arange(6, 13):
        plt.plot(alphas, kay_check(alphas, n, n**p)/np.sqrt(n), label=n) and None
    plt.legend(loc="best") and None

plt.show()
>>> ... ... ... ... ... >>> >>> 
for p in np.linspace(-0.4, -0.2, 11):
    plt.figure(f'p = {np.around(p, 1)}') and None
    for n in 2**np.arange(6, 13):
        plt.plot(alphas, kay_check(alphas, n, n**p)/np.sqrt(n), label=n) and None
    plt.legend(loc="best") and None

plt.show()
>>> ... ... ... ... ... >>> >>> np.linspace(-0.2, -0.4, 11)
array([-0.2 , -0.22, -0.24, -0.26, -0.28, -0.3 , -0.32, -0.34, -0.36,
       -0.38, -0.4 ])
>>> 
for p in np.linspace(-0.4, -0.2, 11):
    plt.figure(f'p = {np.around(p, 2)}') and None
    for n in 2**np.arange(6, 13):
        plt.plot(alphas, kay_check(alphas, n, n**p)/np.sqrt(n), label=n) and None
    plt.legend(loc="best") and None

plt.show()
>>> ... ... ... ... ... >>> >>> 
for p in np.linspace(-0.34, -0.32, 11):
    plt.figure(f'p = {np.around(p, 3)}') and None
    for n in 2**np.arange(6, 13):
        plt.plot(alphas, kay_check(alphas, n, n**p)/np.sqrt(n), label=n) and None
    plt.legend(loc="best") and None

plt.show()
>>> ... ... ... ... ... >>> >>> plt.figure() and None
for n in 2**np.arange(6, 13):
    plt.plot(alphas, kay_check(alphas, n, n**(-1/3))/np.sqrt(n), label=n) and None

plt.legend(loc="best") and None
plt.show()
>>> ... ... >>> >>> >>> sizes
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'sizes' is not defined
>>> np.arange(6, 13)**(-1/3)
array([0.55032121, 0.52275796, 0.5       , 0.48074986, 0.46415888,
       0.44964431, 0.43679023])
>>> (2**np.arange(6, 13))**(-1/3)
array([0.25      , 0.19842513, 0.15749013, 0.125     , 0.09921257,
       0.07874507, 0.0625    ])
>>> 2**(-6*np.arange(6, 13)/3)
array([2.44140625e-04, 6.10351562e-05, 1.52587891e-05, 3.81469727e-06,
       9.53674316e-07, 2.38418579e-07, 5.96046448e-08])
>>> 2**(-np.arange(6, 13)/3)
array([0.25      , 0.19842513, 0.15749013, 0.125     , 0.09921257,
       0.07874507, 0.0625    ])
>>> dev.off()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'dev' is not defined
>>> import power_law_noise as pln
>>> pln.PowerLawNoise(-2, 0)
<power_law_noise.PowerLawNoise object at 0x7f4119b989e8>
>>> p = pln.PowerLawNoise(-2, 0)
>>> p.terms
array([1.])
>>> p = pln.PowerLawNoise(-2, 1)
>>> p.terms
array([ 1., -1.])
>>> y = p(np.zeros(512))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/ben/Source/Open/powerlawnoise/pypowerlawnoise/power_law_noise.py", line 227, in __call__
    return np.dot(noise, self._h[-len(noise):])
  File "<__array_function__ internals>", line 6, in dot
ValueError: shapes (512,) and (1,) not aligned: 512 (dim 0) != 1 (dim 0)
>>> np.linspace(0.25, 1, 4)
array([0.25, 0.5 , 0.75, 1.  ])
>>> q()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'q' is not defined
>>> quit()

Process Python finished
