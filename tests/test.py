import inspect
from functools import wraps
from typing import Any, Callable, Sequence, NewType, TypeVar, cast, Protocol
from typing import Generic

import pytest

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
F = TypeVar('F', bound=Callable[..., T])

class Wrapped(Protocol,Generic[T_co]):
    def __call__(self, *args: Any, **kwargs: Any) -> T_co: ...

class Proxy(Protocol, Generic[T_co]):
    def __getattr__(self, key: Any) -> Any: ...
    def __setattr__(self, key: Any, val: Any) -> None: ...
    def __delattr__(self, key: Any) -> None: ...

class NoParam(Proxy[T]):
    pass

class Fixture(Proxy[T]):
    pass


def parametrize(
    *data: Any,
    args: None | str | Sequence[str]=None,
) -> Callable[[Wrapped[T]], Wrapped[T]]:
    def decorator(func: Wrapped[T]) -> Wrapped[T]:
        params: Sequence[str]
        test_params = inspect.signature(func).parameters
        if args is None:
            params = [name for (name, p) in test_params.items()
                      if p.annotation is not NoParam]
        else:
            params = args.split(',') if isinstance(args, str) else args
        indirect = [name for name in params if _is_indirect(test_params[name])]
        @pytest.mark.parametrize(','.join(params), data, indirect=indirect)
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)
        return cast(Wrapped[T], wrapper)
    return decorator


def _is_indirect(param: inspect.Parameter) -> bool:
    try:
        return param.annotation.__origin__ is Fixture
    except AttributeError:
        return False
