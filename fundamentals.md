### Python

- sources
	- https://github.com/taizilongxu/interview_python
	- https://www.interviewbit.com/python-interview-questions/
- 传参 call by reference
	- immutable: str, tuple, num
	- mutable: list, dict, set
- 元类 metaclass
	- https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python
	- classes are objects, everything except "type" are objects
	- "type" creates class dynamically (on the fly)
	- "type" is the metaclass that creates all classes, all classes are instances of "type"
	- we can alter `__metaclass__()` to dynamically initialize a class
	- we do not need it most of the time though
- 方法 method
	- https://stackoverflow.com/questions/136097/difference-between-staticmethod-and-classmethod
	- https://realpython.com/instance-class-and-static-methods-demystified/
	- instance method: `func(self, args)`
	- `@classmethod`: `func(cls, args)`
	- `@staticmethod`: `func(args)`
- 类变量，实例变量
	- `class.num_of_instance`
	- `class_instance.arg`
- 自省
	- `type(), dir(), getattr(), hasattr(), isinstance()`
- 字典推导式 dictionary comprehension
	- `d = {key: value for (key, value) in iterable}`
- 单双下划线
	- https://stackoverflow.com/questions/1301346/what-is-the-meaning-of-single-and-double-underscore-before-an-object-name
	- https://www.zhihu.com/question/19754941
	- `__func__()`: 特殊方法，如 `__init__(), __call__(), __del__(), __add__(), __getitem__()`, global `__file__(), __name__()`
	- `_func()`: Name Mangling, enforce private to avoid name overlap, not imported
	- `__func()`: convention private, not imported
- 字符串格式化
	- https://stackoverflow.com/questions/5082452/string-formatting-vs-format-vs-f-string-literal
	- `"%s" % name` -> `"{}".format` -> `f"{}"`
- 迭代器 list-like，生成器 generator
	- https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python
	- generate use constant space, `yield`
- 缺省参数 `*args, **kwargs`
	- https://stackoverflow.com/questions/3394835/use-of-args-and-kwargs
	- `*args` 任意数量 no-named 传参
	- `**kwargs` 任意数量 named 传参
- 面向切面编程 AOP，装饰器 decorator
	- https://stackoverflow.com/questions/739654/how-do-i-make-function-decorators-and-chain-them-together
	- 目的是为已经存在的对象添加额外的功能
	- `@decorator_func` and `@decorator_func()`
	- `@functolls.wraps(func)`
	- these are all builtin decorators: `@property, @classmethod, @staticmethod`
- 鸭子类型 duck typing
	- `list.extend()` 只要是可迭代的就行: list, tuple, dict, str, generator
- 重载 overloading
	- https://www.zhihu.com/question/20053359
	- 重载的目的：可变参数类型，可变参数个数
	- python 语言特性已经实现了重载，故不需要
- 新式类，旧式类(经典类) new, old(classic) style classes
	- https://stackoverflow.com/questions/54867/what-is-the-difference-between-old-style-and-new-style-classes-in-python
	- descriptor: http://www.cnblogs.com/btchenguang/archive/2012/09/17/2689146.html
	- python3 只有新式类, 下面是新式类的 behavior changes
	- `super`: https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
	- `__mro__`: MRO (Method Resolution Order) new - C3 (BFS) MRO, old - DFS MRO (and there is no `__mro__` attribute)
- `__new__(cls), __init__(self)`
	- https://stackoverflow.com/questions/674304/why-is-init-always-called-after-new
	- `__new__(cls)` is classmethod, `__init__(self)` is instance method
	- `__metaclass__()` controls class creation, `__new__(cls)` controls class creation, `__init__(self)` is called after class creation
- 单例模式
	- 保证一个类只有一个对象
	- 方法1：通过 `__new__(cls)` 把classmethod指向同一个对象
	- 方法2：通过 `__new__(cls)` 把 `__dict__()` 共享属性
	- 方法3：类装饰器
	- 方法4：import 类的对象
- 作用域
	- local -> enclosing locals -> global -> builtin
- 线程全局锁 GIL (Global Interpreter Lock) - 相当困难 TODO
	- 对于 io 密集型任务，python 的多线程起到作用
	- 对于 cpu 密集型任务，python 的多线程可能会因为争夺资源而变慢
	- 解决方法：多进程，协程
- 协程 TODO
	- 进程和线程的升级版
	- 进程和线程都面临着内核态和用户态的切换问题而耗费许多切换时间,而协程是用户自己控制切换的时机,不再需要陷入系统的内核态
	- `yield` 是协程思想
- 闭包 closure TODO
	- 当一个内嵌函数引用其外部作用域的变量,我们就会得到一个闭包
	- 创建闭包必须满足以下条件
	1. 必须有一个内嵌函数
	2. 内嵌函数必须引用外部函数中的变量
	3. 外部函数的返回值必须是内嵌函数
- 函数式编程 lambda
	- https://www.zhihu.com/question/20125256
	- https://coolshell.cn/articles/10822.html
	- 匿名函数，简洁性高
- 垃圾回收 GC (garbage collection)
	- 引用计数 (reference counting): `ob_refcnt` 简单实时，但维护计数消耗资源，且有循环引用问题
	- 标记-清除 (mark-sweep): 按需分配，当内存不够时清扫空间
	- 分代回收 (generation collection): 根据内存块存活时间划分三代集合，越多代，GC 频率越低
- `read, readline, readlines`
	- `read` 读取整个文件
	- `readline` 用生成器读下一行
	- `readlines` 读取整个文件到迭代器
- `super().__init__()`
	- https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
	- 作用于多继承 multiple inheritance
- 其他 etc.
	- `reference` vs `copy.copy()` vs `copy.deepcopy()`
	- python list 底层：https://www.jianshu.com/p/J4U6rR
	- `is` by reference, `==` by value
	- python2 vs python3: http://chenqx.github.io/2014/11/10/Key-differences-between-Python-2-7-x-and-Python-3-x/
	- range vs xrange: https://stackoverflow.com/questions/94935/what-is-the-difference-between-range-and-xrange-functions-in-python-2-x

### C++



### OS



### System Design



### Network



### Database

