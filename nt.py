import re

OMITTED = {"\n", "\t", " "}
INVALID_NAME = {"(", ")"}.union(OMITTED)


PREF_DICT = {"tk": True, "ast": False}


def read_file(file_name, parent_path) -> str:
    full_path = parent_path + os.sep + file_name
    with open(full_path, "r") as rf:
        return rf.read()


def parse(source: str) -> list:
    tokens = ["("] + \
             [p for p in re.split("(\s|\".*?\"|(?<!')\(|'\(|\)|\[|\])", source) if p.strip() or p == "\n"] + \
             [")"]

    if PREF_DICT["tk"]:
        print(tokens)

    stack = []
    comment = False
    tup_depth = 0
    try:
        for tk in tokens:
            if len(tk) > 0:
                if tk == "#":
                    comment = True
                elif tk == "\n":
                    comment = False

                if not comment:
                    if is_int(tk):
                        stack[-1].append(int(tk))
                    elif is_float(tk):
                        stack[-1].append(float(tk))
                    elif is_str(tk):
                        stack[-1].append(String(tk[1:-1]))
                    elif tk == "true":
                        stack[-1].append(TRUE)
                    elif tk == "false":
                        stack[-1].append(FALSE)
                    elif tk == "null":
                        stack[-1].append(NULL)
                    elif tk == "(":
                        if tup_depth > 0:
                            stack.append(Tuple())
                            tup_depth += 1
                        else:
                            stack.append([])
                    elif tk == "[":
                        stack.append([])
                    elif tk == ")":
                        if len(stack) < 2:
                            break
                        active = stack.pop()
                        if active.__class__ == Tuple:
                            tup_depth -= 1
                        stack[-1].append(active)
                    elif tk == "]":
                        stack[-2].append(stack.pop())
                    elif tk == "'(":
                        tup_depth += 1
                        stack.append(Tuple())
                    elif tk not in OMITTED:
                        stack[-1].append(tk)
    except IndexError as e:
        print(stack)
        raise e

    if PREF_DICT["ast"]:
        print(stack[0])

    return stack[0]


class Error(Exception):
    def __init__(self, msg=""):
        super().__init__(msg)


class Env:
    def __init__(self, outer):
        self.vars = {}
        self.outer: Env = outer

    def _inner_get(self, name):
        if name in self.vars:
            return self.vars[name]
        if self.outer:
            return self.outer.get(name)
        else:
            return None

    def get(self, name):
        value = self._inner_get(name)
        if value is None:
            raise Error("Name '" + name + "' is not defined in this scope.")
        else:
            return value

    def has_name(self, name):
        return self._inner_get(name) is not None

    def put(self, name, value):
        self.vars[name] = value


def check_contract(contract, arg, eval_ftn):
    if callable(contract):
        if contract(arg) is not TRUE:
            raise Error("Contract violation, got a " + str(arg))
    elif isinstance(contract, Function):
        if contract.call(eval_ftn, [arg]) is not TRUE:
            raise Error("Contract violation, got a " + str(arg))


class Function:
    def __init__(self, params, body, def_env):
        self.params: list = params
        self.body = body
        self.def_env = def_env

        self.param_contracts = None
        self.rtn_contract = None

    def has_contract(self):
        return self.param_contracts is not None and self.param_contracts is not None

    def call(self, eval_ftn, args: list):
        """
        Call a user-defined function.

        :param eval_ftn: the python recursive interpreter function
        :param args: evaluated arguments
        :return: the return value of the function called
        """
        params_count = len(self.params)
        var_len_param = "*" in self.params

        if not var_len_param and params_count != len(args):
            raise Error("Arity mismatch. Expected: {}, actual: {}.".format(params_count, len(args)))

        call_env = Env(self.def_env)
        for i in range(params_count):
            name = self.params[i]
            if name == "*":
                if i != params_count - 2:
                    raise Error("Variable length parameter must be the last parameter.")
                name = self.params[i + 1]
                tup = Tuple()
                for j in range(i, len(args)):
                    arg = args[j]
                    if self.param_contracts:
                        con = self.param_contracts[i]
                        check_contract(con, arg, eval_ftn)
                    tup.append(arg)

                nt_list = tuple_to_nt_list(tup, eval_ftn, call_env)
                if isinstance(name, str) and name not in INVALID_NAME:
                    call_env.put(name, nt_list)

                break

            arg = args[i]
            if self.param_contracts:
                con = self.param_contracts[i]
                # print(con, arg)
                check_contract(con, arg, eval_ftn)
            if isinstance(name, str) and name not in INVALID_NAME:
                call_env.put(name, arg)
            else:
                raise Error("Function parameter must be a name.")

        rtn_value = eval_ftn(self.body, call_env)
        if self.rtn_contract is not None:
            check_contract(self.rtn_contract, rtn_value, eval_ftn)

        return rtn_value


class Boolean:
    def __init__(self, v):
        self.value = v

    def __bool__(self):
        return self.value

    def __str__(self):
        return "true" if self.value else "false"

    def __repr__(self):
        return self.__str__()


class Null:
    def __init__(self):
        pass

    def __str__(self):
        return "null"

    def __repr__(self):
        return self.__str__()


class Struct:
    def __init__(self, name, parent_struct, params: list):
        self.name = name
        self.parent_struct: Struct = parent_struct
        self.params = params

        self.param_contracts = None

    def _get_all_params(self) -> list:
        if self.parent_struct:
            return self.parent_struct._get_all_params() + self.params
        else:
            return self.params

    def _default_contract(self, env: Env):
        any_ftn = env.get("any")
        return [any_ftn for _ in range(len(self.params))]

    def _get_all_contracts(self, env: Env) -> list:
        if self.parent_struct:
            return self.parent_struct._get_all_contracts(env) + \
                   (self.param_contracts if self.param_contracts else self._default_contract(env))
        else:
            return self.param_contracts if self.param_contracts else self._default_contract(env)

    def create_instance(self, eval_ftn, env: Env, args: list):
        params = self._get_all_params()
        if len(params) != len(args):
            raise Error("Struct initialization must have equal number of arguments with all members of the struct.")

        param_contracts = self._get_all_contracts(env)
        attrs = {}
        for i in range(len(params)):
            arg = args[i]
            con = param_contracts[i]
            check_contract(con, arg, eval_ftn)
            attrs[params[i]] = arg

        return StructObj(self, attrs)

    def is_child_of_this(self, struct) -> bool:
        if struct.name == self.name:
            return True
        elif struct.parent_struct:
            return self.is_child_of_this(struct.parent_struct)
        return False

    def __str__(self):
        return "Struct-def<" + self.name + ">"


class StructObj:
    def __init__(self, struct: Struct, attrs):
        self.struct: Struct = struct
        self.attrs: dict = attrs

    def __str__(self):
        return "struct<{}>".format(self.struct.name)


class Tuple(list):
    def __init__(self, *args):
        super().__init__(args)

    def __str__(self):
        return "tuple" + super().__repr__()

    def __repr__(self):
        return "tuple" + super().__repr__()


class String:
    def __init__(self, s):
        self.s: str = s

    def __str__(self):
        return '"' + self.s + '"'

    def __add__(self, other):
        return String(self.s + other.s) if isinstance(other, String) else error("String concat must take two strings")


TRUE = Boolean(True)
FALSE = Boolean(False)
NULL = Null()


DIRECT_RETURN_TYPES = {int, float, Boolean, Null, String}


def boolean(value: bool) -> Boolean:
    return TRUE if value else FALSE


def is_fn(f) -> Boolean:
    return boolean(callable(f) or isinstance(f, Function))


def make_nt_apply(eval_ftn, env: Env):
    def apply(ftn, lst):
        if callable(ftn):
            if isinstance(lst, StructObj) and lst.struct.name == "list":
                py_lst = nt_list_to_py_list(lst, eval_ftn, env)
                return ftn(*py_lst)
            else:
                raise Error("Apply takes a list as second argument.")
        elif isinstance(ftn, Function):
            if isinstance(lst, StructObj) and lst.struct.name == "list":
                py_lst = nt_list_to_py_list(lst, eval_ftn, env)
                return ftn.call(eval_ftn, py_lst)
            else:
                raise Error("Apply takes a list as second argument.")
        else:
            raise Error("Apply takes either a builtin function or a user-defined function as first argument.")
    return apply


def make_to_str_ftn(eval_ftn, env: Env):
    def to_str_fn(obj):
        if isinstance(obj, StructObj):
            str_fn_name = obj.struct.name + ".str"
            str_fn = env.get(str_fn_name)
            if callable(str_fn):
                return str_fn(obj)
            elif isinstance(str_fn, Function):
                return str_fn.call(eval_ftn, [obj])
            else:
                raise Error("struct.str must be a unary function")
        else:
            return String(str(obj))

    return to_str_fn


def error(error_str: str):
    raise Error(error_str)


def make_struct(name: str, parent, content: list, env: Env):
    parent_struct = env.get(parent) if parent else None
    if env.has_name(name):
        raise Error("Struct '" + name + "' is already defined in this scope.")

    getters = {}

    def getter_fn(param_name):
        return lambda obj: \
            obj.attrs[param_name] \
            if isinstance(obj, StructObj) else error("Struct '" + name + "' expected, got a " + str(obj))

    for param in content:
        if not isinstance(param, str) or param in INVALID_NAME:
            raise Error("Struct member must be a name.")
        getter_name = name + "." + param
        getters[getter_name] = getter_fn(param)

    struct = Struct(name, parent_struct, content)

    env.put(name, struct)
    env.put(name + "?",  # checker function
            lambda obj: boolean(struct.is_child_of_this(obj.struct)) if isinstance(obj, StructObj) else FALSE)
    env.put(name + ".str", lambda s: String(str(s)))  # default to string function
    for getter_name in getters:
        env.put(getter_name, getters[getter_name])

    return struct


def foldl_n():
    pass


def put_builtins(env: Env, main_eval_ftn):
    env.put("+", lambda a, b: a + b)
    env.put("-", lambda a, b: a - b)
    env.put("*", lambda a, b: a * b)
    env.put("/", lambda a, b: a / b)
    env.put("%", lambda a, b: a % b)
    env.put("<", lambda a, b: boolean(a < b))
    env.put("<=", lambda a, b: boolean(a <= b))
    env.put(">", lambda a, b: boolean(a > b))
    env.put(">=", lambda a, b: boolean(a >= b))
    env.put("==", lambda a, b: boolean(a == b))
    env.put("and", lambda *args: boolean(all(args)))
    env.put("any", lambda x: TRUE)
    env.put("apply", make_nt_apply(main_eval_ftn, env))
    env.put("boolean?", lambda b: boolean(b is TRUE or b is FALSE))
    env.put("float", lambda x: float(x))
    env.put("float?", lambda x: boolean(isinstance(x, float)))
    env.put("int", lambda x: int(x))
    env.put("int?", lambda x: boolean(isinstance(x, int)))
    env.put("fn?", is_fn)
    env.put("null?", lambda n: boolean(n is NULL))
    env.put("or", lambda *args: boolean(any(args)))
    env.put("str", make_to_str_ftn(main_eval_ftn, env))
    env.put("str?", lambda s: boolean(isinstance(s, String)))
    env.put("struct?", lambda s: boolean(isinstance(s, Struct)))
    env.put("tuple?", lambda t: boolean(isinstance(t, Tuple)))
    env.put("tuple.get", lambda t, i: t[i])
    env.put("tuple.length", lambda t: len(t))


class NtFileInterpreter:
    def __init__(self, file_path, required):
        self.file_path = os.path.abspath(file_path)
        self.parent_dir = os.path.dirname(self.file_path)

        self.required: set = required
        self.traceback = []

    def get_import_path(self, name):
        """
        Returns the absolute path if the required file is a user-defined library.

        :param name:
        :return:
        """
        if is_str(name):  # user's lib
            pure_name = name[1:-1]
            if os.path.isabs(pure_name):
                pass
            else:
                return self.parent_dir + os.sep + pure_name
        else:
            return "lib" + os.sep + name + ".nt"

    def interpret(self, is_main: bool, env=None):
        with open(self.file_path, "r") as rf:
            source = rf.read()
        program = parse(source)
        # print(program)
        if env is None:
            env = Env(None)
            put_builtins(env, self.evaluate)

        self.evaluate(["require", "lang"], env)

        res = NULL
        i = 0
        try:
            while i < len(program):
                expr = program[i]
                if expr == "##":  # comment next expr
                    i += 2
                    continue
                if is_main and expr == ":":  # next expr is the main expr
                    res = self.evaluate(program[i + 1], env)
                    break
                self.evaluate(expr, env)
                i += 1
                self.traceback.clear()

            return res
        except Exception as e:
            print("Nt traceback:")
            for tb in self.traceback:
                print(tb)
            raise e

    def evaluate(self, expr, env: Env):
        self.traceback.append(expr)
        if expr.__class__ in DIRECT_RETURN_TYPES:
            return expr
        elif isinstance(expr, str):
            return env.get(expr)
        elif expr.__class__ == Tuple:
            return tuple_to_nt_list(expr, self.evaluate, env)
        elif expr.__class__ == list:
            if len(expr) > 0:
                first = expr[0]
                if first == "def":
                    if len(expr) == 3:
                        name = expr[1]
                        if name in INVALID_NAME:
                            raise Error("Name '" + name + "' is invalid.")
                        else:
                            value = self.evaluate(expr[2], env)
                            env.put(name, value)
                    else:
                        raise Error("Definition must have 3 parts: (def name value).")
                elif first == "if":
                    if len(expr) == 4:
                        cond = self.evaluate(expr[1], env)
                        if cond is TRUE:
                            return self.evaluate(expr[2], env)
                        elif cond is FALSE:
                            return self.evaluate(expr[3], env)
                        else:
                            raise Error("Condition of if-expression must returns a boolean.")
                    else:
                        raise Error("If statement must have 4 parts: (if condition then else)")
                elif first == "cond":
                    for line in expr[1:]:
                        if line.__class__ == list and len(line) == 2:
                            if line[0] == "else":
                                return self.evaluate(line[1], env)
                            else:
                                cond = self.evaluate(line[0], env)
                                if cond is TRUE:
                                    return self.evaluate(line[1], env)
                        else:
                            raise Error("Condition in cond expression must have two parts: "
                                        "[condition body] | [else body], got '" + str(line) + "'.")
                elif first == "let":
                    if len(expr) == 3:
                        local_env = Env(env)
                        for bd in expr[1]:
                            if bd.__class__ == list and \
                                    len(bd) == 2 and \
                                    isinstance(bd[0], str) and \
                                    bd[0] not in INVALID_NAME:
                                local_env.put(bd[0], self.evaluate(bd[1], env))
                            else:
                                raise Error("Bindings in local binding must be of the form (name expr)")
                        return self.evaluate(expr[2], local_env)
                    else:
                        raise Error("Local binding expr must have 3 parts: (let (bindings*) body)")
                elif first == "fn":
                    if len(expr) == 3:
                        return Function(expr[1], expr[2], env)
                    else:
                        raise Error("Function definition must have 3 parts: (fn (params) body).")
                elif first == "con":
                    if len(expr) == 3:
                        name = expr[1]
                        ftn = env.get(name)
                        is_struct = False
                        if isinstance(ftn, Struct):
                            is_struct = True
                        elif not isinstance(ftn, Function):
                            raise Error("Target of contract definition must be a function or a struct.")
                        contracts = expr[2]
                        if len(contracts) > 1 and contracts[-2] == "->":
                            param_cons = contracts[:-2]
                            rtn_con = contracts[-1]
                            if "*" not in ftn.params and len(param_cons) != len(ftn.params):
                                raise Error("Contracts must have the same number of parameters as the function.")
                            eval_param_cons = []
                            for i in range(len(param_cons)):
                                param_con = self.evaluate(param_cons[i], env)
                                if is_fn(param_con):
                                    eval_param_cons.append(param_con)
                                else:
                                    raise Error("Contract must be a boolean-valued function.")
                            eval_rtn_con = self.evaluate(rtn_con, env)
                            if is_fn(eval_rtn_con):
                                ftn.param_contracts = eval_param_cons
                                if not is_struct:
                                    ftn.rtn_contract = eval_rtn_con
                            else:
                                raise Error("Contract must be a boolean-valued function.")
                        else:
                            raise Error("Contract must have at least one return contract.")
                    else:
                        raise Error("Contract definition must have 3 parts: (con func (params* -> rtn)).")
                elif first == "struct":
                    if 3 <= len(expr) <= 4:
                        name = expr[1]
                        if len(expr) == 4:
                            parent = expr[2]
                            content = expr[3]
                        else:
                            parent = None
                            content = expr[2]
                        if not isinstance(name, str) or \
                                not isinstance(content, list) or \
                                (parent and not isinstance(parent, str)):
                            raise Error("Invalid struct definition.")
                        return make_struct(name, parent, content, env)
                    else:
                        raise Error("Struct definition must have at least three parts: (struct name (expr*)). ")
                elif first == "require":
                    if len(expr) == 2:
                        file_name = expr[1]
                        full_path = self.get_import_path(file_name)
                        if full_path not in self.required:  # avoid duplicate request
                            self.required.add(full_path)
                            file_itr = NtFileInterpreter(full_path, self.required)
                            file_itr.interpret(False, env)
                    else:
                        raise Error("Require must have two parts: (require file). ")
                else:  # function call
                    func = self.evaluate(first, env)
                    args = [self.evaluate(arg, env) for arg in expr[1:]]
                    if callable(func):
                        return func(*args)
                    elif isinstance(func, Function):
                        return func.call(self.evaluate, args)
                    elif isinstance(func, Struct):
                        return func.create_instance(self.evaluate, env, args)
                    else:
                        raise Error("Expression " + str(func) + " is not callable.")

        return NULL


# helper functions

def tuple_to_nt_list(tup: Tuple, eval_ftn, env: Env):
    make: Function = env.get("list.make")
    return make.call(eval_ftn, [tup])


def nt_list_to_py_list(lst: StructObj, eval_ftn, env: Env) -> list:
    get: Function = env.get("list.get")
    len_ftn: Function = env.get("list.length")
    length: int = len_ftn.call(eval_ftn, [lst])
    py_lst = []
    for i in range(length):
        py_lst.append(get.call(eval_ftn, [lst, i]))
    return py_lst


def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_str(s: str) -> bool:
    return len(s) >= 2 and \
           (s[0] == '"' and s[-1] == '"')


def parse_args(arg_list: list):
    pass


test = """
(def a 1)
(def f (fn (x) (+ x 1)))
(contract f (int? -> int?))
:(+ a (+ 2 (f 3)))
"""


if __name__ == '__main__':
    import sys
    import os

    if len(sys.argv) > 1:
        src_file = sys.argv[1]
        itr = NtFileInterpreter(src_file, set())
        rtn = itr.interpret(True)
        print(rtn)
