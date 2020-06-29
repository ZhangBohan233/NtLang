import re

OMITTED = {"\n", "\t", " "}
INVALID_NAME = {"(", ")"}.union(OMITTED)


def read_file(file_name, parent_path) -> str:
    full_path = parent_path + os.sep + file_name
    with open(full_path, "r") as rf:
        return rf.read()


def parse(source: str) -> list:
    tokens = ["("] + re.split("(\s|\(|\))", source) + [")"]
    stack = []
    comment = False
    try:
        for tk in tokens:
            if len(tk) > 0:
                if tk == "(":
                    stack.append([])
                elif tk == ")":
                    if len(stack) < 2:
                        break
                    stack[-2].append(stack.pop())
                elif tk not in OMITTED:
                    stack[-1].append(tk)
    except IndexError as e:
        print(stack)
        raise e

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
            raise Error("Contract violation.")
    elif isinstance(contract, Function):
        if contract.call(eval_ftn, arg) is not TRUE:
            raise Error("Contract violation.")


class Function:
    def __init__(self, params, body, def_env):
        self.params: list = params
        self.body = body
        self.def_env = def_env

        self.param_contracts = None
        self.rtn_contract = None

    def has_contract(self):
        return self.param_contracts is not None and self.param_contracts is not None

    def call(self, eval_ftn, args):
        params_count = len(self.params)
        if params_count != len(args):
            raise Error("Arity mismatch. Expected: {}, actual: {}.".format(params_count, len(args)))

        call_env = Env(self.def_env)
        for i in range(params_count):
            name = self.params[i]
            arg = args[i]
            if self.param_contracts is not None:
                con = self.param_contracts[i]
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


class Struct:
    def __init__(self, name, parent_struct, members: list):
        self.name = name
        self.parent_struct: Struct = parent_struct
        self.members = members

    def _get_all_params(self) -> list:
        if self.parent_struct:
            return self.parent_struct._get_all_params() + self.members
        else:
            return self.members

    def create_instance(self, args: list):
        params = self._get_all_params()
        if len(params) != len(args):
            raise Error("Struct initialization must have equal number of arguments with all members of the struct.")

        attrs = {}
        for i in range(len(params)):
            attrs[params[i]] = args[i]

        return StructObj(self, attrs)

    def is_child_of_this(self, struct) -> bool:
        if struct.name == self.name:
            return True
        elif struct.parent_struct:
            return self.is_child_of_this(struct.parent_struct)
        return False


class StructObj:
    def __init__(self, struct: Struct, attrs):
        self.struct = struct
        self.attrs: dict = attrs


TRUE = Boolean(True)
FALSE = Boolean(False)


def boolean(value: bool) -> Boolean:
    return TRUE if value else FALSE


def is_fn(f) -> Boolean:
    return boolean(callable(f) or isinstance(f, Function))


def error(exception: Exception):
    raise exception


def make_struct(name: str, parent, content: list, env: Env):
    parent_struct = env.get(parent) if parent else None
    if env.has_name(name):
        raise Error("Struct '" + name + "' is already defined in this scope.")

    getters = {}

    def getter_fn(param_name):
        return lambda obj: \
            obj.attrs[param_name] \
            if isinstance(obj, StructObj) else error(Error("Struct '" + name + "' expected, got a " + str(obj)))

    for param in content:
        if not isinstance(param, str) or param in INVALID_NAME:
            raise Error("Struct member must be a name.")
        getter_name = name + "." + param
        getters[getter_name] = getter_fn(param)

    struct = Struct(name, parent_struct, content)

    env.put(name, struct)
    env.put(name + "?",
            lambda obj: boolean(struct.is_child_of_this(obj.struct)) if isinstance(obj, StructObj) else FALSE)
    for getter_name in getters:
        env.put(getter_name, getters[getter_name])

    return struct


def put_builtins(env: Env):
    env.put("+", lambda a, b: a + b)
    env.put("<", lambda a, b: boolean(a < b))
    env.put("int", lambda x: int(x))
    env.put("int?", lambda x: boolean(isinstance(x, int)))
    env.put("fn?", is_fn)


class NtFileInterpreter:
    def __init__(self, file_path):
        self.file_path = os.path.abspath(file_path)
        self.parent_dir = os.path.dirname(self.file_path)

    def get_import_path(self, name):
        if (name[0] == '"' and name[-1] == '"') or (name[0] == "'" and name[-1] == "'"):  # user's lib
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
            put_builtins(env)
        res = None
        i = 0
        while i < len(program):
            expr = program[i]
            if expr == "#":  # comment next expr
                i += 2
                continue
            if is_main and expr == ":":  # next expr is the main expr
                res = self.evaluate(program[i + 1], env)
                break
            self.evaluate(expr, env)
            i += 1
        return res

    def evaluate(self, expr, env: Env):
        if isinstance(expr, str):
            if is_int(expr):
                return int(expr)
            elif is_float(expr):
                return float(expr)
            elif expr == "true":
                return TRUE
            elif expr == "false":
                return FALSE
            else:
                return env.get(expr)
        elif isinstance(expr, list):
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
                elif first == "fn":
                    if len(expr) == 3:
                        return Function(expr[1], expr[2], env)
                    else:
                        raise Error("Function definition must have 3 parts: (fn (params) body).")
                elif first == "con":
                    if len(expr) == 3:
                        name = expr[1]
                        ftn = env.get(name)
                        if not isinstance(ftn, Function):
                            raise Error("Target of contract definition must be a function.")
                        contracts = expr[2]
                        if len(contracts) > 1 and contracts[-2] == "->":
                            param_cons = contracts[:-2]
                            rtn_con = contracts[-1]
                            if len(param_cons) != len(ftn.params):
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
                        file_itr = NtFileInterpreter(self.get_import_path(file_name))
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
                        return func.create_instance(args)
                    else:
                        raise Error("Expression " + str(func) + " is not callable.")


# helper functions

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
        itr = NtFileInterpreter(src_file)
        rtn = itr.interpret(True)
        print(rtn)
