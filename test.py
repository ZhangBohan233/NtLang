class LN:
    def __init__(self, head, tail=None):
        self.head = head
        self.tail = tail

    def __str__(self):
        return "[" + str(self.head) + "->" + str(self.tail) + "]"


def make_ele(ele):
    if isinstance(ele, list):
        return make_ln(ele, 0)
    else:
        return ele


def make_ln(lst, i):
    if i <= len(lst) - 1:
        return LN(make_ele(lst[i]), make_ln(lst, i + 1))
    else:
        return None


def reverse(lst: LN, t: LN):
    if lst is None:
        return t
    else:
        return reverse(lst.tail, LN(lst.head, t))


def heads(lists: LN):
    if lists:
        return LN(lists.head.head, heads(lists.tail))
    else:
        return None


def tails(lists: LN):
    if lists:
        return LN(lists.head.tail, tails(lists.tail))
    else:
        return None


def foldl(ftn, init, fl: LN):
    if fl is None:
        return init
    else:
        return foldl(ftn, ftn(fl.head, init), fl.tail)


def foldl2(ftn, init, lists: LN):
    if lists:
        hs = heads(lists)
        ts = tails(lists)
        return foldl2(ftn, apply(ftn, hs, init), ts)
    else:
        return init


def apply(func, lst: LN, extra):
    py_lst = []
    n = lst
    while n:
        py_lst.append(n.head)
        n = n.tail
    py_lst.append(extra)
    return func(*py_lst)


if __name__ == '__main__':

    aa = make_ln([1, 2, 3, 4], 0)

    # bb = reverse(aa, None)
    lst2 = make_ln([[1, 2, 7], [3, 4, 8]], 0)
    print(lst2)

    # print(bb)

    cc = foldl(lambda a, b: a + b, 0, aa)
    print(cc)
    print(foldl(LN, None, aa))

    print("heads", heads(lst2))
    print("tails", tails(lst2))
    dd = foldl2(lambda a, b, c: a + b + c, 0, lst2)
    print(dd)
