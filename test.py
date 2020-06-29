class LN:
    def __init__(self, head, tail=None):
        self.head = head
        self.tail = tail

    def __str__(self):
        return str(self.head) + "->" + str(self.tail)


def make_ln(lst, i):
    if i <= len(lst) - 1:
        return LN(lst[i], make_ln(lst, i + 1))
    else:
        return None


def reverse(lst: LN, t: LN):
    if lst is None:
        return t
    else:
        return reverse(lst.tail, LN(lst.head, t))


if __name__ == '__main__':

    aa = make_ln([1, 2, 3, 4], 0)

    bb = reverse(aa, None)

    print(bb)
