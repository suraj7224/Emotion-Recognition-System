res = "abc"


def myfunc(x=None):
    global res
    if x is not None:
        res = x

print(res)
myfunc()

print(res)
