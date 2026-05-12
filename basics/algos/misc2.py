import itertools, functools

def add_digits_string(s:str):
    # itertools and reduce
    l= list(itertools.chain(s))
    x= [*s]
    print(x, type(x))
    sum_ = functools.reduce(lambda sum, y : sum+int(y), l, 0)
    print(f"sum {sum_}")
    sum_ = functools.reduce(lambda sum, y : sum+int(y), x, 0)
    print(f"sum {sum_}")
    return

def random_crap_using_star(s):
    # https://stackoverflow.com/questions/2921847/what-do-double-star-asterisk-and-star-asterisk-mean-in-a-function-call
    d = { "one":1, "two": 2}
    x = {**d}
    print(sum(map(ord, [*s])))
    return

def crap_generator():
    for i in range(1, 4):
        yield i

if __name__ == "__main__":
    import traceback

    print("main")
    add_digits_string("123")
    random_crap_using_star("priya")

    g = crap_generator()
    n = next(g, None)
    print(n)

    n = next(g, None)
    print(n)

    n = next(g, None)
    print(n)

    try: 
        n = next(g)
    except StopIteration as e:
        print("bad boy!")
        traceback.print_exc()
    print(n)