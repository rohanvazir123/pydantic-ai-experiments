# Copyright 2024 The Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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