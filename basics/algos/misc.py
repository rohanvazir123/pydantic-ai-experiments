import string
punct = set(string.punctuation)
print(punct)

my_list = [ 1, 2, 3, 4, 5]
print(my_list[-1:-3:-1])

import string
string.digits.index('9')

d = { "one":1, "two" : 2, "three": 3 }
d.pop("one")
l = [1, 2, 3, 4, 5]
l.pop(0)
print(l)
l.pop()
print(l)

int('9')
string.hexdigits.index('f')

hex(ord('b'))
help(ord)
hex(ord('A'))
ord('b')-ord('a')
chr(ord('A')+1)

l = " ".join(["hello", "world"])
print(l)
l = "_".join("hello")
print(l)

import itertools
s = ((1, "one"), (1, "onesy"), (2, "two"), (2, "twos"), (3,"three"))

for key, group in itertools.groupby(s, lambda x : x[0]):
  print(key, len(list(group)))

from collections import Counter
for k,v in Counter("gogoraram").items ():
  print(k,v)
s = sum(Counter("gogoraram").values ())
print(s)

m = map(str.lower, filter(str.isalpha, "h1elloWor123ld"))
"".join(m)

l = [1, 2, 3, 5, 6, 7, 9, 11, 12]
c = list(filter(lambda x: x%2 != 0, l))
len(c)

l1 = [1, 2, 3]; l2 = [3, 4,7, 7]; l3 = [5, 6, 7]
m = map(lambda x, y, z: x+y+z, l1, l2, l3)
l = list(m)
print(l)

f = filter(lambda x: x%3 == 0, l)
l = list(f)
print(l)
from functools import reduce
r = reduce(lambda x, y: x+y, l)
print(r)

l = [ i for i in range(5) ]
print(l)
[ i for i in reversed(range(5))]

l = [ e for e in range(1,10)]
print(l)
for i, e in reversed(list(enumerate(l[1:], 1))):
  print(i,e, end=" ")

a = [[0]*4]*10
a
l = [ i for i in reversed(range(1, 101)) ]
print(l)

import re
string = 'bobby246'
match = re.search(r'\d+$', string)
if match is not None:
    print('The string ends with a number')

    print(match.group())  # 👉️ '246'
    print(int(match.group()))  # 👉️ 246

    print(match.start())  # 👉️ 5

else:
    print('The string does NOT end with a number')

from ast import Lambda
from typing import Iterable
import itertools

def grp_by(s: Iterable, g):
  print("\ngrp_by")
  for k,v in g:
    print (k, list(v))
s  = "aabbbcccccddeeffg"

print(type(itertools.groupby(s)))
print(type(grp_by))

grp_by(s, itertools.groupby(s))
a_list = [("Animal", "cat"), 
          ("Animal", "dog"), 
          ("Bird", "peacock"), 
          ("Bird", "pigeon")]
grp_by(a_list, itertools.groupby(a_list, lambda x : x[0]))


import heapq
h = []
heapq.heappush(h, ("one", 35))
heapq.heappush(h, ("two", 1))
heapq.heappush(h, ("three", 2))
heapq.heappush(h, ("hone", 30))

print(heapq.nlargest(2,h))
print(heapq.nsmallest(2,h))

heapq.heappop(h)

a = [1, 2, 3]
b = a[:]
print(id(a), id(b))
a1 = [ a, a:=a+ [(a[-1]+1)], a:=a+ [(a[-1]+1)]]
print(a1)
b1 = a1[:]
print(id(a1), id(b1), id(a1[0]), id(b1[0]))


from heapq import *
h = [0, 1, 2]
print(heappushpop(h, -1))
print(h)

h = [0, 1, 2]
print(heapreplace(h, -1))
print(h)

import itertools
l =  [1, 2, 3, 4]
it = iter(l)
it = itertools.islice(it, 3)
print(next(it, None))
print(next(it, None))
print(next(it, None))
print(next(it, None))
print(next(it, None))
print(next(it, None))

import itertools
l1 = [ 1, 3, 5]
l2 = [ 2, 4, 6 ]
it1, it2 = iter(l1), iter(l2)
it = itertools.chain(it1, it2)
print(next(it, -1)); print(next(it,-1)); print(next(it, -1))
print(next(it, -1)); print(next(it,-1)); print(next(it, -1))
# note: you cannot have a generator go reverse in python

import collections
Student = collections.namedtuple("Student", ("name", "gpa"))
s1 = Student("s1", "0")
s2 = Student("s2", "2.5")
l = [ s1, s2 ]
print(l)

x = s1+s2; print(x)

def foo(*args):
  print(type(args))
  for i in args:
    print(i)
  return

l = [1, 2, 3]
print(*l)
foo(*l)



l1 = [1, 2, 3]
l2 = [4, 5]
myiters = [ iter(x)for x in [l1, l2] ]
print(myiters)
o = [ y for _ in range(len(l1)+len(l2)) for it in myiters if (y:=next(it, None)) is not None ]
print(o)

# Remove None from a list
l = ['pram', 'priya', None, 'vik', None]
res = list(filter(lambda item: item is not None, l))
print(f"Filtered list: {res}")

d = set(filter(lambda item: item is not None, l))
print(f"Filtered set: {d}")

from functools import reduce
print(reduce(lambda x,y: x**2 + y**2, [1, 2, 3, 4, 5]))

print(sum(x*x for x in [1, 2, 3, 4]))

print(reduce(lambda a,b: a+b, map(lambda x: x**2, [1, 2, 3, 4])))

import bisect

x = [ 10, 20, 30, 35, 40, 45, 50, 400, 500, 900, 1000]
print(bisect.bisect_left(x, 30))
print(bisect.bisect_left(x, 32))
print(bisect.bisect_left(x, 55))
print(bisect.bisect_left(x, 5))
print(bisect.bisect_left(x, 594))
print(5//2, 5/2, 3*'a')
values =  [  1,   4,    5,   9,   10,   40,  50,    90,  100,  400, 500,  900,  1000 ]
symbols = [ 'I', 'IV', 'V', 'IX', 'X', 'XL', 'L',  'XC', 'C', 'CD', 'D', 'CM', 'M'   ]
sym_to_val = dict(zip(symbols, values))
print(sym_to_val)
s = 'hello1'
print(s[0:1])

strs = ["flower","flow","flight"]
shortest_str = strs[0]
for s in strs:
  if len(s) < len(shortest_str):
    shortest_str = s
print(shortest_str)


A = ['1', '2', '3']
B = ['5', '6', '7']
#A.append(B)
A.extend(B)
print(A)

list_of_str = ['ab', 'cd' ,'ef']
str_to_append = "pqr"


def perm_list(list_of_str, str_to_append):
  for s in list_of_str:
    for c in str_to_append:
      pass