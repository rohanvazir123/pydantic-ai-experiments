from typing import List, Dict
from collections import namedtuple

# TODO
# elements are increasing but not necessarily sequentially
# a, b, c, d then if b,c then c>b but c may not be b+1
def longest_contiguous_subarray(L:List[int]) -> dict:
  print(f"\nlongest_contiguous_subarray L {L}", end=" => ")
  max_sa = {"sum" : 0, "seq" :[] }
  for e in L:
    if e+max_sa["sum"] > max_sa["sum"]:
      max_sa["sum"] += e
      max_sa["seq"].append(e)
  return max_sa

# TODO
def longest_increasing_subarray(L:List[int]) -> dict:
  print(f"longest_increasing_subarray L {L}", end=" => ")
  max_sa = {"sum" : 0, "seq" :[] }
  for e in L:
    if e+max_sa["sum"] > max_sa["sum"]:
      max_sa["sum"] += e
      max_sa["seq"].append(e)
  return max_sa

def max_sum_contiguous_subarray(L:List[int]) -> dict:
  print(f"max_sum_contiguous_subarray L {L}", end=" => ")
  max_csa_global = {"sum" : 0, "seq" :[] }
  max_csa_local = {"sum" : 0, "seq" :[] }
  for e in L:
    if e >= 0:
      max_csa_local["sum"] += e
      max_csa_local["seq"].append(e)

    else:
      if max_csa_local["sum"] > max_csa_global["sum"]:
        max_csa_global["sum"] =max_csa_local["sum"]
        max_csa_global["seq"] =max_csa_local["seq"]
      # reset
      max_csa_local["sum"] = 0
      max_csa_local["seq"] = []

    if max_csa_local["sum"] > max_csa_global["sum"]:
      max_csa_global["sum"] =max_csa_local["sum"]
      max_csa_global["seq"] =max_csa_local["seq"]
  return max_csa_global

def max_sum_subarray(L:List[int]) -> dict:
  print(f"max_sum_subarray L {L}", end=" => ")
  max_sa = {"sum" : 0, "seq" :[] }
  for e in L:
    if e+max_sa["sum"] > max_sa["sum"]:
      max_sa["sum"] += e
      max_sa["seq"].append(e)
  return max_sa


# Find the length of the longest subarray without repeating characters.
# Returns the length and the actual subarray
import copy
from typing import Tuple
def longest_subarray_non_repeating(s: str) -> Tuple[int, List[int]]:
  if not s:
    return 0, []
  global_max_len = 0
  global_max_subarray = []
  local_max_len = 0
  substr_list = []
  for i, e in enumerate(s):
    if e not in substr_list:
      local_max_len += 1
      substr_list.append(e)
    else:
      if local_max_len > global_max_len:
        global_max_len = local_max_len
        global_max_subarray = copy.deepcopy(substr_list)
       
      last_index = len(substr_list) - 1 - substr_list[::-1].index(e)
      substr_list = substr_list[last_index+1:]
      substr_list.append(e)
      local_max_len = len(substr_list) 

    if local_max_len > global_max_len:
      global_max_len = local_max_len
      global_max_subarray = copy.deepcopy(substr_list)
  
  return global_max_len, global_max_subarray

#Leetcode #3
#Longest Substring Without Repeating Characters
#Given a string s, find the length of the longest substring without repeating characters.
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        global_max_len = 0
        local_max_len = 0
        substr_list = []
        for e in s:
            if e not in substr_list:
                local_max_len += 1
                substr_list.append(e)
            else:
                global_max_len = max(local_max_len, global_max_len)
                repeat_index = substr_list.index(e)
                substr_list = substr_list[repeat_index+1:]
                substr_list.append(e)
                local_max_len = len(substr_list)  
        global_max_len = max(local_max_len, global_max_len)
        return global_max_len
  
def convert(s: str, numRows: int) -> str:
  num_rows = numRows
  num_cols = len(s)//numRows
  l = len(s)
  zigzag =   [ [0] * num_cols  ] * num_rows
  for i in range(num_rows):
    for j in range(num_cols):
      print(j, i)
      if j<=num_rows:
        zigzag[j][i] = s[j]
      else:
        zigzag[j+1][i-1] = s[j]
  print(zigzag)

def two_sum(l, sum):
  output = []
  s = set()
  for i in range(len(l)):
    if (sum - l[i]) in s:
      output.append((l[i], sum - l[i]))
    else:
      s.add(l[i])
  return output
  
if __name__ == '__main__':

  # Two sum
  l = [2, 1, 3, 4, 5]
  print(f"Two sum pairs: {two_sum(l, 7)}") 
  l1 = [5, 6, 11, 8, 3, 2, 20, 9]
  print(f"Two sum pairs: {two_sum(l1, 11)}") 
  
  ## max subaray
  print(max_sum_subarray([-1]))
  print(max_sum_subarray([-1, 1]))
  print(max_sum_subarray([-1, 1, 2, 3, -4, 0, 5]))
  print(max_sum_subarray([17, 0, 2, -1, 30, 0, -17, 32]))
  print(max_sum_subarray([1, 2, 3, 4]))
  print(max_sum_subarray([-1, -2, -3, -4]))
  print(max_sum_subarray([-1, 2, 3, -4]))

  ## max contiguous subarray
  print()
  print(max_sum_contiguous_subarray([-1]))
  print(max_sum_contiguous_subarray([-1, 1]))
  print(max_sum_contiguous_subarray([-1, 1, 2, 3, -4, 0, 5]))
  print(max_sum_contiguous_subarray([17, 0, 2, 30, -1, 0, -17, 32]))
  print(max_sum_contiguous_subarray([1, 2, 3, 4]))
  print(max_sum_contiguous_subarray([-1, -2, -3, -4]))

  print()
  #convert('PAYPALISHIRING', 3)
  