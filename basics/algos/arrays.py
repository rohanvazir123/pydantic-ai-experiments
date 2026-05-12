from typing import List
import sys

"""
A subsequence is a sequence that can be derived from an array 
by deleting some or no elements without changing the order of the remaining elements. 
For example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].
There are multiple variations of longest subsequence problems.
1. Longest common subsequence
2. Longest increasing consecutive subsequence
3. Longest increasing contiguous subsequence, contiguous means adjacent to each other
3b. Longest increasing subsequence

4. Maximum subarray
5. 3 sum
"""

"""
Longest common subsequence
nC1 + nC2 + … nCn = 2n are the possible subsequences of an array with 'n' elements.
The time complexity of brute force algorithm is, therefore, 0(n*2^n).
Add to that O(n) time to check if a subsequence is common to both the strings.
This can be improved by dynamic programming.

1) Optimal Substructure: 
Let the input sequences be A[0..m-1] and B[0..n-1] of lengths m and n respectively. 
And let L(X[0..m-1], Y[0..n-1]) be the length of LCS of the two sequences X and Y. 

Following is the recursive definition of L(A[0..m-1], B[0..n-1]).

If last characters of both sequences match (or A[m-1] == B[n-1]) then 
L(A[0..m-1], B[0..n-1]) = 1 + L(A[0..m-2], B[0..n-2])

If last characters of both sequences do not match (or A[m-1] != B[n-1]) then 
L(A[0..m-1], B[0..n-1]) = MAX ( L(A[0..m-2], B[0..n-1]), L(A[0..m-1], B[0..n-2]) )
"""


def find_len_longest_common_subsequence(A: list[int], B: list[int]) -> list[list[int]]:
    print(" find_len_longest_common_subsequence")
    print(f"A {A} B {B}")
    # Note the initializations carefully

    # Length of arrays
    m = len(A)
    n = len(B)

    # Initialize 2d dynamic programming array
    # note: m+1 and n+1 here, as we [i][0] and [0][j] are all zeroes
    # when zero elements, lcs len is 0 (obviously), we need an extra row and col to store that
    L = [[0] * (n + 1) for i in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if A[i - 1] == B[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    return L


# given precomputed 2d lcs lenghts, return the actual lcs (string)
def longest_common_subsequence(A: list[int], B: list[int], L: list[list[int]]) -> str:
    m, n = len(A), len(B)

    index = L[m][n]
    lcs: list[str] = [""] * index
    i, j = m, n
    while i > 0 and j > 0:
        if A[i - 1] == B[j - 1]:
            lcs[index - 1] = str(A[i - 1])  # pick the elem
            i -= 1
            j -= 1
            # Note carefully that we decrement index only here
            index -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            # (i-1)th index contributed to LCS
            i -= 1
        else:
            # (j-1)th index contributed to LCS
            j -= 1

    return "".join(lcs)


"""
The Longest Increasing Subsequence (LIS) problem is to find the length of the longest subsequence of a given sequence 
such that all elements of the subsequence are sorted in increasing order. 
For example, the length of LIS for {10, 22, 9, 33, 21, 50, 41, 60, 80} is 6 and LIS is {10, 22, 33, 50, 60, 80}. 
If we closely observe the problem then we can convert this problem to longest Common Subsequence Problem. 
Firstly we will create another array of unique elements of original array and sort it. 
Now the longest increasing subsequence of our array must be present as a subsequence in our sorted array. 
That’s why our problem is now reduced to finding the common subsequence between the two 
"""


def longest_increasing_subsequence(A: list[int], B: list[int], L: list[list[int]]) -> list[int]:
    m, n = len(A), len(B)
    i, j = m, n
    index = L[m][n]
    lcs: list[int] = [0] * index
    while i > 0 and j > 0:
        if A[i - 1] == B[j - 1]:
            lcs[index - 1] = A[i - 1]
            i -= 1
            j -= 1
            # Note carefully that we decrement index only here
            index -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs


"""
Longest increasing consecutive subsequence.

Input : a[] = {3, 10, 3, 11, 4, 5, 6, 7, 8, 12} 
Output : 6 
Explanation: 3, 4, 5, 6, 7, 8 is the longest increasing subsequence whose adjacent element differs by one. 

Input : a[] = {6, 7, 8, 3, 4, 5, 9, 10} 
Output : 5 
Explanation: 6, 7, 8, 9, 10 is the longest increasing subsequence 

Longest increasing consecutive subsequence
In this we maintain an array L, where L[i] stores the length of the longest increasing subsequence for index 'i'.
We also maintain element_to_index dictionary map A[i]-1 element to it's index 
  Let L[i] store the length of the longest subsequence which ends with A[i].
  Now the following property holds, for every A[i], 
  if A[i]-1 is present in the array, 
  then 
    L[i]= element_to_index [A[i]-1 ] + 1
  else]
    L[i] = 1, as we have a new subsequence starting with A[i]
  In order to print the elements, all we need to maintain is the end index of the Longest increasing consecutive subsequence,
  besides the len of the longest increasing subsequence.
"""


def print_int_array(A: list[int], start, end):
    for x in range(start, end):
        print(A[x], end=" ")
    return


def print_longest_increasing_consecutive_subsequence(
    A: list[int], longest_len: int, max_index: int
):
    print(" print_longest_increasing_consecutive_subsequence()")
    start = A[max_index] - longest_len + 1
    end = A[max_index] + 1  # Note in python, [start, end)
    for x in range(start, end):
        print(x, end=" ")
    print()
    return


def find_len_longest_increasing_consecutive_subsequence(A: list[int]) -> tuple:
    print(" find_len_longest_increasing_consecutive_subsequence()")
    # Note the initializations carefully

    # Length of array
    n = len(A)

    # Initializa dynamic programming array
    L = [0] * n

    longest_len = -sys.maxsize - 1
    max_index = -1

    # Map element to it's index
    element_to_index: dict[int, int] = {}

    for i in range(n):
        if A[i] - 1 in element_to_index:
            last_index = element_to_index[A[i] - 1]
            L[i] = L[last_index] + 1
        else:
            L[i] = 1
        element_to_index[A[i]] = i

        if L[i] > longest_len:
            longest_len = L[i]
            max_index = i
    return longest_len, max_index


"""
Longest increasing contiguous subsequence.
Example 1:
Input: nums = [1,3,5,4,7]
Output: 3
Explanation: The longest continuous increasing subsequence is [1,3,5] with length 3.
Even though [1,3,5,7] is an increasing subsequence, it is not continuous as elements 5 and 7 are separated by element 4.

Example 2:
Input: nums = [2,2,2,2,2]
Output: 1
Explanation: The longest continuous increasing subsequence is [2] with length 1. 
Note that it must be strictly increasing.

The logic for this is simple as the elements must be one after the other (contiguous) and increasing.
In this case, keep comparing A[i] with A[i-1] until A[i] becomes less than A[i-1] and set the stop_index to i
As we iterate through the array, always calculte max_len_so_far = max(max_len_so_far, i - stop_index  + 1 )
In order to print the elements, we just print all elements between stop_index and stop_index+max_len_so_far + 1
"""


def find_len_longest_contiguous_increasing_consecutive_subsequence(A: list[int]) -> int:
    # Note the initializations carefully

    # Length of array
    n = len(A)

    max_len_so_far = stop_index = 0

    for i in range(n):
        if i and A[i] < A[i - 1]:
            stop_index = i
        max_len_so_far = max(max_len_so_far, i - stop_index + 1)

    return max_len_so_far


"""
Maximum sum subarray (Kadane's algorithm)
The maximum subarray problem is the task of finding the largest possible sum of a contiguous subarray, 
within a given one-dimensional array A[1…n] of numbers.
Kadane's sliding window algorithm
  - keep a local maximum (subarray sum) at each index and global_maximum = max(local_maximum, global_maximum)
  - local_maximum = max(A[i], A[i] + local_maximum)
"""


def find_max_sum_subarray(A: list[int]) -> int:
    n = len(A)
    local_max = 0
    global_max = -sys.maxsize
    for i in range(n):
        local_max = A[i] + local_max
        global_max = max(local_max, global_max)
        # reset local_max to 0 if it turns negative and start building it all over again
        local_max = 0 if local_max < 0 else local_max
    return global_max


### miscellaneous ###
"""
3 sum
Given an integer array nums, 
return all the triplets [nums[i], nums[j], nums[k]] 
such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
Notice that the solution set must not contain duplicate triplets.

Example 1:
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]

Example 2:
Input: nums = []
Output: []

Example 3:
Input: nums = [0]
Output: []
"""

"""
These 2 sum and 3 sum problems are not very hard to do if you maintain a hash
"""


def two_sum(A, sum):
    n = len(A)
    s = set()
    for i in range(n):
        if sum - A[i] in s:
            return (A[i], sum - A[i])
    s.add(A[i])
    return ()


# This implementation sems wrong
def three_sum(A, sum):
    n = len(A)
    s = set()
    for i in range(0, n):
        for j in range(i + 1, n):
            if sum - A[i] - A[j] in s:
                return (A[i], A[j], sum - A[i] - A[j])
        # Be careful here, note that you have to add A[j]
        # you cannot add sum-A[i]-A[j]as it *MAY* not be in A
        s.add(A[j])
    return ()


# This one is right
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        def _two_sum(nums: List[int], start_idx: int, sum: int) -> set:
            triplets = set()
            lookup_table = set()
            for i in range(start_idx, len(nums)):
                if sum - nums[i] in lookup_table:
                    l = [nums[i], sum - nums[i], -sum]
                    l.sort()
                    t = tuple(l)
                    if t not in triplets:
                        triplets.add(t)
                else:
                    lookup_table.add(nums[i])
            return triplets

        dedup_ = set()
        result = set()
        for i in range(len(nums)):
            if nums[i] not in dedup_:
                dedup_.add(nums[i])
                if triplets := _two_sum(nums, i + 1, -nums[i]):
                    result.update(triplets)
        return list(result)


# Note a variation of the above would be find ALL 2 SUM/3 SUM triplets

# Stock buy and sell problem - two variants : once and twice

from typing import List, Tuple


def buy_and_sell_stock(stock_prices: List[float]) -> Tuple[float, List[float]]:
    max_profit = 0.0
    min_buy_price_so_far = float("inf")
    max_profit_per_day = [0.0] * len(stock_prices)
    for i, price_today in enumerate(stock_prices):
        min_buy_price_so_far = min(price_today, min_buy_price_so_far)
        max_profit_if_sold_today = price_today - min_buy_price_so_far
        max_profit = max(max_profit_if_sold_today, max_profit)
        max_profit_per_day[i] = max_profit
    return max_profit, max_profit_per_day


# Partition array into elements
# less than left and greater than right
def partition_array(l: List[int], left: int, right: int):
    start = 0
    for i in range(len(l)):
        if l[i] <= left:
            l[start], l[i] = l[i], l[start]
            start += 1
    end = len(l) - 1
    for j in range(end, start - 1, -1):
        if l[j] >= right:
            l[end], l[j] = l[j], l[end]
            end -= 1
    return l


# Buy and sell stock twice
def buy_and_sell_stock_twice(stock_prices: List[float]) -> float:
    # first buy-sell
    max_profit, max_profit_per_day_first_buy_sell = buy_and_sell_stock(stock_prices)
    # second buy-sell

    max_sell_price_so_far = float("-inf")
    max_total_profit = 0.0

    # find the max profit from i to n-1 days, when n = len(stock_prices)
    # here 'i' divides the days into two parts, in the first parts, we do first buy-sell
    # in the second part, the second buy-sell
    for i, price_today in reversed(list(enumerate(stock_prices))):
        max_sell_price_so_far = max(max_sell_price_so_far, price_today)
        max_profit_if_bought_today = max_sell_price_so_far - price_today
        max_total_profit = max(
            max_profit_if_bought_today + max_profit_per_day_first_buy_sell[i],
            max_profit,
        )
    return max_total_profit


def test_find_len_longest_increasing_consecutive_subsequence():
    print()
    print("test_find_len_longest_increasing_consecutive_subsequence")
    A = [2, 3, 10, 4, 11, 4, 6, 7, 8, 9, 1, 2, 10, 11, 12]

    # test find_len_longest_increasing_consecutive_subsequence
    longest_len, max_index = find_len_longest_increasing_consecutive_subsequence(A)
    print_longest_increasing_consecutive_subsequence(A, longest_len, max_index)


def test_find_len_longest_common_subsequence():
    print()
    print("test_find_len_longest_common_subsequence")
    s1 = "AGGTAB"
    s2 = "GXTXAYB"
    A = list(s1)
    B = list(s2)

    L = find_len_longest_common_subsequence(A, B)
    print(f"LCS len is {L[len(A)][len(B)]}")
    lcs = longest_common_subsequence(A, B, L)
    print(f"LCS is {lcs}")
    return


def test_find_len_longest_increasing_subsequence():
    print()
    print("test_find_len_longest_increasing_subsequence")
    A = [50, 3, 10, 7, 40, 80]
    # A = [10, 11, 22, 9, 11, 33, 21, 50, 41, 60]

    B = sorted(set(A))
    L = find_len_longest_common_subsequence(A, B)
    print(f"LIS len is {L[len(A)][len(B)]}")
    print(f"LIS is {longest_increasing_subsequence(A, B, L)}")
    return


def test_find_max_sum_subarray():
    print()
    print("test_find_max_sum_subarray")
    A = [-200, 3, 7, -10, 4, 45, -11, 4, 6, 9, 101]
    print(f"Max sum subarray {find_max_sum_subarray(A)}")


def test_three_sum():
    print()
    A = [1, 4, 45, 6, 10, 8]
    sum = 22
    print(f"3sum {three_sum(A, sum)}")


def test_partition_array():
    print()
    l = [3, 19, 17, 4, 6, 8, 21, 13, 9, 11, 1]
    print(l)
    out_list = partition_array(l, 4, 10)
    print(out_list)


if __name__ == "__main__":
    test_find_len_longest_common_subsequence()
    test_find_len_longest_increasing_subsequence()
    test_find_len_longest_increasing_consecutive_subsequence()
    test_find_max_sum_subarray()
    test_three_sum()
    test_partition_array()
