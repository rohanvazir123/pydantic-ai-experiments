import collections
import heapq
from typing import List

def heap_test():
  l = [5, 6, 3, 1, 2, 10, 4]
  heapq.heapify(l)
  print(f"list before n_largest: {list(l)}")
  n_largest = heapq.nlargest(2, l)
  print(f"list after n_largest: {list(l)}")
  n_smallest = heapq.nsmallest(2, l)
  print(f"n_smallest: {n_smallest}")

def find_median(l):
  # left_heap is max_heap, and right_heap is min_heap
  left_heap, right_heap, curr_median = [], [], None

  for i, e in enumerate(l):
    # If current median is not calculated yet, current median = first number
    if curr_median is None:
      curr_median = e
      #Simulate left_heap as max_heap, so need to negate the sign, so that max element is always at the top
      heapq.heappush(left_heap, -e)
      continue
    
    # next number is less than the current median, push it to to the left (max) heap
    if e < curr_median:
      heapq.heappush(left_heap, -e)
    else:
      # next number is greater than, =  the current median, push it to to the right (min) heap
      heapq.heappush(right_heap, e)

    l_len = len(left_heap)
    r_len = len(right_heap)
    
    #If the left_heap and right_heap are unbalanced (length of any heap is greater than the other by more than 1), rebalance them
    if (l_len > r_len + 1):
      #Flip the sign before pushing it to the heap
      heapq.heappush(right_heap, -heapq.heappop(left_heap))
    elif r_len > l_len + 1:
      heapq.heappush(left_heap, -heapq.heappop(right_heap))

    l_len = len(left_heap)
    r_len = len(right_heap)

    # Updating current median
    if l_len == r_len:
      curr_median = ((-left_heap[0]) + right_heap[0]) / 2
    elif l_len > r_len:
      curr_median = -left_heap[0]
    elif r_len > l_len:
      curr_median = right_heap[0]
    
  return curr_median



if __name__ == "__main__":
  #heap_test()

  #Find median from a list of numbers
  l = [10, 9, 11, 12, 7, 4]
  median = find_median(l)
  print(f"median: {median}")