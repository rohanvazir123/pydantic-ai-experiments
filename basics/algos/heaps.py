# heaps

# Python library

# Min-Heap - a complete binary tree
# any node n[i] is such that it is not more than its children n[2i+1] and n[2i+2]
# n[i] <= n[2i+1] and n[i] <= n[2i+2]

# min-heap
# import heapq
# h = [ 1, 2, 3, 4]
# inplace heapify
# heapq.heapify(h)
# heapq.nsmallest(n, h); heapq.nlargest(n, h) - the elements are sorted 
# heapq.pushheap(item)
# item = heapq.popheap()
# h[0] is the smallest item
# first push item1 to heap and pop the smallest item2
# item2 = heapq.pushpopheap(item1)
# first pop smallest item1 and then push item1
# item1 = heapq.replaceheap(item2)
# merge sorted arrays (pass in iterators) which are a list of sorted lists
# heapq.merge(*sorted_arrars)

from typing import List, Tuple

def heapify(L, root):
  left = 2*root+1
  right = 2*root+2
  smallest = root
  if left < len(L) and L[smallest] > L[left]:
    smallest = left
  if right < len(L) and L[smallest] > L[right]:
    smallest = right
    
  if smallest != root:
    L[root], L[smallest] = L[smallest], L[root]
  
  return heapify(L, smallest)

# sort descending (not ascending)
def build_heap(L, root):
  for i in range(len(L)//2 -1, -1, -1):
    heapify(L, i)


# Pythonic solution for merging sorted arrays
def merge_sorted_arrays_pythonic(sorted_arrays):
  return list(heapq.merge(*sorted_arrays))

# Find k smallest elements by merging sorted arrays 
def k_smallest_from_sorted_arrays(sorted_arrays:List[int], k):
  
  min_heap : List(Tuple[int, int]) = []

  # Make sorted array iterators for each array in sorted arrays
  sorted_arrays_iters = [ iter(x) for x in sorted_arrays]

  # Put first element from each iterator in min heap
  for i, it in enumerate(sorted_arrays_iters):
    if first_element := next(it, None):
      heapq.heappush(min_heap,  (first_element, i) )

  # Output list to hold k smallest
  k_smallest : List[int] = []

  # Remove successive elements from sorted array iterators
  # as they produce the smallest element
  while min_heap and k:
    smallest_elem, smallest_iterator_index = heapq.heappop(min_heap)
    k_smallest.append(smallest_elem)
    k -= 1
    smallest_element_iter = sorted_arrays_iters[smallest_elem_index],
    if next_element_from_smallest_element_iter := next(smallest_element_iter, None):
      heapq.heapush(min_heap, (next_element_from_smallest_element_iter, smallest_iterator_index))
      
  return k_smallest

import heapq

class HeapNode(object):
  def __init__(self, val: int):
    self.val = val

  def __repr__(self):
    return f'HeapNode value: {self.val}'

  def __lt__(self, other):
    return self.val < other.val
      
if __name__ == "__main__":
  h1 = HeapNode(5)
  h2 = HeapNode(6)
  print(h2, h1)
  print(h2 < h1)
