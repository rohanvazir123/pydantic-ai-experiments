
# Basic Binary search - bin search always works on sorted Aays
# Return index of the first match, otherwise return -1
def bin_search(A: list[int], num :int) -> int:
    left, right, result = 0, len(A) - 1, -1
    while (left <= right):
        mid = (left+right) // 2 # note this can overflow, it should be left + (right-left) // 2
        if num == A[mid]:
            result = mid
            break
        elif num < A[mid]:
            right = mid-1
        else:
            left = mid+1
    return result

# Binary search with repeated elements in the Aay
# Return index of the first element found, otherwise return -1
def bin_search_first_hit(A: list[int], num :int) -> int:
    left, right, result = 0, len(A) - 1, -1
    while (left <= right):
        mid = (left+right) // 2 # note this can overflow, it should be left + (right-left) // 2
        if num == A[mid]:
            result = mid # result is mid, found a possible match
            right = mid  - 1 # keep looking on the left
        elif num < A[mid]:
            right = mid-1
        else:
            left = mid+1
    return result

# Binary search with repeated elements in the Aay
# Return index of the first greater element found (or last equal element + 1), otherwise -1.
# The idea is to find the last equal element (the next one if exists will be greater)
def bin_search_greater(A: list[int], num :int) -> int:
    left, right, result = 0, len(A) - 1, -1
    while (left <= right):
        mid =  left + (right-left) // 2 # note this can overflow, it should be left + (right-left) // 2
        if num == A[mid]:
            result = mid # result is mid, found a possible match
            left = mid  + 1 # keep looking on the right
        elif num < A[mid]:
            right = mid-1
        else:
            left = mid+1
    result += 1
    if result >= len(A): return -1
    # TODO get rid of this redundant check
    if A[result] > num: return result
    return -1

# Basic Binary search - bin search always works on sorted Arrays
def bin_search_rotated(A: list[int], num :int) -> int:
    left, right, result = 0, len(A) - 1, -1
    while (left <= right):
        mid = left + (right-left) // 2
        # print(A[left], A[mid], A[right])
        if A[mid] == num:
            result = mid
            break
        if A[left] <= A[mid]: # note <= here
            # left is sorted
            if num >= A[left] and num < A[mid]: # note >=
                right = mid-1
            else:
                left = mid+1
        else:
            # right must be sorted
            if num > A[mid] and num  <= A[right]: # note <=
                left = mid+1
            else:
                right = mid-1
    return result


# 17	-4	0	3	27	118	225	250	704	760 

def test_bin_search():
    print("test_bin_search")
    A = [1, 2, 5, 8, 10, 24, 44, 55]

    index = bin_search_first_hit(A, 44)
    print(f"44, {index}, {A[index] if index !=-1 else None}")

def test_bin_search_first_hit():
    print("test_bin_search_first_hit")

    A = [1, 2, 5, 8, 10, 24, 34, 34, 44, 44, 55]

    index = bin_search_first_hit(A, 44)
    print(f"44, {index},  {A[index] if index !=-1 else None}")

    index = bin_search_first_hit(A, 34)
    print(f"34, {index},  {A[index] if index !=-1 else None}")


def test_bin_search_greater():
    print("test_bin_search_greater")
    A = [1, 2, 5, 8, 10, 24, 34, 34, 44, 44, 55]

    index = bin_search_greater(A, 8)
    print(f"8, {index}, {A[index] if index !=-1 else None}")

    index = bin_search_greater(A, 34)
    print(f"34, {index}, {A[index] if index !=-1 else None}")

    index = bin_search_greater(A, 55)
    print(f"55, {index},  {A[index] if index !=-1 else None}")

def test_bin_search_rotated():
    print("test_bin_search_rotated")
    A = [15, 24, 34, 44, 49, 56, 1, 2, 3, 4]

    index = bin_search_rotated(A, 15)
    print(f"15, {index},  {A[index] if index !=-1 else None}")

    index = bin_search_rotated(A, 44)
    print(f"44, {index},  {A[index] if index !=-1 else None}")

    index = bin_search_rotated(A, 34)
    print(f"34, {index},  {A[index] if index !=-1 else None}")

    index = bin_search_rotated(A, 3)
    print(f"3, {index},  {A[index] if index !=-1 else None}")

    index = bin_search_rotated(A, 24)
    print(f"24, {index},  {A[index] if index !=-1 else None}")

    index = bin_search_rotated(A, 43)
    print(f"43, {index}, {A[index] if index !=-1 else None}")

    index = bin_search_rotated(A, 1)
    print(f"1, {index},  {A[index] if index !=-1 else None}")

    if __name__ == '__main__':
        test_bin_search()
        test_bin_search_first_hit()
        test_bin_search_greater() # first greater
        test_bin_search_rotated()
