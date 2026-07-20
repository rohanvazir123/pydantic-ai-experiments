def permutate1(a, left, right):
    result = []
    if left == right:
        return ["".join(a)]

    for i in range(left, right):
        a[i], a[left] = a[left], a[i]
        result.extend(permutate1(a, left+1, right))
        a[i], a[left] = a[left], a[i]

    return result

def permutate2(a, left, right):
    result = []
    if left == right:
        # return the entire list CONVERTED to a string
        return ["".join(a)]

    # Track characters already used at this position in this stack frame
    visited_at_level = set()

    for i in range(left, right):
        if a[i] in visited_at_level:
            continue
        visited_at_level.add(a[i])  # Mark character as used 
        a[i], a[left] = a[left], a[i]
        result.extend(permutate2(a, left + 1, right))
        a[i], a[left] = a[left], a[i]  # Backtrack

    return result

def permutate3(a, left, right):
    result = []
    status = False
    if  left-2 >0 and a[left-2] < a[left-1]:
        a[left-1], a[left-2] = a[left-2], a[left-1]
        status = True
    if left == right:
        return ["".join(a)], status

    for i in range(left, right):
        a[i], a[left] = a[left], a[i]
        ret, status = permutate3(a, left+1, right)
        result.extend(ret)
        a[i], a[left] = a[left], a[i]
        if status:
            return result, status


    return result, status


if __name__ == "__main__":
    import pprint
    a = "aac"

    # with dups
    pprint.pprint(permutate1(list(a), left=0, right=len(a)))

    # no dups
    pprint.pprint(permutate2(list(a), left=0, right=len(a)))

    pprint.pprint(permutate1(list("12"), left=0, right=2))


    print("")
    l = ["123", "213", "231" , "312", "321", "12", "21", "2"]
    print(l)
   
    for e in l:
        print("input: ", e)
        print(permutate3(list(e), left=0, right=len(e)))
        print("")

