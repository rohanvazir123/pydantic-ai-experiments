# function to check if a substring 
# s[low..high] is a palindrome
def checkPal(str, low, high):
    while low < high:
        if str[low] != str[high]:
            return False
        low += 1
        high -= 1
    return True

# function to find the longest palindrome substring
def getLongestPal(s):
    
    n = len(s)

    # all substrings of length 1 are palindromes
    maxLen = 1
    start = 0

    # nested loop to mark start and end index
    for i in range(n):
        for j in range(i, n):

            # check if the current substring is 
            # a palindrome
            if checkPal(s, i, j) and (j - i + 1) > maxLen:
                start = i
                maxLen = j - i + 1

    return s[start:start + maxLen]

if __name__ == "__main__":
    s = "forgeeksskeegfor"
    print(getLongestPal(s))