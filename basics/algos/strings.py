# backtracking  - permutations of a string
def print_string_permutations(s:list) -> None:
  def permutate(s, left, right, count):
    if left == right:
      count += 1
      print("".join(s))
      return count
    for i in range(left, right):
      # swap
      s[i], s[left] = s[left], s[i]
      count = permutate(s, left+1, right, count)
      # restore the swapped elements
      s[i], s[left] = s[left], s[i]
    return count
  
  # begin
  return permutate(s, 0, len(s), 0)

def apply_permute(perm:list[int], l:list) -> None:
  for i in range(len(l)):
    while perm[i] != i:
      print(i, perm[i])
      l[i], l[perm[i]] = l[perm[i]], l[i]
      a = perm[i]
      perm[a], perm[i] = perm[i], perm[a]
       #perm[perm[i]], perm[i] = perm[i], perm[perm[i]]
      #perm[i], perm[perm[i]] = perm[perm[i]], perm[i]

def count_unique_chars(s):
  d = [ 0 ] * 255
  for i in s:
    d[ord(i)] += 1

  for i,e in enumerate(d):
    if e: print(chr(i), e)

  print('\n')
  for i in range(0, 256):
    print(chr(i))
      
if __name__ == '__main__':
  print("main")
  count = print_string_permutations(list("vik"))
  print(f"count = {count}")
  
  perm = [2, 0, 1, 3]
  l = ['a','b','c','d']
  apply_permute(perm, l)
  print(f"After = {l}")
  
  a, b = 1, 2
  a, b = b, a
  print(a, b) 
  
  a, b = 1, 2
  b, a = a, b
  print(a, b)
  print('\n')

  s = '[hammer999]'
  count_unique_chars(s) 