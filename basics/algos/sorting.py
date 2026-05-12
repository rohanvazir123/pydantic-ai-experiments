def dutch_flag(l, lt, gt):
  print('Starting...')
  start = 0
  for i in range (len(l)):
    if l[i] <= lt:
      l[start], l[i] = l[i], l[start]
      start += 1

  end = len(l) - 1
  for j in range(end, start-1, -1):
    if l[j] >= gt:
      l[end], l[j] = l[j], l[end]
      end -= 1
  return l

def merge_intervals(l):
  l.sort(key = lambda x:x[0])
  merge_list = []

  #Add first element in the output list
  merge_list.append(l[0])
  
  for i in range(1, len(l)):
    input_ele = l[i]
    
    #Remove last element from the list for comparison
    last_ele = merge_list.pop()

    start_merge, stop_merge = last_ele[0], last_ele[1]
    start_curr, stop_curr = input_ele[0], input_ele[1]

    if start_curr > stop_curr:
      raise ValueError(f"Invalid input - start interval {start_curr} cannot be greater than end interval {stop_curr}")
 
    if start_curr <= stop_merge:
      #Intervals can be merged in this case
      new_int = (start_merge, max(stop_curr, stop_merge))
      merge_list.append(new_int)
    else:
      #Nothing to merge, add both in the list
      merge_list.append(last_ele)
      merge_list.append(input_ele)
  return merge_list

if __name__ == '__main__':
  # Dutch flag
  l = [3, 19, 17, 4, 6, 8, 21, 13, 9, 8, 11, 1]
  out_list = dutch_flag(l, 4, 10)
  print(f"Final output = {out_list}")

  input_list = [[7, 100], [95, 200]]
  merged_interval_list = merge_intervals(input_list)
  print(f"Merged intervals = {merged_interval_list}")