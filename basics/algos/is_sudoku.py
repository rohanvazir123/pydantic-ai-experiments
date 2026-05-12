import collections
from typing import List
from itertools import islice

def isValidSudoku(board: List[List[str]]) -> bool:
  if not board or len(board) < 9:
    return False
  
  def _is_valid(list_: List[str]):
    opt = [item for item in set(list_) if list_.count(item) > 1]
    opt.remove('.')
    if (len(opt) > 0):
      return False
    return True
  
  def _is_grid_valid(list_):
    flat_list = []
    for sublist in list_:
      for item in sublist:
        flat_list.append(item)

    opt = [item for item in set(flat_list) if flat_list.count(item) > 1]
    opt.remove('.')
    if (len(opt) > 0):
      return False
    return True

  row_len = len(board[0])
  col_len = row_len
  
  # Check if each row is valid
  all_rows = [row for row in board]
  row_valid = False
  for one_row in all_rows:
    one_row_valid = _is_valid(one_row)
    if one_row_valid == False:
      return False
    else:
      row_valid = True

  # Check if each column is valid
  all_cols = []
  for i in range(row_len):
    col = []
    for j in range(col_len):
      col.append(board[j][i])
    all_cols.append(col)
  
  col_valid = False
  for one_col in all_cols:
    one_col_valid = _is_valid(one_col)
    if one_col_valid == False:
      return False
    else:
      col_valid = True

  #Check if each grid is valid
  grid_valid = False
  row_list = []
  size = 3
  for i in range(row_len):
    one_row = []
    for j in range(col_len):
        one_row.append(board[i][j])
    row_list.append(one_row)
   
  # Get three sublists with three rows each
  start_size = 0
  end_size = 3
  
  #Scan each sublist and make 3X3 grids out of each sublist
  while end_size <= len(row_list):
    sublist = row_list[start_size:end_size]
    #Make 3X3 grids out of each sublist  
    start_col = 0
    end_col = 3
    while end_col <= col_len:
      one_grid = []
      for row in sublist:
        one_grid.append(row[start_col:end_col])
      #print('one_grid ', one_grid)      
      one_grid_valid = _is_grid_valid(one_grid)
      #print('one_grid_valid: ',one_grid_valid)
      if one_grid_valid == False:
        return False
      else:
        grid_valid = True
      start_col += size
      end_col += size
    
    start_size += size
    end_size += size

  if row_valid and col_valid and grid_valid:
    return True
  else:
    return False


if __name__ == "__main__":  
  board = [["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
  is_valid = isValidSudoku(board)
  print(f"is_valid: {is_valid}")
  board2 = [["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
  is_valid_2 = isValidSudoku(board2)
  print(f"is_valid: {is_valid_2}")
  board3 = [[]]
  is_valid_3 = isValidSudoku(board3)
  print(f"is_valid: {is_valid_3}")
  board4 = [[".",".","4",".",".",".","6","3","."],[".",".",".",".",".",".",".",".","."],["5",".",".",".",".",".",".","9","."],[".",".",".","5","6",".",".",".","."],["4",".","3",".",".",".",".",".","1"],[".",".",".","7",".",".",".",".","."],[".",".",".","5",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."]]
  is_valid_4 = isValidSudoku(board4)
  print(f"is_valid: {is_valid_4}")