# Copyright 2024 The Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
from typing import List

def diagonalSum( mat: List[List[int]]) -> int:
  if not mat or len(mat) < 1 or len(mat) > 100:
    return 0
  
  row_len = len(mat[0])
  col_len = row_len
  list_indices = []
  n = len(mat)
    
  all_rows = [row for row in mat]
  for row in all_rows:
    if len(row) < n:
      return 0
    
  sum = 0
  i = 0
  j = 0
  
  while i < row_len:
    sum += mat[i][j]
    list_indices.append((i,j))
    i += 1
    j += 1

  i = 0
  j = col_len - 1
  while i < row_len:
    if((i,j) not in list_indices):
      sum += mat[i][j]
    i += 1
    j -= 1
  
  print(f"Sum: {sum}")
  return sum

if __name__ == "__main__":  
  mat1 = [[1,2,3],[4,5,6],[7,8,9]]
  sum_1 = diagonalSum(mat1)
  print(f"Sum 1: {sum_1}")
  mat2 = [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
  sum_2 = diagonalSum(mat2)
  print(f"Sum 2: {sum_2}")
  mat3 = [[5]]
  sum_3 = diagonalSum(mat3)
  print(f"Sum 3: {sum_3}")
  mat4 = [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1]]
  sum_4 = diagonalSum(mat4)
  print(f"Sum 4: {sum_4}")
  