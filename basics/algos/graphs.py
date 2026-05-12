import sys
from typing import List, Tuple
from functools import namedtuple

co_ords = namedtuple("Coordinates", ["x", "y"])

WHITE, BLACK = range(2)

def search_maze(maze:List[List[int]], start:co_ords, end:co_ords) -> List[int]:
  def search_maze_helper(curr:co_ords):
    if not ( (0 <= curr.x and curr.x <= len(maze)) and (0 <= curr.y and curr.y <= len(maze[curr.x])) and (maze[curr.x][curr.y] == WHITE) ):
      return False
    path.append(curr)
    maze[curr.x][curr.y] = BLACK
    
    # if we found the end, yayy!
    if curr == end:
      return True
    
    x, y = curr.x, curr.y
    if any(
        (
          search_maze_helper(co_ords(x-1,y)),
          search_maze_helper(co_ords(x+1, y)),
          search_maze_helper(co_ords(x, y+1)),
          search_maze_helper(co_ords(x, y-1)) 
        )
      ):
      return True
    
    path.pop(-1)
    
    return False
     
  
  path = []
  search_maze_helper(start, end)
  return path

if __name__ == "__main__":
  pass