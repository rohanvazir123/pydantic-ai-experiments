'''
You are given binary matrix grid of size m x n, 
where '1' denotes land and '0' signifies water. 
Determine the count of islands present in this grid. 
An island is defined as a region of contiguous 
land cells connected either vertically or horizontally, 
and it is completely encircled by water. 
Assume that the grid is bordered by water on all sides.

grid = [
[1,1,0,1],
[1,1,0,1],
[1,1,0,0],
]

Output 
2

'''


def numIslands(grid):
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def dfs(r, c):
        grid[r][c] = 0
        if r + 1 < rows and grid[r + 1][c] == 1:
            dfs(r + 1, c)
        if r > 0 and grid[r - 1][c] == 1:
            dfs(r - 1, c)
        if c + 1 < cols and grid[r][c + 1] == 1:
            dfs(r, c + 1)
        if c > 0 and grid[r][c - 1] == 1:
            dfs(r, c - 1)
        return
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                count += 1
                dfs(i, j)
    
    return count