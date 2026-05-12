from typing import NamedTuple


class Coordinates(NamedTuple):
    x: int
    y: int


WHITE, BLACK = range(2)


def search_maze(
    maze: list[list[int]], start: Coordinates, end: Coordinates
) -> list[Coordinates]:
    def search_maze_helper(curr: Coordinates):
        if not (
            0 <= curr.x < len(maze)
            and 0 <= curr.y < len(maze[curr.x])
            and maze[curr.x][curr.y] == WHITE
        ):
            return False
        path.append(curr)
        maze[curr.x][curr.y] = BLACK

        # if we found the end, yayy!
        if curr == end:
            return True

        x, y = curr.x, curr.y
        if any(
            (
                search_maze_helper(Coordinates(x - 1, y)),
                search_maze_helper(Coordinates(x + 1, y)),
                search_maze_helper(Coordinates(x, y + 1)),
                search_maze_helper(Coordinates(x, y - 1)),
            )
        ):
            return True

        path.pop(-1)

        return False

    path: list[Coordinates] = []
    search_maze_helper(start)
    return path
  
def print_path(path: list[Coordinates]):
    for c in path:
        print(f"({c.x}, {c.y})", end=" ")
    print() 


if __name__ == "__main__":
    maze = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
    ]
    start = Coordinates(0, 0)
    end = Coordinates(4, 4)
    path = search_maze(maze, start, end)
    print_path(path)
