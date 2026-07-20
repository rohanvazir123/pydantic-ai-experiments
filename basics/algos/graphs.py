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

from typing import NamedTuple


class Coordinates(NamedTuple):
    x: int
    y: int


WHITE, BLACK = range(2)


def search_maze(
    maze: list[list[int]], start: Coordinates, end: Coordinates
) -> list[Coordinates]:
    def search_maze_helper(curr: Coordinates) -> bool:
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
            search_maze_helper(Coordinates(x + dx, y + dy))
            for dx, dy in ((-1, 0), (1, 0), (0, 1), (0, -1))
        ):
            return True

        path.pop(-1)

        return False

    path: list[Coordinates] = []
    search_maze_helper(start)
    return path


def print_path(path: list[Coordinates]) -> None:
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
