from collections import deque

def bfs_final(graph, start, target):
    queue = deque([start])
    parents = {start: None}

    while queue:
        node = queue.popleft()
        
        if node == target:
            # 🚀 exact idea: build forward while walking backward
            path = deque()
            while node is not None:
                path.appendleft(node)  # O(1) insertion at the front
                node = parents[node]
            return list(path)

        for neighbor in graph[node]:
            if neighbor not in parents:
                parents[neighbor] = node
                queue.append(neighbor)

    return None


