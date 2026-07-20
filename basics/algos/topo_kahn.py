from collections import deque

def topological_sort_kahn(num_nodes, edges):


    adj = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes

    # Make in-degree map for all dst nodes
    # for adj map for all src nodes
    for src, dst in edges:
        adj[src].append(dst)
        in_degree[dst] += 1

    queue = deque()

    # append all nodes with in_degree zero
    for node in range(num_nodes):
        if in_degree[node] == 0:
            queue.append(node)

    result = []
    while queue:
        node = queue.popleft()
        result.append(node)

        # Add all neighbors of this node
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != num_nodes:
        return None
    return result
