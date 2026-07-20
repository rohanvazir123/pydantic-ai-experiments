# 1. Setup your tools
stack = [start_node]
visited = {start_node}  # Mark it visited immediately!

# 2. Start the processing loop
while stack:
    current = stack.pop()
    # DO WORK HERE: Process the 'current' node (e.g., add to your component list)
    
    # 3. Explore the neighbors
    for neighbor in graph[current]:
        if neighbor not in visited:
            visited.add(neighbor)   # Mark visited RIGHT NOW
            stack.append(neighbor)  # Then push it
