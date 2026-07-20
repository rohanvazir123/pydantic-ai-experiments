from collections import deque

class Node:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        # The sibling pointer
        self.next = next

def connect_siblings_bfs(root: Node) -> Node:
    """
    Connects each node to its next right node using BFS.
    """
    if not root:
        return None
    
    # Initialize queue with the root node
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        # Traverse all nodes at the current level
        for i in range(level_size):
            curr = queue.popleft()
            
            # If it's not the last node in the current level,
            # link it to the next node in the queue
            if i < level_size - 1:
                curr.next = queue[0]
            
            # Push children to the queue for the next level
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
                
    return root
