from collections import deque
from typing import Tuple

class TreeNode:
  def __init__(self, key):
    self.key = key
    self.left, self.right = None, None

class BST:
  def __init__(self):
    self.root = None
    
  def insert(self, key):
    if self.root is None:
      self.root = TreeNode(key)
      return
  
    parent_iter_node = iter_node = self.root
    # Duplicate keys cannot be inserted
    while iter_node:
      # Duplicate keys cannot be inserted
      if key == iter_node.key:
        return
      if key < iter_node.key:
        parent_iter_node = iter_node
        iter_node = iter_node.left
      else:
        parent_iter_node = iter_node
        iter_node = iter_node.right

    if key < parent_iter_node.key:
      parent_iter_node.left = TreeNode(key)
    else:
      parent_iter_node.right = TreeNode(key)
    return

  def inorder(self):           
    """Inorder BST traversal
    Args: None
    Returns : None
    """
    # Helper function to do inorder traversal
    def _inorder_recursive(root):
      if root is None:
        return
      _inorder_recursive(root.left)
      print(root.key, end=" ")
      _inorder_recursive(root.right)
      return
    _inorder_recursive(self.root)

    return

  def bfs_traversal(self):
    def _bfs(root):
      if root is None:
        return
      d = deque()  
      # First append the first node that is root  
      d.append(root)
      while d:
        # Pop the root and print the key
        node = d.popleft()
        print(node.key, sep = "", end=" ")
        # If the node has left child, add it to the queue
        if node.left is not None:
          d.append(node.left)
        # If the node has right child, add it to the queue
        if node.right is not None:
          d.append(node.right)
        return

    # BEGIN
    if not self.root:
      print("tree is empty")
      return
    _bfs(self.root)
    return

  # find the node with key
  def _find(self, root, key):
    if root is None:
      return None, None, False
    
    parent_iter_node = None
    iter_node = root
    is_left = False

    # Traverse the tree till we find the node to be deleted
    while iter_node:
      # Found the node that we are looking for
      if key == iter_node.key:
        if parent_iter_node is None:
          return None, iter_node, False 
        # Check the direction - if it is on left or right of the parent
        is_left = True if key < parent_iter_node.key else False
        break   
      elif key > iter_node.key:
        parent_iter_node = iter_node
        iter_node = iter_node.right
      elif key < iter_node.key:
        parent_iter_node = iter_node
        iter_node = iter_node.left

    return parent_iter_node, iter_node, is_left

  def delete_node(self, key):
    if not self.root:
      return
    if key == self.root.key and not(self.root.left or self.root.right):
      self.root = None
      return
    # helper function to delete the node
    def _delete_node(root, key):
      # Find the node to be deleted, its parent node and its direction from the parent node
      parent_iter_node, iter_node, is_left = self._find(root, key)
      if not iter_node:
        print(f"key {key} not found")
        return
      
      if (not iter_node.right) and (not iter_node.left):
        # Neither right child nor left child (leaf node)
        if parent_iter_node:
          if is_left:
            parent_iter_node.left = None 
          else:
            parent_iter_node.right = None
          iter_node = None  
      elif iter_node.right and not iter_node.left:
        # only right child, overwrite with right child
        iter_node.key = iter_node.right.key
        iter_node.right =  iter_node.right.right      
      elif iter_node.left and not iter_node.right:
        # only left child, overwrite with left child
        iter_node.key = iter_node.left.key
        iter_node.left = iter_node.left.left
      elif iter_node.left and iter_node.right:
        # Both left and right children
        # Take the minimum value from the right subtree, and copy the key
        # Delete the minimum value node from the right subtree
        def _get_min(iter_node):
          node = iter_node.right
          while node and node.left:
            node = node.left
          return node
          
        min_node = _get_min(iter_node)
        if min_node:
            min_key = min_node.key
            # first delete the node with min key from right subtree
            _delete_node(iter_node, min_key)
            # copy the min to iter_node
            iter_node.key = min_key         
            
    # BEGIN 
    # find the node to be deleted
    _delete_node(self.root, key)

    return

  def diameter_height(self) -> Tuple[int, int]:
    # Helper function to find height and diameter
    def _diameter_height(root) -> Tuple[int, int]:
      if root is None:
        return 0, 0
      
      # Find height of the tree first
      left_diameter, left_height = _diameter_height(root.left)
      right_diameter, right_height = _diameter_height(root.right)
      h = max(left_height, right_height) + 1
            
      # Longest path going through left subtree, root, right subtree
      lp = left_height + right_height + 1
      
      d = max(left_diameter, right_diameter, lp)

      return d, h
        
    # BEGIN
    return _diameter_height(self.root)

if __name__ == '__main__':
    # Tree 1
    bst = BST()
    bst.insert(3)
    bst.insert(4)
    bst.insert(5)
    bst.insert(2)
    bst.insert(1)
    bst.insert(6)
    bst.insert(7)
    print("Printing tree with DFS (inorder)")
    bst.inorder()
    print('\n')

    d, h = bst.diameter_height()
    print(f"Diameter: {d}")
    exit()
    
    print('Delete key 3')
    bst.delete_node(3)
    print("Printing tree with BFS")
    bst.bfs_traversal()
    print('\n')

    print('Delete key 4')
    bst.delete_node(4)
    print("Printing tree with BFS")
    bst.bfs_traversal()
    print('\n')

    print('Delete key 2')
    bst.delete_node(2)
    print("Printing tree with BFS")
    bst.bfs_traversal()
    print('\n')

    print('Delete key 1')
    bst.delete_node(1)
    print("Printing tree with BFS")
    bst.bfs_traversal()
    print('\n')

    print('Delete key 5')
    bst.delete_node(5)
    print("Printing tree with BFS")
    bst.bfs_traversal()
    print('\n')
    
