class BST:
  class Node:
    def __init__(self, key):
      self.key = key
      self.left = None
      self.right = None
    
  def __init__(self):
    self.root = None
    return
  
  def insert(self, key):
    if not self.root:
      self.root = self.Node(key)
      return

    parent, root = None, self.root
    while root:
      parent = root
      if key == root.key: 
        return # no duplicates allowed
      
      root = root.left if key < root.key else root.right
        
    if key < parent.key:
      parent.left = self.Node(key)
    else:
      parent.right = self.Node(key)
    return
  
  def delete(self, key):
    return
  
  def find(self, key):
    root = self.root
    while root:
      if root.key == key:
        return root
      if key < root.left:
        root = root.left
      else:
        root = root.right
    return root
  
  def dump_inorder(self, root):
    if root:
      self.dump_inorder(root.left)
      print(root.key)
      self.dump_inorder(root.right)
      
  def dump(self):
    return self.dump_inorder(self.root)
    

def test_bst_insert():
  bst = BST()
  bst.insert(5)
  bst.insert(4)
  bst.insert(6)
  bst.insert(3)
  bst.insert(4.5)
  bst.insert(6.5)
  bst.insert(5.5)
  bst.insert(1)
  bst.dump()
  return bst
  
def test_bst_find():
  bst = test_bst_insert()
  bst.find(5.5)
  return

def test_bst_delete():
  return
  
if __name__ == '__main__':
  test_bst_insert()