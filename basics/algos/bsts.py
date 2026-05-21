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