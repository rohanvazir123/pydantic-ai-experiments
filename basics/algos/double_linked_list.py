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

class DoubleListNode:
  def __init__(self, key: int | None = None):
    self.key, self.next, self.prev = key, None, None
    
class DoubleLinkedList:
  def __init__(self):
    self.head = DoubleListNode()
    self.tail = self.head
    
  def append(self, key):
    node = DoubleListNode(key)
    node.prev = self.tail
    self.tail.next = node
    if self.head == self.tail:
      self.head.next = node
    self.tail = node  
    
  def __str__(self):
    node = self.head
    l : list = []
    while node:= node.next:
      l.append(str(node.key))
    return " ".join(l)
      
  def delete(self):
    print('Delete')
    
if __name__ == '__main__':
  double_link_list = DoubleLinkedList()
  double_link_list.append(1)
  double_link_list.append(2)
  double_link_list.append(3)
  print(double_link_list)