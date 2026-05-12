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