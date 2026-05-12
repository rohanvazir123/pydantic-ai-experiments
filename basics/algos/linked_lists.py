from typing import Optional
import copy

class MyListNode:
  def __init__(self, key:int=None):
    self.key, self.next = key, None
    
class MyList:
  def __init__(self):
    self.dummy_head = MyListNode("head")
    self.tail = self.dummy_head
    self.count = 0

  def print(self):
    node = self.dummy_head.next
    while node is not None:
      print(node.key, sep=" ")
      node = node.next
    return  

  def append(self, key):
    if self.tail:
      new_node = MyListNode(key)
      self.tail.next = new_node
      self.tail = new_node
    self.count += 1
    return 
  
  def prepend(self, key): 
    # TODO   
    return
  
  def delete(self, key):
    # TODO
    return
  
  def reverse(self, start=0, end=None):
    current = self.dummy_head
    prev_, next_ = self.dummy_head, None
    start_node = prev_

    end = end if end else self.count

    # move the current node to the start node
    for i in range(0, start+1):
      if current:
        start_node = prev_
        prev_ = current
        current = current.next
        next_ = current.next
    
    for j in range(start, end):
      if not current:
        break
      current.next = prev_
      prev_ = current
      current = next_
      next_ = current.next if current else None
      
    temp = start_node.next
    start_node.next = prev_
    temp.next = current

    return
  
  def delete_alternates(self):
    current = self.dummy_head.next
    while current:
      if current.next:
        current.next  = current.next.next if current.next.next else None
        current = current.next
      else:
        break
    return
  
  def delete_duplicates(self):
    return
  
  def sort(self):
    return
  
  def find_cycles(self):
    return
  
  def lookup(self, key) -> Optional[MyListNode]:
    return None

  def print(self):
    # print(f"Printing list with head {hex(id(self.dummy_head))}")
    node = self.dummy_head.next
    while node:
      print(node.key, end="->")
      node = node.next
    print("")
    return

def add_numbers(list1:MyList, list2:MyList) -> MyList:
  import copy
  output : MyList = MyList()

  '''
  print("\n\nPrinting nos to be added")
  list1.print(); list2.print()
  print("Printing nos to be added in the reverse")
  list1.reverse(); list1.print(); list2.reverse(); list2.print()
  print("Printing nos to be added")
  list1.reverse(); list1.print(); list2.reverse(); list2.print()
  print("\n\n")
  '''
  print("\n\nPrinting nos to be added")
  list1.print(); list2.print()

  #Initialize
  node1 = list1.dummy_head.next
  node2 = list2.dummy_head.next
  carry = 0

  while node1 or node2:
    sum = (node1.key if node1 else 0) + (node2.key if node2 else 0) + carry
    if sum >= 10:
      sum = sum - 10
      carry = 1
    else:
      carry = 0

    if sum:
      output.append(sum)
  
    if node1: node1 = node1.next
    if node2: node2 = node2.next

  print('Printing output list')
  output.reverse(); output.print()
  return output.reverse()
  

if __name__ == '__main__':

  # clear screen
  import os
  os.system('cls')

  # create a custom linked list
  mylist = MyList()
  for i in range(1, 10):
    mylist.append(i)

  # delete alternate nodes from the list
  mylist.print()
  print("")
  mylist.delete_alternates()
  mylist.print()

  # reverse list
  # TODO

  # add two integers represented by a list in the reverse
  list1: MyList = MyList(); list2: MyList = MyList()
  for i in [ 8, 9, 7]: list1.append(i)
  for i in [ 3, 4]: list2.append(i)
  add_numbers(list1, list2)

