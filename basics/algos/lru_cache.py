import collections

class LRUCache:
  def __init__(self, capacity:int):
    self.capacity = capacity
    self.cache :collections.OrderedDict[int, str] = collections.OrderedDict()
    
  def lookup(self, key:int):
    if key not in self.cache:
      return None
    value = self.cache.pop(key)
    self.cache[key] = value
    return value
  
  def insert(self, key:int, value:str):
    if key in self.cache:
      value = self.cache.pop(key)
    elif len(self.cache) == self.capacity:
      self.cache.popitem(last=False)
    self.cache[key] = value
    return
        
  def erase(): 
    return

if __name__ == '__main__':
  lru_cache = LRUCache(3)
  lru_cache.insert(1, 'one')
  lru_cache.insert(2, 'two')
  print(lru_cache.lookup(1))
  lru_cache.insert(3, 'three')
  print(lru_cache.cache)
  
  lru_cache.insert(4, 'four')
  print(lru_cache.lookup(1))
  lru_cache.insert(5, 'five')
  lru_cache.insert(6, 'six')
  print(lru_cache.cache)
  
  #print(lru_cache.lookup(1))
  #print(lru_cache.cache)