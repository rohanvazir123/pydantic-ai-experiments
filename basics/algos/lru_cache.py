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