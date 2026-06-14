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

import sys, pprint

def parse_file(filename: str) -> dict:
  kv_map = {}
  with open(filename, "r") as f:
    while True:
      line = f.readline()
      if not line: 
        break
      k, v = line.strip().rsplit(":", 1)
      kv_map[k] = v
  return kv_map

if __name__ == "__main__":
  if len(sys.argv) > 1:
    kv_map = parse_file(sys.argv[1])
    # dict.items returns tuple
    # pprint.pprint(list(kv_map.items()))
    print("this should print a list of tuples with each tuple being a key, value pair, they are sorted by keys")
    pprint.pprint(sorted(kv_map.items()))
    print("this should print only sorted keys")
    # new_kv_map = { k:kv_map[k] for k in sorted(kv_map) if 'hello' in k }
    new_kv_map = { k:kv_map[k] for k in sorted(kv_map) }
    print("this should print sorted dict")
    pprint.pprint(new_kv_map)
      