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
      