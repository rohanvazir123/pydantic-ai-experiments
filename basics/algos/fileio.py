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

import csv, sys, pprint
from typing import Generator


import csv
from typing import Generator

def parse_file_in_batches(
    filename: str, 
    batch_size: int, 
    separator: str = ",", 
    filter: str = "MEDIUM"
) -> Generator[list[list[str]], None, None]:  # Fixed type hint
    batch: list[list[str]] = []
    with open(filename, "r", newline="") as f:
        reader = csv.reader(f, delimiter=separator)
        for row in reader:
            if not row:
                continue
            if filter and row[0].upper() != filter:
                continue
            
            # Cleans row and skips first 4 columns
            cleaned_row = [x.strip() for i, x in enumerate(row) if i not in (0, 1, 2, 3)]
            batch.append(cleaned_row)
            
            if len(batch) == batch_size:
                yield batch
                batch = []
                
    if batch:
        yield batch  # Correct ending


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python fileio.py <filename>")
    sys.exit(1)
  filename = sys.argv[1]
  for i, batch in enumerate(parse_file_in_batches(filename, batch_size=100, separator=',', filter='MEDIUM')):
    print(f"Batch {i}:", "size: ", len(batch), end=" ")
    pprint.pprint(batch, indent=4, width=80)

