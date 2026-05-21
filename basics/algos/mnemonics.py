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

#[2, 3, 1, 1, 4]
#[3, 2, 1, 0, 4]
from typing import List


def print_perms(s):

    num_to_str_map = {
        "2" : ["A", "B", "C"],
        "3" : ["D", "E", "F"],
        "4" : ["G", "H", "I"],
        "5" : ["J", "K", "L"],
        "6" : ["M", "N", "O"],
        "7" : ["P", "Q", "R", "S"],
        "8" : ["T", "U", "V"],
        "9" : ["W", "X", "Y", "Z"],
    }   
    
    def _print_perms(s, start, end):   
        if start > end:
            # Empty array, add nothing to the output
            return []
        val = num_to_str_map[s[start]]

        # prev_output = _print_perms(s, start+1, end)
        # return [ c+s_ for s_ in prev_output for c in val ] if prev_output else val
        if prev_output := _print_perms(s, start+1, end):
            return [ c+s_ for s_ in prev_output for c in val ]
        return val
       
                 
    return _print_perms(s, 0, len(s)-1)

if __name__ == '__main__':
    import pprint
    s = "532"
    output = print_perms(s)
    pprint.pprint(output)
    
    
    