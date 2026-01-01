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

import os
import re


def cleanup_notebook_scripts():
    for file in os.listdir("."):
        if not file.endswith(".py") or file.endswith("_cleaned.py") or file == os.path.basename(__file__):
            # print(f"Skipping non-Python/already cleaned file: {file}")
            continue
        print(f"Cleaning file: {file}")
        input_file = file
        output_file = file.replace(".py", "_cleaned.py")
        with open(input_file) as f_orig:
            # This regex finds and removes lines starting with '# In['
            script = re.sub(r"# In\[.*\]:\n", "", f_orig.read())
            # remove get_ipython calls
            script = re.sub(r"get_ipython\(\).run_line_magic\('.*`, '.*'\)\n", "", script)
            # remove any other get_ipython calls
            script = re.sub(r"get_ipython\(\)\..*\n", "", script)
            # remove trailing whitespace
            script = re.sub(r"[ \t]+$", "", script, flags=re.MULTILINE)

        with open(output_file, "w") as fh:
            fh.write(script)


if __name__ == "__main__":
    print("Cleaning up notebook scripts...")
    cleanup_notebook_scripts()