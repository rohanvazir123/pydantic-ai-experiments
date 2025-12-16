import re
import os

from openai import files


def cleanup_notebook_scripts():
    for file in os.listdir("."):
        if not file.endswith(".py"):
            print(f"Skipping non-Python file: {file}")
            continue
        input_file = file
        output_file = file.replace(".py", "_cleaned.py")
        with open(input_file, "r") as f_orig:
            # This regex finds and removes lines starting with '# In['
            script = re.sub(r"# In\[.*\]:\n", "", f_orig.read())

        with open(output_file, "w") as fh:
            fh.write(script)


if __name__ == "__main__":
    print("Cleaning up notebook scripts...")
    cleanup_notebook_scripts()
