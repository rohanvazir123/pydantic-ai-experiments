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