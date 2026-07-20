mport re
class Solution:
    def simplifyPath(self, path: str) -> str:

        # single char path
        if len(path) == 1:
            if path[0] == '/':
                return path
            else:
                return ""

        # Init stack with first chat
        stack = [path[0]]

        special_chars = {'/', '.'}
        
        # Remove multiple // and convert to /
        # path = re.sub(r'/+', '/', path)

        for i, c in enumerate(path[1:]):

            # Always push on stack
            stack.append(c)

            # If c is not a special char, keep going
            if c not in special_chars:
                continue
            
            # We are here, so c MUST be a special char
            if len(stack) > 3:
                if stack[-4] == '/' and stack[-3] == '.' and stack[-2] == '.' and stack[-1] == '/':
                    # A double period '..' represents the previous/parent directory.
                    # '/home/user/Documents/../Pictures'
                    for _ in range(3):
                        stack.pop()
                    # keep popping until '/' reached or stack is empty
                    while len(stack) > 1:
                        stack.pop()
                        if stack and stack[-1] == '/':
                            break

            if len(stack) > 2:
                if stack[-3] == '/' and stack[-2] == '.' and stack[-1] == '/': 
                    # A single period './' represents the current dir
                    # '/home/user/Documents/./Pictures' => ''/home/user/Documents/Pictures'
                    # print("stack BEFORE POP: ", "".join(stack))
                    for _ in range(2):
                        stack.pop()
                    # print("stack AFTER POP: ", "".join(stack))



        out_path = "".join(stack)
        # //////.. => /
        out_path = re.sub(r'/+', '/', out_path)

        if len(out_path) == 1:
            return out_path

        return out_path[:-1] if out_path[-1] == '/' else out_path

        