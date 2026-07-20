class Solution:
    def calculate(self, s: str) -> int:

        stack = []
        l = len(s)

        # By default operator is always + 
        # and so we append the current number as integer, that we assembled (see line 16)
        operator = '+'
        current_number = 0

        for i, c in enumerate(s):
            if c.isdigit():
                current_number = 10*current_number + int(c)
            
            if c in '+-*/' or i == l-1:
                if operator == '+':
                    stack.append(current_number)
                elif operator == '-':
                    stack.append(-current_number)
                elif operator == '*':
                    stack[-1]  *= current_number
                elif operator == '/':
                    stack[-1] = int(stack[-1] / current_number) if stack[-1] < 0 else  stack[-1] // current_number

                operator = c
                current_number = 0

        return sum(stack)
