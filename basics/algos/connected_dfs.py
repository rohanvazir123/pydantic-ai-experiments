from collections import defaultdict
class Solution:
    
    def accountsMerge(self, accounts: list[list[str]]) -> list[list[str]]:

        # init
        graph = defaultdict(list)
        email_to_name = dict()

        def _build_graph():
            # build graph, where first_email => other emails, and other emails => first email
            # first email is the group leader
            # first_email of a group gets connected to first_email of another group, 
            # which it is a member of
            for a in accounts:
                name = a[0]
                first_email = a[1]
                for e in a[1:]:
                    graph[first_email].append(e)
                    graph[e].append(first_email)
                    email_to_name[e] = name


        def _connected_components() -> list:
            # DFS to find connected component groups, merge groups 
            # A group leader of group_A, B group leader of group_B
            # A is a member of group_B, group_A and group_B get merged
            stack = []
            visited = set()
            merged_components = []

            # DFS for each node in the graph
            for email, name in email_to_name.items():
                if email in visited:
                    continue
                
                visited.add(email)
                stack = [email]
                components = []

                while stack:
                    current = stack.pop()
                    components.append(current)

                    for n in graph[current]:
                        if n in visited:
                            continue
                        visited.add(n)
                        stack.append(n)

                # Update merged components for each email + name
                components.sort()
                merged_components.append([name]+components)

            return merged_components

        _build_graph()
        return _connected_components()



        


        