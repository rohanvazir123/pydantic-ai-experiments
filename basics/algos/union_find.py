class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n  # optional, only needed if you want to track the size of each set


    # classic recursive find with path compression
    def find(self, x: int) -> int:
        # Path compression: make every node on the path point directly to the root
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    # Instead of the classic recursive, do it iteratively to avoid stack overflow on deep trees.
    # The iterative find method works by traversing up the tree to find the root of the set 
    # and then performing path compression in a second pass.
    def find(self, x: int) -> int:
        # pass 1: walk up to the root, no mutation yet
        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        # pass 2: path compression - re-point every node on that
        # same path straight at the root, so future find() calls
        # thr   ough any of them are ~O(1) instead of re-walking the chain
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]

        return root

    # Union by rank: attach the shorter tree under the taller one, 
    # When we union two trees, we attach the shorter tree under the taller one,
    # which helps keep the overall tree flat and ensures that future find operations are efficient.
    # The union operation is a key part of the Union-Find data structure. 
    # It allows us to merge two disjoint sets into a single set. 
    # The union operation first finds the roots of the two sets using the find method. 
    # If the roots are different, it means the two elements belong to different sets, and we can merge them. 
    # We use the rank array to determine which tree is shorter and attach it under the taller tree. 
    # If both trees have the same rank, we can choose either one as the new root and increment its rank by 1. 
    # This ensures that the tree remains balanced and keeps the time complexity of future operations low.
    def union_by_rank(self, x: int, y: int) -> bool:

        # Find the root of x and y
        root_x, root_y = self.find(x), self.find(y)

        # Root are the same, nothing to do
        if root_x == root_y:
            return False

        # if 'rep_x' is the represents a taller tree than 'rep_y',  
        #   then we make 'rep_x' the parent of 'rep_y'
        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        elif self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            # if both trees have the same rank, we can choose either one as the new root
            self.parent[root_y] = root_x
            self.rank[root_x] += 1  # increment the rank of the new root

        return True

    # Union two sets together, return True if they were separate and merged, False if already in the same set
    # Union by size: attach the smaller tree under the larger one,
    # so the tree never grows deeper than log(n) on its own
    # This is the "union by size" optimization. 
    # The size array keeps track of the number of elements in each set.
    def union_by_size(self, x: int, y: int) -> bool:

        # Find the root of x and y
        root_x, root_y = self.find(x), self.find(y)

        # Root are the same, nothing to do
        if root_x == root_y:
            return False  # already in the same set, no-op

        # If 'rep_x' is the represents a larger set than 'rep_y', 
        #   then we make 'rep_x' the parent of 'rep_y'
        if self.size[root_x] > self.size[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            # If 'rep_y' is the represents a larger set than 'rep_x' or they are equal,
            #   then we make 'rep_y' the parent of 'rep_x'
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]

        return True

    # Check if two elements are in the same set
    # This method checks if two elements x and y are in the same set by comparing their roots. 
    # It uses the find method to get the root of each element and returns True if they are the same, 
    # indicating that x and y are connected. 
    # If the roots are different, it returns False, indicating that x and y are in different sets. 
    # This method is useful for determining if two elements are related or connected 
    # in the context of the Union-Find data structure.
    def is_connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


if __name__ == "__main__":
    # 0-1-2 merge into one group; 3-4 merge into another; 5 stays alone
    uf = UnionFind(6)

    # Union by rank example
    uf.union_by_rank(0, 1)
    uf.union_by_rank(1, 2)
    uf.union_by_rank(3, 4)

    print("is_connected(0, 2):", uf.is_connected(0, 2))  # True  - same group via 1
    print("is_connected(0, 3):", uf.is_connected(0, 3))  # False - different groups
    print("is_connected(5, 5):", uf.is_connected(5, 5))  # True  - trivially itself

    groups: dict[int, list[int]] = {}
    for node in range(6):
        groups.setdefault(uf.find(node), []).append(node)
    print("groups:", list(groups.values()))


'''
Accounts Merge (721) is a "these things are the same, group them together" problem. 
Shared email between two accounts means "same person" — a pure equivalence relation.
That's exactly what plain Union-Find models: union(a, b) means "a and b are now known to be in the same group,
   transitively. No opposition, no constraint-checking — just merging.

Possible Bipartition (886) is a "these things must be different, can we validly split into two groups" problem.
A dislike-edge between two people means "these two must end up in opposite groups." 
That's an inequality constraint, not an equivalence — which is why it doesn't map onto union_find.py as written. 
The natural fit there is BFS/DFS 2-coloring: color a node, force all its neighbors to the opposite color, 
and fail if you ever find an edge connecting two same-colored nodes.

Can Union-Find solve bipartition too? 
Yes, but it needs a trick the plain DSU doesn't have: either (a) 
double the node count — one node for "person X in group A," one for "person X in group B" — 
and union each dislike-pair's A-node with the other's B-node and vice versa, then check no person's two halves ended up unioned together; or (b) track a parallel "enemy" pointer alongside parent/rank so union can enforce "these two are opposites" instead of "these two are the same." Neither is in union_find.py right now — it's the plain connectivity version, which is exactly what Accounts Merge needs and exactly what Bipartition doesn't.

So: your read is correct, the concept really is different — 
Accounts Merge is where the file you have now applies directly.

'''