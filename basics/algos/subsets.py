def subsets(nums: list[int]) -> list[list[int]]:
    result = []
    current_subset = []
    
    def backtrack(index: int) -> None:
        # Base Case: We have made a decision for every single element
        if index == len(nums):
            result.append(current_subset.copy()) # O(N) copy
            return
            
        # Branch 1: CHOOSE to INCLUDE nums[index]
        current_subset.append(nums[index])
        backtrack(index + 1)                 # EXPLORE
        current_subset.pop()                 # UN-CHOOSE (Backtrack)
        
        # Branch 2: CHOOSE to EXCLUDE nums[index]
        backtrack(index + 1)                 # EXPLORE
        
    backtrack(0)
    return result
