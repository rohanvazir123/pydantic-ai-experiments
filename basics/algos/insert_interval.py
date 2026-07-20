class Solution:

    def insert(self, intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:
        import bisect

        def merge(intervals: list[list[int]]) -> list[list[int]]:
            # sort by start times
            intervals.sort(key = lambda x : x[0])

            merged_intervals = [intervals[0]]

            for iv in intervals[1:]:
                # Get the last interval
                last = merged_intervals[-1]

                # If overlapping merge and update, else append
                if iv[0] > last[1]:
                    # no overlap
                    merged_intervals.append(iv)
                else:
                    last[1] = max(iv[1], last[1])

            return merged_intervals

        rindex = bisect.bisect_right(intervals, newInterval, key=lambda x: [x[0], x[1]])
        # print(rindex, newInterval, intervals)

        # insert
        intervals.insert(rindex, newInterval)

        return merge(intervals)
       

'''
The above approach is O(nlogn + n) which is not good!

Better approach
Given that guarantee, consider restructuring as a single linear pass with three phases, instead of bisect+insert+generic-merge:
1. Copy over all intervals that end strictly before newInterval starts (no overlap possible, unchanged).
2. Among the remaining intervals, absorb every one that overlaps newInterval into a single growing merged interval (expanding newInterval's bounds as you go).
3. Copy over everything left (all start strictly after the merged interval ends).

That's one pass, no bisect, no insert (which is itself O(n) due to shifting), no sort
'''