'''
starts = [1, 3, 6, 10]
ends = [4, 5, 10, 12]
profits = [20, 20, 100, 70]

schedule jobs to maximmize profit

20 + 100 + 70 = 190
'''


from bisect import bisect_right


def js(starts: list[int], ends: list[int], profits: list[int]) -> int:
    jobs = sorted(zip(starts, ends, profits), key=lambda x: x[1])
    dp = [0] * len(jobs) + 1
    jobs_to_compare = [job[1] for job in jobs]
    for i in range(1, len(jobs) + 1):
        start, _end, profit = jobs[i - 1]
        # find number of jobs to finish before start of current job
        num_jobs = bisect_right(jobs_to_compare, start)

        dp[i] = max(dp[i - 1], dp[num_jobs] + profit)

    return dp[-1]


if __name__ == "__main__":
    assert js([1, 3, 6, 10], [4, 5, 10, 12], [20, 20, 100, 70]) == 190  # docstring example
    assert js([1, 2, 3, 3], [3, 4, 5, 6], [50, 10, 40, 70]) == 120  # LeetCode 1235 example
    assert js([1, 3], [2, 4], [5, 6]) == 11  # non-overlapping, take both
    assert js([1, 2], [4, 3], [5, 10]) == 10  # overlapping, take the higher-profit one
    assert js([1], [2], [5]) == 5  # single job
    assert js([], [], []) == 0  # no jobs
    print("all tests passed")

