class Solution:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        ex_times = [0 for _ in range(n)]

        # stack of currently ACTIVE function ids (pushed on start, popped on end)
        f, e, t = logs[0].split(':')
        f, t = int(f), int(t)
        stack = [f]
        prev_time = t

        for log in logs[1:]:
            fid2, etype2, ts2 = log.split(':')
            fid2, ts2 = int(fid2), int(ts2)

            if etype2 == 'start':
                # whoever was running (top of stack) ran until this new call started
                if stack:
                    ex_times[stack[-1]] += ts2 - prev_time
                stack.append(fid2)
                prev_time = ts2
            else:
                # the function ending was running through ts2, inclusive
                ex_times[stack.pop()] += ts2 - prev_time + 1
                prev_time = ts2 + 1

        return ex_times