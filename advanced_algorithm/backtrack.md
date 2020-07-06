# 回溯法

## 背景

回溯法（backtrack）常用于遍历列表所有子集，是 DFS 深度搜索一种，一般用于全排列，穷尽所有可能，遍历的过程实际上是一个决策树的遍历过程。
时间复杂度一般 O(N!)，它不像动态规划存在重叠子问题可以优化，回溯算法就是纯暴力穷举，复杂度一般都很高。

## 模板

```python

```

核心就是从选择列表里做一个选择，然后一直递归往下搜索答案，如果遇到路径不通，就返回来撤销这次选择。

## 示例

### [subsets](https://leetcode-cn.com/problems/subsets/)

> 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

遍历过程

![image.png](https://img.fuiboom.com/img/backtrack.png)

```python
class Solution:
    def subsets(self, nums):
        def back_track(first_idx=0,cur_subset=[]):
            # 满足结束条件,增加路径
            if len(cur_subset) == cur_length:
                result.append(cur_subset[:])
            # 在选择列表中做选择
            for i in range(first_idx,len(nums)):
                #做选择
                cur_subset+=[nums[i]]
                # 继续前进直至走不通或结束
                back_track(i+1,cur_subset)
                # 撤销选择
                cur_subset.pop()
        result  = []
        for cur_length in range(len(nums)+1):
            back_track()
        return result
```

### [subsets-ii](https://leetcode-cn.com/problems/subsets-ii/)

> 给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。说明：解集不能包含重复的子集。

```python
def subsetsWithDup(self, nums):
    def _back_track(first_idx=0,cur_set=[]):
        if len(cur_set)==cur_length:
            res.append(cur_set[:])
        for next_idx in range(first_idx,len(nums)):
            # 在同一位置上遇到重复元素，则跳过
            if next_idx>first_idx and nums[next_idx-1]==nums[next_idx]:
                continue
            cur_set.append(nums[next_idx])
            _back_track(next_idx+1,cur_set)
            cur_set.pop()
    #将数组按大小排序
    nums.sort()
    res = []
    for cur_length in range(len(nums)+1):
        _back_track()
    return res
```

### [permutations](https://leetcode-cn.com/problems/permutations/)

> 给定一个   没有重复   数字的序列，返回其所有可能的全排列。

思路：需要记录已经选择过的元素，满足条件的结果才进行返回

```python
def permute(self, nums):
    def _back_track(visited,cur_set):
        # 返回条件:当前结果和输入集合长度一致
        if len(cur_set)==len(nums):
           res.append(cur_set[:])
        for next_num in nums:
            # 跳过已经添加过的元素
            if next_num in visited:
                continue
            #选择
            cur_set.append(next_num)
            # visited标记加过的元素
            _back_track(visited+[next_num],cur_set)
            # 回溯
            cur_set.pop()
    res = []
    _back_track([],[])
    return res
```

### [permutations-ii](https://leetcode-cn.com/problems/permutations-ii/)

> 给定一个可包含重复数字的序列，返回所有不重复的全排列。

```python
def permuteUnique(self, nums):
    def _back_track(visited=[],cur_set=[]):
        # 返回条件:当前结果和输入集合长度一致
        if len(cur_set)==len(nums):
            res.append(cur_set[:])
        for next_idx in range(len(nums)):
            # 跳过已经添加过的元素
            if next_idx in visited:
                continue
            # 当前搜索和上一次搜索结果相同,且上次的结果会被再次访问(不在使用中)
            if next_idx>0 and nums[next_idx]==nums[next_idx-1] and next_idx-1 not in visited:
                continue
            #选择
            cur_set.append(nums[next_idx])
            # visited标记加过的元素
            _back_track(visited+[next_idx],cur_set)
            # 回溯
            cur_set.pop()
    nums.sort()
    res = []
    _back_track([],[])
    return res
```



### [combination-sum](https://leetcode-cn.com/problems/combination-sum/)

> 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
> 关键:保证下一层索引不小于比上一层
```python
def combinationSum(self, candidates, target: int):
    def _back_track(first_idx=0,cur_res=[],cur_target=0):
        if cur_target==0:
            res.append(cur_res[:])
            return
        for idx in range(first_idx,len(candidates)):
            candi = candidates[idx]
            #剪枝
            if cur_target-candi<0:
                break
            # 保证下一层索引不小于比上一层
            _back_track(idx,cur_res+[candi],cur_target-candi)
    
    candidates.sort()#为了剪枝,提速
    res = []
    _back_track(0,[],target)
    return res
```

### [letter-combinations-of-a-phone-number](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

> 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。

```python
def letterCombinations(self, digits: str):
    def _back_track(idx,prefix):
        if idx==len(digits):
            if len(prefix)>0:
                res.append(prefix)
            return
        digit = list(digits)[idx]
        S =digit2str_dict[int(digit)]
        #在当前步穷尽所有可能
        for s in S:
            _back_track(idx+1,prefix+s)
    #数字到字母的映射
    k = [2,3,4,5,6,7,8,9]
    v = ['abc','def','ghi','jkl','mno','pqrs','tuv','wxyz']
    digit2str_dict = dict(zip(k,v))
    # 
    res = []
    _back_track(0,'')
    return res
```

### [palindrome-partitioning](https://leetcode-cn.com/problems/palindrome-partitioning/)

> 分割回文串

```python
class Solution:
    def partition_dp(self, s: str) :
        # 动态规划找到所有回文串
        dp = [[] for _ in range(len(s)+1)]#dp[i]表示s[i:]组成的所有回文串
        dp[-1] = [[]]
        for left_i in range(len(s)-1,-1,-1):
            for right_i in range(left_i,len(s)):
                if s[left_i:right_i+1] == s[left_i:right_i+1][::-1]:
                    dp[left_i] += [[s[left_i:right_i+1]]+item for item in dp[right_i+1]]
        return dp[0]


    def partition_dp_dfs(self, s: str) :
        # 动规+dfs
        # step1.动态规划找到所有回文串
        dp = [[False]*len(s) for _ in range(len(s))]
        for l in range(len(s)):
            for i in range(len(s)-l):
                j = i+l
                #动态转移方程
                if s[i]==s[j] and (j-i<=2 or dp[i+1][j-1]):
                    dp[i][j] = True
        # print(dp)
        # step2. dfs找到所有组合
        res = []
        def _dfs(left_idx,cur_res):
            #left_idx:起始索引
            if left_idx==len(s):
                res.append(cur_res)
            for right_idx in range(left_idx,len(s)):
                if dp[left_idx][right_idx]:
                    #dfs 继续向下搜索
                    _dfs(right_idx+1,cur_res+[s[left_idx:right_idx+1]])
        _dfs(0,[])
        return res
```

### [restore-ip-addresses](https://leetcode-cn.com/problems/restore-ip-addresses/)

> 复原IP地址
>这个问题最朴素的解法是暴力法：遍历点可能的所有位置并判断合法性。需要11×10×9=990 次检查。
可以通过约束规划+回溯来减少检查次数，优化时间复杂度。
约束规划：对每个点的放置设置限制。每个点的放置位置只有 3 种可能：上个点的1/2/3个数字之后。只需要检测3×3×3=27种情况。
回溯：当已经放置的点使得无法摆放其他点来生成有效IP地址时，回到之前，改变上一个摆放点的位置。并试着继续。
要实现这个方法，我们需要一个地址栈来保存临时结果。
需要一个回溯函数，以（1）上一个点的位置（2）待放置点的数量（3）当前地址栈 为参数。再把回溯翻译成代码逻辑：如果还有需要放置的点，我们遍历三个有效位置，并判断两个位置之间的部分是否是有效整数，是则（1）将结果压入地址栈，递归调用函数，继续放下一个点；（2）递归返回后，移除栈顶元素，进行回溯。
当点放置完，检查剩余部分是否是有效整数，是则将结果添加到输出列表。

```python
class Solution:
    def restoreIpAddresses(self, s: str):
        def backtrack(prev_pos = -1, dots = 3,cur_res = []):
            if dots==0:
                cur_seg = s[prev_pos+1:]
                if int(cur_seg) not in range(256) or (len(cur_seg)>1 and cur_seg[0]=='0'):
                    return
                res.append('.'.join(cur_res+[cur_seg]))
            else:
                # 每个点的放置位置只有 3 种可能
                for cur_pos in range(prev_pos+1,min(prev_pos+4,len(s))):
                    cur_seg = s[prev_pos+1:cur_pos+1]
                    # 不在0-255或有多余的前缀0
                    if int(cur_seg) not in range(256) or (len(cur_seg)>1 and cur_seg[0]=='0'):
                        continue
                    cur_res+=[cur_seg]
                    backtrack(cur_pos,dots-1,cur_res)
                    cur_res.pop()
        n = len(s)
        res = []
        backtrack()
        return res
                
    def restoreIpAddresses_(self, s: str):
        def _back_track(cur_s=s,cur_res=[]):
            if len(cur_res)==4:
                if len(cur_s)==0:
                    # 增加结果
                    res.append('.'.join(cur_res))
                return
            #剪枝:剩下的字符无法构成ip地址
            if len(cur_s) not in range(1*(4-len(cur_res)),3*(4-len(cur_res))+1):
                return
            for i in range(1,min(4,1+len(cur_s))):
                # 剪枝:当前字符无法构成ip地址的一部分
                if not cur_s[:i].isdigit() or int(cur_s[:i]) not in range(256):
                    break
                if i>1 and cur_s[0]=='0':
                    break
                #增加下一个
                _back_track(cur_s[i:],cur_res+[cur_s[:i]])

        res = []
        _back_track()
        return res
```


## 练习

- [ ] [subsets](https://leetcode-cn.com/problems/subsets/)
- [ ] [subsets-ii](https://leetcode-cn.com/problems/subsets-ii/)
- [ ] [permutations](https://leetcode-cn.com/problems/permutations/)
- [ ] [permutations-ii](https://leetcode-cn.com/problems/permutations-ii/)
- [ ] [combination-sum](https://leetcode-cn.com/problems/combination-sum/)
- [ ] [letter-combinations-of-a-phone-number](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)
- [ ] [palindrome-partitioning](https://leetcode-cn.com/problems/palindrome-partitioning/)
- [ ] [restore-ip-addresses](https://leetcode-cn.com/problems/restore-ip-addresses/)
