# 动态规划

问题可以分解成规模较小的子问题。因此，我们可以存储并复用子问题的解，而不是递归的（也重复的）解决这些子问题，这就是动态规划法。

## 背景

先从一道题目开始~

如题  [triangle](https://leetcode-cn.com/problems/triangle/)

> 给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。

例如，给定三角形：

```text
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
```

自顶向下的最小路径和为  11（即，2 + 3 + 5 + 1 = 11）。

使用 DFS（遍历 或者 分治法）

遍历

![image.png](https://img.fuiboom.com/img/dp_triangle.png)

分治法

![image.png](https://img.fuiboom.com/img/dp_dc.png)

优化 DFS，缓存已经被计算的值（称为：记忆化搜索 本质上：动态规划）

![image.png](https://img.fuiboom.com/img/dp_memory_search.png)

动态规划就是把大问题变成小问题，并解决了小问题重复计算的方法称为动态规划

动态规划和 DFS 区别

- 二叉树 子问题是没有交集，所以大部分二叉树都用递归或者分治法，即 DFS，就可以解决
- 像 triangle 这种是有重复走的情况，**子问题是有交集**，所以可以用动态规划来解决

动态规划，自底向上

```python
class Solution:
    def minimumTotal(self, triangle) -> int:
        # 自底向上的动态规划
        dp = triangle[:]
        for i in range(len(triangle)-2,-1,-1):
            for j in range(len(triangle[i])):
                dp[i][j]+=min(dp[i+1][j],dp[i+1][j+1])
        return dp[0][0]
```

动态规划，自顶向下

```python
class Solution:
    def minimumTotal(self, triangle) -> int:
        #自顶向下的动态规划
        dp = triangle[:]
        for i in range(1,len(triangle)):
            for j in range(len(triangle[i])):
                dp[i][j]+=min([dp[i-1][j] if j in range(0,len(triangle[i-1])) else float('inf'),
                            dp[i-1][j-1] if j-1 in range(0,len(triangle[i-1])) else float('inf')])
        return min(dp[-1])
```

## 递归和动规关系

递归是一种程序的实现方式：函数的自我调用

```python
def f(x):
    f(x-1)
```

动态规划：是一种解决问 题的思想，大规模问题的结果，是由小规模问 题的结果运算得来的。动态规划可用递归来实现(Memorization Search)

## 使用场景

满足两个条件

- 满足以下条件之一
  - 求最大/最小值（Maximum/Minimum ）
  - 求是否可行（Yes/No ）
  - 求可行个数（Count(\*) ）
- 满足不能排序或者交换（Can not sort / swap ）

如题：[longest-consecutive-sequence](https://leetcode-cn.com/problems/longest-consecutive-sequence/)  位置可以交换，所以不用动态规划

## 四点要素

1. **状态 State**
   - 灵感，创造力，存储小规模问题的结果
2. 方程 Function
   - 状态之间的联系，怎么通过小的状态，来算大的状态
3. 初始化 Intialization
   - 最极限的小状态是什么, 起点
4. 答案 Answer
   - 最大的那个状态是什么，终点

## 常见四种类型

1. Matrix DP (10%)
1. Sequence (40%)
1. Two Sequences DP (40%)
1. Backpack (10%)

> 注意点
>
> - 贪心算法大多题目靠背答案，所以如果能用动态规划就尽量用动规，不用贪心算法

## 1、矩阵类型（10%）

### [minimum-path-sum](https://leetcode-cn.com/problems/minimum-path-sum/)

> 给定一个包含非负整数的  *m* x *n*  网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

思路：动态规划
1、state: f[x][y]从起点走到 x,y 的最短路径
2、function: f[x][y] = min(f[x-1][y], f[x][y-1]) + A[x][y]
3、intialize: f[0][0] = A[0][0]、f[i][0] = sum(0,0 -> i,0)、 f[0][i] = sum(0,0 -> 0,i)
4、answer: f[n-1][m-1]

```python
class Solution:
    #----------------------------------
    #自下而上的一维动态规划
    def minPathSum_dp1(self, grid):
        m,n = len(grid),len(grid[0])
        dp = [float('inf')]*n
        for i in range(m-1,-1,-1):
            for j in range(n-1,-1,-1):
                if i==m-1 and j==n-1:
                    dp[j] = grid[i][j]
                #最后一列，只能向下移动
                elif i!=m-1 and j==n-1:
                    dp[j] = grid[i][j]+dp[j]
                #最后一行，只能向右移动
                elif i==m-1 and j!=n-1:
                    dp[j] = grid[i][j]+dp[j+1]
                else:
                    dp[j] = grid[i][j]+min(dp[j],dp[j+1])
        return dp[0]
    #----------------------------------
    #自下而上的二维动态规划
    def minPathSum_dp2(self, grid):
        m,n = len(grid),len(grid[0])
        dp = [[float('inf')]*n for _ in range(m)]
        for i in range(m-1,-1,-1):
            for j in range(n-1,-1,-1):
                if i==m-1 and j==n-1:
                    dp[i][j] = grid[i][j]
                else:
                    dp[i][j] = grid[i][j]+\
                        min(dp[i+1][j] if i+1 in range(m) else float('inf'),\
                            dp[i][j+1] if j+1 in range(n) else float('inf'))
        return dp[0][0]
    #----------------------------------
    #二维动态规划,但用grid而不需要额外的存储空间
    def minPathSum(self, grid):
        m,n = len(grid),len(grid[0])
        for i in range(m-1,-1,-1):
            for j in range(n-1,-1,-1):
                if i==m-1 and j==n-1:
                    continue
                #最后一列，只能向下移动
                elif i!=m-1 and j==n-1:
                    grid[i][j] += grid[i+1][j]
                #最后一行，只能向右移动
                elif i==m-1 and j!=n-1:
                    grid[i][j] += grid[i][j+1]
                else:
                    grid[i][j] += min(grid[i][j+1],grid[i+1][j])
        return grid[0][0]
    #----------------------------------
    #暴力法
    def minPathSum_bruteForce(self, grid):
        def _cacu_cost(grid,i,j):
            m,n = len(grid),len(grid[0])
            if i==m-1 and j==n-1:
                return grid[i][j]
            elif i>=m or j>=n:
                return float('inf')
            else:
                return grid[i][j]+min(_cacu_cost(grid,i+1,j),_cacu_cost(grid,i,j+1))
        return _cacu_cost(grid,0,0)
        
```

### [unique-paths](https://leetcode-cn.com/problems/unique-paths/)

> 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
> 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
> 问总共有多少条不同的路径？

```python
def uniquePaths(self, m: int, n: int) -> int:
    #动态方程:dp[i][j] = dp[i-1][j] + dp[i][j-1]
    dp = [0]*n
    for i in range(m-1,-1,-1):
        for j in range(n-1,-1,-1):
            if i==m-1 and j==n-1:
                dp[j] = 1
            #最后一行,只能向右走
            elif i==m-1 and j!=n-1:
                dp[j] = dp[j+1]
            #最后一列,只能向下走
            elif i!=m-1 and j==n-1:
                dp[j] = dp[j]
            else:
                dp[j] +=dp[j+1]
    return dp[0]
```

### [unique-paths-ii](https://leetcode-cn.com/problems/unique-paths-ii/)

> 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
> 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
> 问总共有多少条不同的路径？
> 现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

```python
def uniquePathsWithObstacles(self, obstacleGrid) -> int:
    m,n = len(obstacleGrid),len(obstacleGrid[0])
    dp = [0]*n
    for i in range(m-1,-1,-1):
        for j in range(n-1,-1,-1):
            if obstacleGrid[i][j] == 1:
                dp[j] = 0
                continue
            if i==m-1 and j==n-1:
                dp[j] = 1
            #最后一行,只能向右走
            elif i==m-1 and j!=n-1:
                dp[j] = dp[j+1]
            #最后一列,只能向下走
            elif i!=m-1 and j==n-1:
                dp[j] = dp[j]
            else:
                dp[j] +=dp[j+1]
    return dp[0]
```

## 2、序列类型（40%）

### [climbing-stairs](https://leetcode-cn.com/problems/climbing-stairs/)

> 假设你正在爬楼梯。需要  *n*  阶你才能到达楼顶。

```python
def climbStairs(self, n: int) -> int:
    if n<=2:
        return n
    dp = [0]*n
    dp[0],dp[1] = 1,2
    for i in range(2,n):
        dp[i] = dp[i-1]+dp[i-2]
    return dp[-1]
```

### [jump-game](https://leetcode-cn.com/problems/jump-game/)

> 给定一个非负整数数组，你最初位于数组的第一个位置。
> 数组中的每个元素代表你在该位置可以跳跃的最大长度。
> 判断你是否能够到达最后一个位置。

```python
class Solution:
    def canJump_dp(self, nums) -> bool:
        #动态规划:自底向上
        dp = [False]*len(nums)
        dp[-1] = True
        cur_succ_idx = len(nums)-1#当前最近的成功索引,能到达该索引表示能达到最后一个位置
        for i in range(len(nums)-2,-1,-1):
            if i+nums[i]>=cur_succ_idx:
                cur_succ_idx = i
                dp[i] = True
        return dp[0]
    def canJump_greedy(self, nums) -> bool:
        # 贪心算法:实时维护最远可达位置,若x可达且可跳跃y,那么最远可达x+y,所有x+i,i<=y也可达
        max_jump = 0
        for i in range(len(nums)):
            if i<=max_jump and i+nums[i]>max_jump:
                max_jump =i+nums[i]
            if max_jump>=len(nums)-1:
                return True
        return False

```

### [jump-game-ii](https://leetcode-cn.com/problems/jump-game-ii/)

> 给定一个非负整数数组，你最初位于数组的第一个位置。
> 数组中的每个元素代表你在该位置可以跳跃的最大长度。
> 你的目标是使用最少的跳跃次数到达数组的最后一个位置。

```python
class Solution:
    def jump_greedy(self, nums) -> int:
        #贪心算法
        n = len(nums)
        cur_max_idx,next_max_idx,step = 0,0,0
        #不访问最后一个元素,防止最后一步为最远可达位置时step+1
        for i in range(n-1):
            if i<=next_max_idx:#i在下一步可达范围内
                if i+nums[i]>next_max_idx:
                    next_max_idx = i+nums[i]
                #到达当前步最远可达位置
                if i==cur_max_idx:
                    step+=1
                    cur_max_idx = next_max_idx
        return step
    #自上而下的动态规划,超出时间限制
    def jump_dp(self, nums) -> int:
        dp = list(range(len(nums)))
        for i in range(1,len(nums)):
            for j in range(i):
                if j+nums[j]>=i:
                    dp[i] = min(dp[i],dp[j]+1)
        print(dp)
        return dp[-1]
```

### [longest-palindromic-substring](https://leetcode-cn.com/problems/longest-palindromic-substring/)

> 找到 s 中最长的回文子串。

```python
def longestPalindrome(self, s: str) -> str:
    #动态规划
    res = ''
    dp = [[False]*len(s) for i in range(len(s))]
    for l in range(len(s)):
        for i in range(len(s)-l):
            j = i+l
            if i==j:
                dp[i][j] = True
            elif i+1==j:
                dp[i][j] = (s[i]==s[j])
            else:
                dp[i][j] = (s[i]==s[j] and dp[i+1][j-1])
            if dp[i][j] and l+1>len(res):
                res = s[i:j+1]
    return res
```
### [palindrome-partitioning](https://leetcode-cn.com/problems/palindrome-partitioning/)

> 给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。
> 返回 s 所有可能的分割方案。

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

### [palindrome-partitioning-ii](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)

> 给定一个字符串 _s_，将 _s_ 分割成一些子串，使每个子串都是回文串。
> 返回符合要求的最少分割次数。

```python
def minCut(self, s: str) -> int:
    # 动态规划,dp[i]表示s[i:]的最小分割数
    # 转移方程dp[j] = 在j出分隔(j>=i)+s[j]
    dp = [len(s)+1]*len(s)+[-1]

    for left_i in range(len(s)-1,-1,-1):
        #遍历所有可能的分隔点,取最小
        for right_i in range(len(s)-1,left_i-1,-1):
            if s[left_i:right_i+1]==s[left_i:right_i+1][::-1]:
                dp[left_i] = min(dp[left_i] ,1+dp[right_i+1])
    return dp[0]
```

注意点

- 判断回文字符串时，可以提前用动态规划算好，减少时间复杂度

### [longest-increasing-subsequence](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

> 给定一个无序的整数数组，找到其中最长上升子序列的长度。

```python
def lengthOfLIS(self, nums) -> int:
    # 动态规划:dp[i]表示以i起始的最长上升子序列的长度
    # 转移方程:dp[j] = max(1+dp[j]) j>i and nums[i]<nums[j]
    n = len(nums)
    if n<2:
        return n
    dp = [1]*n
    for i in range(n-2,-1,-1):
        for j in range(i,n):
            if nums[i]<nums[j]:
                dp[i] = max(dp[i],1+dp[j])
    return max(dp)
```

### [word-break](https://leetcode-cn.com/problems/word-break/)

> 给定一个**非空**字符串  *s*  和一个包含**非空**单词列表的字典  *wordDict*，判定  *s*  是否可以被空格拆分为一个或多个在字典中出现的单词。

```python
def wordBreak(self, s, wordDict) -> bool:
    # 动态规划:dp[i] 表示前i个字符是否可以被切分
    # 转移方程:dp[i] = dp[j] && s[j+1~i] in wordDict
    dp = [False]*len(s)
    for right_i in range(len(s)):
        for cut_i in range(right_i+1):
            #s[:cur_i]可切分 && s[cur_i:right_i+1]可切分
            if (cut_i==0 or dp[cut_i-1]) and s[cut_i:right_i+1] in wordDict:
                dp[right_i] = True
                break
    return dp[-1]
```

小结

常见处理方式是给 0 位置占位，这样处理问题时一视同仁，初始化则在原来基础上 length+1，返回结果 f[n]

- 状态可以为前 i 个
- 初始化 length+1
- 取值 index=i-1
- 返回值：f[n]或者 f[m][n]

## Two Sequences DP（40%）

### [longest-common-subsequence](https://leetcode-cn.com/problems/longest-common-subsequence/)

> 给定两个字符串  text1 和  text2，返回这两个字符串的最长公共子序列。
> 一个字符串的   子序列   是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
> 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。

```python
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    m,n = len(text1),len(text2)
    #dp[i][j]表示text1[:i]和text2[:j]的最长公共子串
    dp = [[0]*(n+1) for _ in range(m+1)]
    #dp的第一行和第一列为base case,答案为0,不遍历
    for i in range(1,m+1):
        for j in range(1,n+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1]+1
            else:
                dp[i][j] = max(dp[i-1][j],dp[i][j-1])
    return dp[-1][-1]
```

注意点

- 从 1 开始遍历到最大长度
- 索引需要减一

### [edit-distance](https://leetcode-cn.com/problems/edit-distance/)

> 给你两个单词  word1 和  word2，请你计算出将  word1  转换成  word2 所使用的最少操作数  
> 你可以对一个单词进行如下三种操作：
> 插入一个字符
> 删除一个字符
> 替换一个字符

思路：和上题很类似，相等则不需要操作，否则取删除、插入、替换最小操作次数的值+1

```python
def minDistance(self, word1: str, word2: str) -> int:
        m,n = len(word1),len(word2)
        if m*n == 0:
            return m+n
        #dp[i][j]为word1[:i]和word2[:j]的编辑距离
        dp = [[0]*(n+1) for _ in range(m+1)]
        #边界状态初始化:空串和s的编辑距离为len(s)
        for i in range(n+1):
            dp[0][i] = i
        for i in range(m+1):
            dp[i][0] = i
        #状态转移方程 
        # (1)w1[i]==w2[j] : dp[i][j]=min(dp[i-1][j-1],dp[i-1][j]+1,dp[i][j-1]+1)
        # (2)w1[i]!=w2[j] : dp[i][j]=1+min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1])
        for i in range(1,m+1):
            for j in range(1,n+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j]=min(dp[i-1][j-1],dp[i-1][j]+1,dp[i][j-1]+1)
                else:
                    dp[i][j]=1+min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1])
        return dp[-1][-1]
```

说明

> 另外一种做法：MAXLEN(a,b)-LCS(a,b)

## 零钱和背包（10%）

### [coin-change](https://leetcode-cn.com/problems/coin-change/)

> 给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回  -1。

思路：和其他 DP 不太一样，i 表示钱或者容量

```python
def coinChange(self, coins, amount) -> int:
    #dp[i]表示要合成i金额需要的最少硬币个数
    dp = [float('inf')]*(amount+1)
    #边界状态初始化
    dp[0] = 0
    # 动态转移方程:dp[i] = dp[i-c]+1
    for coin in coins:#默认coins有序
        for _amount in range(coin,amount+1):
            dp[_amount] = min(dp[_amount],dp[_amount-coin]+1)
    return dp[-1] if dp[-1] != float('inf') else -1 
```

注意

> dp[i-a[j]] 决策 a[j]是否参与

### [backpack](https://www.lintcode.com/problem/backpack/description)

> 在 n 个物品中挑选若干物品装入背包，最多能装多满？假设背包的大小为 m，每个物品的大小为 A[i]

```python
class Solution:
    """
    @param m: An integer m denotes the size of a backpack
    @param A: Given n items with size A[i]
    @return: The maximum size
    """
    def backPack(self, m, A):
        #dp[i][j]表示能否用A[:i+1]填满j
        dp = [[False]*(m+1) for i in range(len(A))]
        
        # 边界状态初始化:一定能填满空背包
        for i in range(len(A)):
            dp[i][0] = True
        
        #状态转移方程
        # dp[i][j] = (1)dp[i-1][j](2)dp[i-1][j-A[i]]
        for i in range(len(A)):
            for j in range(1,m+1):
                #A[:i-1]已经可以填满j
                dp[i][j] = dp[i-1][j]
                #A[:i-1]填满j-A[i-1]
                if not dp[i][j] and A[i-1]<=j:
                    dp[i][j] = dp[i-1][j-A[i-1]]
            # print(dp[i])
        return max([i for i in range(m+1) if dp[-1][i]])

```

### [backpack-ii](https://www.lintcode.com/problem/backpack-ii/description)

> 有 `n` 个物品和一个大小为 `m` 的背包. 给定数组 `A` 表示每个物品的大小和数组 `V` 表示每个物品的价值.
> 问最多能装入背包的总价值是多大?

思路：f[i][j] 前 i 个物品，装入 j 背包 最大价值

```python
class Solution:
    """
    @param m: An integer m denotes the size of a backpack
    @param A: Given n items with size A[i]
    @param V: Given n items with value V[i]
    @return: The maximum value
    """
    def backPackII(self, m, A, V):
        #dp[i][j]表示用A[:i+1]填背包至j的最大价值
        dp = [[0]*(m+1) for i in range(len(A))]
        
        # 边界状态初始化:空背包的价值为0
        for i in range(len(A)):
            dp[i][0] = 0
        
        #状态转移方程
        # dp[i][j] = max(dp[i-1][j],dp[i-1][j-A[i]]+V[i])
        for i in range(len(A)):
            for j in range(1,m+1):
                #至少为使用A[:i-1]填充的价值
                dp[i][j] = dp[i-1][j]
                #A[:i-1]填满j-A[i-1]
                if A[i-1]<=j:
                    dp[i][j] = max(dp[i][j], dp[i-1][j-A[i-1]]+V[i-1])
        return max(dp[-1])

```

## 练习

Matrix DP (10%)

- [ ] [triangle](https://leetcode-cn.com/problems/triangle/)
- [ ] [minimum-path-sum](https://leetcode-cn.com/problems/minimum-path-sum/)
- [ ] [unique-paths](https://leetcode-cn.com/problems/unique-paths/)
- [ ] [unique-paths-ii](https://leetcode-cn.com/problems/unique-paths-ii/)

Sequence (40%)

- [ ] [climbing-stairs](https://leetcode-cn.com/problems/climbing-stairs/)
- [ ] [jump-game](https://leetcode-cn.com/problems/jump-game/)
- [ ] [jump-game-ii](https://leetcode-cn.com/problems/jump-game-ii/)
- [ ] [palindrome-partitioning-ii](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)
- [ ] [longest-increasing-subsequence](https://leetcode-cn.com/problems/longest-increasing-subsequence/)
- [ ] [word-break](https://leetcode-cn.com/problems/word-break/)

Two Sequences DP (40%)

- [ ] [longest-common-subsequence](https://leetcode-cn.com/problems/longest-common-subsequence/)
- [ ] [edit-distance](https://leetcode-cn.com/problems/edit-distance/)

Backpack & Coin Change (10%)

- [ ] [coin-change](https://leetcode-cn.com/problems/coin-change/)
- [ ] [backpack](https://www.lintcode.com/problem/backpack/description)
- [ ] [backpack-ii](https://www.lintcode.com/problem/backpack-ii/description)
