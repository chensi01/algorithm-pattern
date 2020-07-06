# 哈希表

## 简介

哈希表是一种数据结构，使用哈希函数组织数据，以支持快速插入和搜索

哈希函数将键映射到存储桶。
[如何设计键](https://leetcode-cn.com/explore/learn/card/hash-table/206/practical-application-design-the-key/824/)

保持数组中的每个元素与其索引相互对应的最好方法

有两种不同类型的哈希表：

哈希集合:集合数据结构的实现之一，用于存储非重复值。python中是set()
哈希映射:是映射 数据结构的实现之一，用于存储(key, value)键值对。python中是dict()

[two-sum](https://leetcode-cn.com/problems/two-sum/)

> 两数之和

思路：使用哈希映射,空间换时间

```python
class Solution:
    def twoSum(self, nums, target):
        #方法1:暴力法,遍历每个元素x并查找target-x元素
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                if nums[i]==target-nums[j]:
                    return [i,j]
        return False
        # 方法2:为方便查找使用哈希映射,空间换时间
        hash_map = {}
        for i in range(len(nums)):
            x,y = nums[i],target-nums[i]
            #一定要先查找再插入,避免x=y时的冲突覆盖
            #查找target-x元素
            if y in hash_map:
                return [hash_map[y],i]
            #插入
            hash_map[x] = i
            

```



[4sum-ii](https://leetcode-cn.com/problems/4sum-ii/)

> 四数之和

思路：将四数之和转化为两个两数之和

```python
def fourSumCount(self, A, B, C, D) -> int:
    #四个for遍历时间复杂度O(N^4)
    #思路:将四数之和转化为两个两数之和:改为两个两个for遍历时间复杂度O(N^2)
    N = len(A)
    res = 0
    from collections import defaultdict
    hash_dict_AB = defaultdict(int)
    
    for i in range(N):
        for j in range(N):
            hash_dict_AB[A[i]+B[j]]+=1
    
    for k in range(N):
        for l in range(N):
            n = C[k] + D[l]
            res+=hash_dict_AB[-n]

    return res

```




[find-duplicate-subtrees](https://leetcode-cn.com/problems/find-duplicate-subtrees/)

> 寻找重复的子树

思路：序列化二叉树

```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def findDuplicateSubtrees(self, root: TreeNode):
        hash_map = {}
        res = []
        def _dfs(node):
            if node is None:
                return '#'
            node_rep = [str(node.val),_dfs(node.left),_dfs(node.right)]
            node_rep = ','.join(node_rep)
            hash_map[node_rep]=hash_map.get(node_rep,0)+1
            if hash_map[node_rep]==2:
                res.append(node)
            return node_rep
        _dfs(root)
        return res
```


[valid-sudoku](https://leetcode-cn.com/problems/valid-sudoku/)

> 有效的数独

思路：一次遍历

```python
def isValidSudoku(self, board) -> bool:
    from collections import defaultdict
    hash_map_row = defaultdict(dict)
    hash_map_col = defaultdict(dict)
    hash_map_box = defaultdict(dict)
    for i in range(9):
        for j in range(9):
            if board[i][j]=='.':
                continue
            n = int(board[i][j])

            hash_map_row[i][n]=hash_map_row[i].get(n,0)+1
            hash_map_col[j][n]=hash_map_col[j].get(n,0)+1
            if hash_map_row[i][n]>1 or hash_map_col[j][n]>1:
                return False

            box_idx = (i//3)*3+j//3
            hash_map_box[box_idx][n]=hash_map_box[box_idx].get(n,0)+1
            if hash_map_box[box_idx][n]>1:
                return False
    
    return True

```


[top-k-frequent-elements](https://leetcode-cn.com/problems/top-k-frequent-elements/)

> 构建哈希表+堆排序
> 再python中使用字典结构用作哈希表，用 collections 库中的 Counter 方法去构建哈希表
> python的堆排序:使用 heapq 库中的 nlargest 方法

```python
class Solution:
    def topKFrequent_solu(self, nums, k: int) -> List[int]:
        hash_dict = collections.Counter(nums)
        return heapq.nlargest(k,hash_dict.keys(),key=hash_dict.get)

    def topKFrequent(self, nums, k: int) -> List[int]:
        hash_dict = {}
        for n in nums:
            hash_dict[n] = hash_dict.get(n,0)+1
        # O(nlogn):堆排序,归并排序
        num_freq = list(hash_dict.items())
        #构造堆
        def _heapify(nums,n,i,k=1):
            largest_idx = i
            left,right = 2*i+1,2*i+2
            #找到当前堆结构的最大值
            if left<n and nums[left][k]>nums[largest_idx][k]:
                largest_idx = left
            if right<n and nums[right][k]>nums[largest_idx][k]:
                largest_idx = right
            #如果当前堆顶不是最大值，则和子节点交换，并递归的构造子节点
            if largest_idx != i:
                nums[i],nums[largest_idx] = nums[largest_idx],nums[i]
                _heapify(nums,n,largest_idx)

        #自下而上构造最大堆
        n = len(num_freq)
        for i in range(n-1,-1,-1):
            _heapify(num_freq,n,i)
        #取出topk
        res = []
        for i in range(1,k+1):
            res +=[num_freq[0][0]]
            #拿掉堆顶元素
            num_freq[n-i],num_freq[0] = num_freq[0],num_freq[n-i]
            #重新构造
            _heapify(num_freq,n-i,0)
        return res

```