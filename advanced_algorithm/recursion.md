# 递归

## 介绍

将大问题转化为小问题，通过递归依次解决各个小问题

## 示例

[reverse-string](https://leetcode-cn.com/problems/reverse-string/)

> 编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组  `char[]`  的形式给出。

```python
class Solution:
    # 递归法
    def reverseString_recursive(self, s) -> None:
        left_i,right_i = 0,len(s)-1
        def _helper(s,left_i,right_i):
            if left_i<right_i:
                s[left_i],s[right_i] = s[right_i],s[left_i]
                _helper(s,left_i+1,right_i-1)
        _helper(s,left_i,right_i)
    # 双指针法
    def reverseString(self, s) -> None:
        left_i,right_i = 0,len(s)-1
        while left_i<right_i:
            s[left_i],s[right_i] = s[right_i],s[left_i]
            left_i,right_i = left_i+1,right_i-1
```

[swap-nodes-in-pairs](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

> 给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
> **你不能只是单纯的改变节点内部的值**，而是需要实际的进行节点交换。

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if head is None:
            return head
        left,right = head,head.next
        if right:
            tail = head.next.next
            head = right
            right.next = left
            left.next = self.swapPairs(tail)
        return head

```
[unique-binary-search-trees](https://leetcode-cn.com/problems/unique-binary-search-trees/)

> 给定一个整数 n，生成所有由 1 ... n 为节点所组成的二叉搜索树的种类。

```python
def numTrees(self, n: int) -> int:
    #dp[i]表示长度为i的二叉搜索树个数
    dp = [0]*(n+1)
    #边界状态
    dp[0],dp[1] = 1,1
    #动态转移方程:1~i-1为左子树,i+1~n为右子树。左右子树的笛卡尔积。
    # dp[i] = dp[i-1]*dp[n-i]
    #遍历结点个数i
    for i in range(2,n+1):
        #遍历可能的头结点
        for j in range(1,i+1):
            dp[i] += (dp[j-1]*dp[i-j])
    return dp[-1]
```
[unique-binary-search-trees-ii](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/)

> 给定一个整数 n，生成所有由 1 ... n 为节点所组成的二叉搜索树。

```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def generateTrees(self, n: int):
        #递归
        def _helper(start,end):
            if start>end:
                return [None]
            trees = []
            for head in range(start,end+1):
                left_trees = _helper(start,head-1)
                right_trees = _helper(head+1,end)
                for l in left_trees:
                    for r in right_trees:
                        tree = TreeNode(head)
                        tree.left = l
                        tree.right = r
                        trees+=[tree]
            return trees
        return _helper(1,n) if n else []

```

## 递归+备忘录

[fibonacci-number](https://leetcode-cn.com/problems/fibonacci-number/)

> 斐波那契数，通常用  F(n) 表示，形成的序列称为斐波那契数列。该数列由  0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
> F(0) = 0,   F(1) = 1
> F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
> 给定  N，计算  F(N)。

```python
def fib(self, N: int) -> int:
    if N<=1:
        return N
    dp = [0,1]
    for i in range(2,N+1):
        dp[0],dp[1] = dp[1],dp[0]+dp[1]
    return dp[-1]
```

## 练习

- [ ] [reverse-string](https://leetcode-cn.com/problems/reverse-string/)
- [ ] [swap-nodes-in-pairs](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)
- [ ] [unique-binary-search-trees-ii](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/)
- [ ] [fibonacci-number](https://leetcode-cn.com/problems/fibonacci-number/)
