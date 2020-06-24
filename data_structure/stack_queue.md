# 栈和队列

## 简介

栈的特点是后入先出

![image.png](https://img.fuiboom.com/img/stack.png)

根据这个特点可以临时保存一些数据，之后用到依次再弹出来，常用于 DFS 深度搜索

队列一般常用于 BFS 广度搜索，类似一层一层的搜索

## Stack 栈

[min-stack](https://leetcode-cn.com/problems/min-stack/)

> 设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。

思路：用两个栈实现，一个最小栈始终保证最小值在顶部

```python
class MinStack:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        #设置辅助栈,存储当前栈的最小值
        self.min_stack = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        if len(self.min_stack) == 0:
            self.min_stack.append(x)
        else:
            self.min_stack.append(min(self.min_stack[-1],x))

    def pop(self) -> None:
        self.stack = self.stack[:-1]
        self.min_stack = self.min_stack[:-1]

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
```

[evaluate-reverse-polish-notation](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)

> **波兰表达式计算** > **输入:** ["2", "1", "+", "3", "*"] > **输出:** 9
> **解释:** ((2 + 1) \* 3) = 9

思路：通过栈保存原来的元素，遇到表达式弹出运算，再推入结果，重复这个过程

```python
def evalRPN(self, tokens) -> int:
    operators = '+-*/'
    #通过栈保存原来的元素
    tokens_stack = []
    for token in tokens:
        #遇到表达式弹出运算，再推入结果
        if token in operators:
            left,right = tokens_stack[-2:]
            #用eval()计算结果
            tokens_stack = tokens_stack[:-2]+[str(int(eval(left+token+right)))]
        else:
            tokens_stack.append(token)
    return int(tokens_stack[0])


# #从后往前遍历,会超时
# def evalRPN_(self, tokens: List[str]) -> int:
#     if len(tokens)==1:
#         return int(tokens[0])
#     operator_list = ['+','-','*','/']
#     #op:先提取运算符
#     op = tokens.pop()
#     #right:根据运算符和数字的个数找到right子串,递归计算结果
#     if tokens[-1] not in operator_list:
#         right = int(tokens.pop())
#     else:
#         right_op_nums = 0
#         right_dig_nums = 0
#         right_tokens = []
#         while len(right_tokens)==0 or (right_dig_nums+right_op_nums>0):
#             token = tokens.pop()
#             right_tokens = [token]+right_tokens
#             if token in operator_list:
#                 right_op_nums+=1
#             else:
#                 right_dig_nums+=1
#             if right_dig_nums==2:
#                 if right_op_nums==1:
#                     break
#                 right_op_nums-=1
#                 right_dig_nums-=1
#         right = self.evalRPN(right_tokens)
#     #left:剩下的是left子串,递归计算结果
#     left = self.evalRPN(tokens)
#     #归并结果
#     if op=='+':return left+right
#     elif op=='-':return left-right
#     elif op=='*':return left*right
#     elif op=='/':return int(str(left/right).split('.')[0])
```

[decode-string](https://leetcode-cn.com/problems/decode-string/)

> 给定一个经过编码的字符串，返回它解码后的字符串。
> s = "3[a]2[bc]", 返回 "aaabcbc".
> s = "3[a2[c]]", 返回 "accaccacc".
> s = "2[abc]3[cd]ef", 返回 "abcabccdcdcdef".

思路：通过栈辅助进行操作

```python
def decodeString(self, s: str) -> str:
    res_stack = []
    for _s in s:
        if _s != ']':
            res_stack.append(_s)
            continue
        encoded_string = ''
        while res_stack[-1].isalpha():
            encoded_string=res_stack.pop()+encoded_string
        res_stack.pop()
        multiple = ''
        while res_stack and res_stack[-1].isdigit():
            multiple=res_stack.pop()+multiple
        # print(encoded_string)
        res_stack+=(encoded_string*int(multiple))
    return ''.join(res_stack)
```

利用栈进行 DFS 递归搜索模板

```python
def decodeString(self, s: str) -> str:
    #递归法:将 [ 和 ] 分别作为递归的开启与终止条件：
    def dfs(s,i):
        res,multi = "",0
        while i<len(s):
            #解码重复次数
            if s[i].isdigit():
                multi = multi*10 + int(s[i])
            #开启一层递归
            elif s[i]=='[':
                #解码方括号中的encoded_string
                i,dencoded_string = dfs(s,i+1)
                #重复次数
                res+=(multi*dencoded_string)
                #
                multi=0
            #结束递归
            elif s[i]==']':
                #返回上一层递归的索引位置
                return i,res
            elif s[i].isalpha():
                res+=s[i]
            i+=1
        return res
    return dfs(s,0)
```

[binary-tree-inorder-traversal](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

> 给定一个二叉树，返回它的*中序*遍历。

```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def inorderTraversal(self, root: TreeNode):
        if root is None:
            return []
        res = []
        stack = []
        cur_node = root
        while len(stack) or cur_node:
            #左子树
            while cur_node:
                stack.append(cur_node)
                cur_node = cur_node.left
            #访问根节点
            cur_node = stack.pop()
            res.append(cur_node.val)
            #右子树
            cur_node = cur_node.right
        return res
```

[clone-graph](https://leetcode-cn.com/problems/clone-graph/)

> 给你无向连通图中一个节点的引用，请你返回该图的深拷贝（克隆）。

```python
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = []):
        self.val = val
        self.neighbors = neighbors

class Solution:
    def __init__(self):
        # 使用一个 HashMap 存储所有已被访问和复制的节点
        self.clonedNodes = {}
    # 连通图:图中任意两点都是连通的
    def cloneGraph_bfs(self, node: 'Node') -> 'Node':
        if node is None:
            return None
        cloned = {}
        from collections import deque
        #使用队列存放未分配邻居结点的克隆节点
        queue = deque([node])
        cloned[node] = Node(node.val,[])
        #----------------------------
        while queue:
            cur_n = queue.popleft()
            for neighbor in cur_n.neighbors:
                if neighbor not in cloned:
                    cloned[neighbor]=Node(neighbor.val,[])
                    queue.append(neighbor)
                cloned[cur_n].neighbors+=[cloned[neighbor]]
        return cloned[node]



    def cloneGraph_dfs(self, node: 'Node') -> 'Node':
        if node is None:
            return None
        if node not in self.clonedNodes: 
            # 进入递归前，先创建克隆节点并保存在 HashMap , 避免死循环。
            self.clonedNodes[node] = Node(node.val)
            self.clonedNodes[node].neighbors=[self.cloneGraph(neighbor) for neighbor in node.neighbors]
        return self.clonedNodes[node]

```

[number-of-islands](https://leetcode-cn.com/problems/number-of-islands/)

> 给定一个由  '1'（陆地）和 '0'（水）组成的的二维网格，计算岛屿的数量。一个岛被水包围，并且它是通过水平方向或垂直方向上相邻的陆地连接而成的。你可以假设网格的四个边均被水包围。

思路：通过深度搜索遍历可能性（注意标记已访问元素）

```python
class Solution:
    def numIslands(self, grid) -> int:
        #并查集数据结构
        class UnionFind:
            def __init__(self, grid):
                m, n = len(grid), len(grid[0])
                #联通分量的个数
                self.count = 0
                #self.parent代表邻接结点
                self.parent = [-1] * (m * n)
                for i in range(m):
                    for j in range(n):
                        if grid[i][j] == "1":
                            # 初始化邻接结点为自己
                            self.parent[i * n + j] = i * n + j
                            self.count += 1
            
            def find(self, i):
                #连通分量的id是其中某个结点的id
                # 查找结点属于哪个集合:查找连通分量的id,依次遍历邻接结点直到连通分量的id等于结点id
                if self.parent[i] != i:
                    self.parent[i] = self.find(self.parent[i])
                return self.parent[i]
            
            def union(self, x, y):
                rootx = self.find(x)
                rooty = self.find(y)
                if rootx != rooty:
                    self.parent[rooty] = rootx
                    self.count -= 1
            
            def getCount(self):
                return self.count
        #---------------------------------
        if len(grid)==0 or len(grid[0])==0:
            return 0
        row,col = len(grid),len(grid[0])
        uf = UnionFind(grid)
        for i in range(row):
            for j in range(col):
                if grid[i][j] != '1':
                    continue
                for r,c in [[i+1,j],[i,j+1],[i-1,j],[i,j-1]]:
                    if r in range(row) and c in range(col) and grid[r][c]=='1':
                        uf.union(r*col+c,i*col+j)
        return uf.getCount()


    def numIslands_search(self, grid) -> int:
        # 将二维网格看成无向图.扫描网格,利用深/广度优先搜索将所有1周围的1置为0。
        # 岛屿个数为深/广度优先搜索的个数
        def _dfs(grid,i,j):
            grid[i][j] = '0'
            row,col = len(grid),len(grid[0])
            for r,c in [[i+1,j],[i,j+1],[i-1,j],[i,j-1]]:
                if r in range(row) and c in range(col) and grid[r][c]=='1':
                    _dfs(grid,r,c)
        def _bfs(grid,i,j):
            grid[i][j] = '0'
            row,col = len(grid),len(grid[0])
            from collections import deque
            queue = deque([[i,j]])
            while queue:
                i,j = queue.popleft()
                for r,c in [[i+1,j],[i,j+1],[i-1,j],[i,j-1]]:
                    if r in range(row) and c in range(col) and grid[r][c]=='1':
                        queue.append([r,c])
                        grid[r][c] = '0'
        #--------------------------------
        if len(grid)==0 or len(grid[0])==0:
            return 0
        row,col = len(grid),len(grid[0])
        res = 0
        for i in range(row):
            for j in range(col):
                if grid[i][j] == '1':
                    res+=1
                    # _dfs(grid,i,j)
                    _bfs(grid,i,j)
        return res
```

[largest-rectangle-in-histogram](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

> 给定 _n_ 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
> 求在该柱状图中，能够勾勒出来的矩形的最大面积。

思路：求以当前柱子为高度的面积，即转化为寻找小于当前值的左右两边值

![image.png](https://img.fuiboom.com/img/stack_rain.png)

用栈保存小于当前值的左的元素

![image.png](https://img.fuiboom.com/img/stack_rain2.png)

```python
class Solution:
    def largestRectangleArea(self, heights) -> int:#-----------------------
        #方法1:单调栈
        #左->右遍历并维护一个单调递增栈,栈内元素下面的元素就是左侧第一个小于它的元素
        n = len(heights)
        mono_stack,left = [],[]
        for left_i in range(n):
            #若大于当前高度，则出栈以维护栈的单调性
            while len(mono_stack) and mono_stack[-1][0]>=heights[left_i]:
                mono_stack.pop()
            left.append(mono_stack[-1][1] if len(mono_stack) else -1)
            mono_stack.append([heights[left_i],left_i])
        #右->左遍历
        mono_stack,right = [],[]
        for right_i in range(n-1,-1,-1):
            #若大于当前高度，则出栈以维护栈的单调性
            while len(mono_stack) and mono_stack[-1][0]>=heights[right_i]:
                mono_stack.pop()
            right.append(mono_stack[-1][1] if len(mono_stack) else n)
            mono_stack.append([heights[right_i],right_i])
        right = right[::-1]
        #根据每个height的左右边界结算面积
        res = 0
        for i in range(n):
            res = max(res,heights[i]*(right[i]-left[i]-1))
        return res
        #-----------------------
        #方法2:枚举高
        res = 0
        n = len(heights)
        for i in range(n):
            height = heights[i]
            start,end = i,i
            while start>=0 and heights[start]>=height:
                start-=1
            while end<n and heights[end]>=height:
                end+=1
            _res = (end-start-1)*height
            res = max(_res,res)
        return res
            
        return res
        #-----------------------
        #方法3:枚举宽
        res = 0
        n = len(heights)
        for start in range(n):
            for end in range(start,n):
                res = max(min(heights[start:end+1])*(end+1-start),res)
        return res
```

## Queue 队列

常用于 BFS 宽度优先搜索

[implement-queue-using-stacks](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

> 使用栈实现队列

```python
class MyQueue:

    def __init__(self):
        """Initialize your data structure here."""
        self.queue = []
        #辅助栈,进行peek和pop操作
        self.helper = []

    def push(self, x: int) -> None:
        """Push element x to the back of queue."""
        self.queue.append(x)#使用相当于stack的pop


    def pop(self) -> int:
        """Removes the element from in front of queue and returns that element."""
        if len(self.helper)==0:
            while len(self.queue):#判断栈是否为空
                self.helper.append(self.queue.pop())#出/入栈
        return self.helper.pop()


    def peek(self) -> int:
        """Get the front element."""
        if len(self.helper)==0:
            while len(self.queue):#判断栈是否为空
                self.helper.append(self.queue.pop())#出/入栈
        return self.helper[-1]


    def empty(self) -> bool:
        """Returns whether the queue is empty."""
        return len(self.queue)==0 and len(self.helper)==0
```

二叉树层次遍历

```python
def levelorder(self, root):
    if root is None:
        return []
    res = []
    queue = [root]
    while queue:
        next_level_queue = []
        for node in queue:
            res += [node.val]
            next_level_queue += [node.left] if node.left else []
            next_level_queue += [node.right] if node.right else []
        queue = next_level_queue
    return res
```

[01-matrix](https://leetcode-cn.com/problems/01-matrix/)

> 给定一个由 0 和 1 组成的矩阵，找出每个元素到最近的 0 的距离。
> 两个相邻元素间的距离为 1

```python
class Solution:
    def updateMatrix_bfs(self, matrix):
        #思路:以0为中心,利用队列进行广度优先搜索
        m,n = len(matrix),len(matrix[0])
        distance = [[0]*n for _ in range(m)]
        #找到所有0(1)标记已访问避免死循环(2)加入队列
        visited = [(i,j) for i in range(m) for j in range(n) if matrix[i][j] == 0]
        queue = collections.deque(visited)
        visited = set(visited)#如果不是set而是list就会超时
        #广度优先搜索
        while queue:
            i,j = queue.popleft()
            for neigh_i,neigh_j in [[i-1,j],[i+1,j],[i,j-1],[i,j+1]]:
                if neigh_i not in range(m) or neigh_j not in range(n) or (neigh_i,neigh_j) in visited:
                    continue
                distance[neigh_i][neigh_j] = distance[i][j]+1
                queue.append((neigh_i,neigh_j))
                visited.add((neigh_i,neigh_j))
        return distance
    def updateMatrix_dp(self, matrix):
        m,n = len(matrix),len(matrix[0])
        distance = [[m*n+1]*n for _ in range(m)]
        #动态规划
        for i in range(m):
            for j in range(n):
                if matrix[i][j]==0:
                    distance[i][j] = 0
        #向左上方
        for i in range(m):
            for j in range(n):
                #动态转移方程
                if i-1>=0:
                    distance[i][j] = min([distance[i][j],distance[i-1][j]+1])
                if j-1>=0:
                    distance[i][j] = min([distance[i][j],distance[i][j-1]+1])
        #向右下方
        for i in range(m-1,-1,-1):
            for j in range(n-1,-1,-1):
                if i+1<m:
                    distance[i][j] = min([distance[i][j],distance[i+1][j]+1])
                if j+1<n:
                    distance[i][j] = min([distance[i][j],distance[i][j+1]+1])
        return distance
```

## 总结

- 熟悉栈的使用场景
  - 后出先出，保存临时值
  - 利用栈 DFS 深度搜索
- 熟悉队列的使用场景
  - 利用队列 BFS 广度搜索

## 练习

- [ ] [min-stack](https://leetcode-cn.com/problems/min-stack/)
- [ ] [evaluate-reverse-polish-notation](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)
- [ ] [decode-string](https://leetcode-cn.com/problems/decode-string/)
- [ ] [binary-tree-inorder-traversal](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)
- [ ] [clone-graph](https://leetcode-cn.com/problems/clone-graph/)
- [ ] [number-of-islands](https://leetcode-cn.com/problems/number-of-islands/)
- [ ] [largest-rectangle-in-histogram](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)
- [ ] [implement-queue-using-stacks](https://leetcode-cn.com/problems/implement-queue-using-stacks/)
- [ ] [01-matrix](https://leetcode-cn.com/problems/01-matrix/)
