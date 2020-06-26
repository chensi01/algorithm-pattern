# 二叉树

## 知识点

### 二叉树遍历

**前序遍历**：**先访问根节点**，再前序遍历左子树，再前序遍历右子树
**中序遍历**：先中序遍历左子树，**再访问根节点**，再中序遍历右子树
**后序遍历**：先后序遍历左子树，再后序遍历右子树，**再访问根节点**

注意点

- 以根访问顺序决定是什么遍历
- 左子树都是优先右子树

#### 前序递归

```python
def preorderRecursiveTraversal(self, root):
    if not root:
        return []
    res = []
    res += [root.val]
    res += self.preorderRecursiveTraversal(root.left)
    res += self.preorderRecursiveTraversal(root.right)
    return res
```

#### 前序非递归

```python
def preorderTraversal(self, root):
    if not root:
        return []
    res = []
    # 左子树
    node = root
    # stack用于存储未访问右子树的结点
    stack = []
    while node or len(stack):
        # 一直访问左子树
        while node:
            res += [node.val]
            stack += [node]
            node = node.left
        # 访问stack中结点的右子树
        node = stack.pop()
        node = node.right
    return res
```

#### 中序非递归

```python
def inorderTraversal(self, root):
    if not root:
        return []
    res = []
    # 左子树
    node = root
    # stack用于存储未访问根节点和右子树的结点
    stack = []
    while node or len(stack):
        # 一直访问左子树
        while node:
            stack += [node]
            node = node.left
        # 访问stack中结点的右子树
        node = stack.pop()
        res += [node.val]
        node = node.right
    return res

```

#### 后序非递归

```python
def postorderTraversal(self, root):
    if not root:
        return []
    res = []
    node = root
    # stack用于存储未访问根节点和右子树的结点
    stack = []
    #
    lastVisitNode = None
    while node or len(stack):
        # 访问左子树
        while node:
            stack += [node]
            node = node.left
        # 倒序访问stack中结点的右子树
        next_node = stack[-1]
        # 根节点必须在右节点弹出之后，再弹出：访问根节点的条件:该结点的右子树(1)已访问(2)为空
        if next_node.right is None or next_node.right == lastVisitNode:
            res += [next_node.val]
            # 标记该结点的右子树已访问
            lastVisitNode = next_node
            # 从stack中删除该结点
            stack.pop()
        # 该结点的右子树未被访问->访问右子树
        else:
            node = next_node.right
    return res
```

注意点

- 核心就是：根节点必须在右节点弹出之后，再弹出

#### DFS 深度搜索-从上到下

```python
def preorderTraversal_dfs(self, root):
    res = []
    if not root:
        return []

    # 深度优先搜索
    def _dfs(node, result):
        if node is not None:
            result += [node.val]
            _dfs(node.left, result)
            _dfs(node.right, result)

    _dfs(root, res)
    return res
```

#### DFS 深度搜索-从下向上（分治法）

```python
# 前序递归+深度优先搜索,从下到上,分治法:分段递归处理返回结果再合并
def preorderTraversal_divide_conquer(self, root):
    # 深度优先搜索
    def _divide_conquer(node):
        if node is not None:
            left = _divide_conquer(node.left)
            right = _divide_conquer(node.right)
            return [node.val] + left + right
        return []

    return _divide_conquer(root)
```

注意点：

> DFS 深度搜索（从上到下） 和分治法区别：前者一般将最终结果通过指针参数传入，后者一般递归返回结果最后合并

#### BFS 层次遍历

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

### 分治法应用

先分别处理局部，再合并结果

适用场景

- 快速排序
- 归并排序
- 二叉树相关问题

分治法模板

- 递归返回条件
- 分段处理
- 合并结果

```python
# 分治法模板
# (1)设置递归结束条件
# (2)分段递归处理
# (3)合并并返回结果
# def divide_and_conquer(self,input):
#     if input 满足递归结束条件:
#         return []
#     #divide
#     input1,input2 = input
#     res1 = self.divide_and_conquer(input1)
#     res2 = self.divide_and_conquer(input2)
#     #conquer
#     res = merge(res1,res2)
```

#### 典型示例

```python
#通过分治法遍历二叉树
```

#### 归并排序  

```python
# 归并排序 -稳定- 分治法
# divide:划分子序列,使每个子序列有序
# conquer:将两个有序子序列二路归并成一个有序表
def mergeSort(self, nums):
    if len(nums)<=1:
        return nums
    #divide
    mid_idx = len(nums)//2
    nums_left = self.mergeSort(nums[:mid_idx])
    nums_right = self.mergeSort(nums[mid_idx:])
    # conquer
    i,j = 0,0
    while i<len(nums_left) and j<len(nums_right):
        if nums_left[i]<nums_right[j]:
            nums[i+j] = nums_left[i]
            i+=1
        else:
            nums[i+j] = nums_right[j]
            j+=1
    #
    while i<len(nums_left):
        nums[i+j] = nums_left[i]
        i+=1
    while j<len(nums_right):
        nums[i+j] = nums_right[j]
        j+=1
    return nums
```

注意点

> 递归需要返回结果用于合并

#### 快速排序  


```python
# 快速排序 - 分治法
# (1)把数组中某个值设为基准值(2)把数组分为左右两段，左段小于基准值右段大于基准值，递归处理左右段，再合并
# 类似分治法没有合并过程
def quickSort(self,nums):
    # 快速排序:递归地把小/大于基准值的子数列排序
    if len(nums)<2:
        return
    #divide
    pivot = nums[0]
    left,right = 0,len(nums)-1
    while left<right:
        while left<right and nums[right]>pivot:
            right-=1
        nums[left]=nums[right]
        
        while left<right and nums[left]<=pivot:
            left+=1
        nums[right] = nums[left]
    nums[left] = pivot
    
    nums_left,nums_right = nums[:left],nums[left+1:]
    self.mergeSort(nums_left)
    self.mergeSort(nums_right)
    nums[:left],nums[left+1:] = nums_left,nums_right
```

注意点：

> 快排由于是原地交换所以没有合并过程
> 传入的索引是存在的索引（如：0、length-1 等），越界可能导致崩溃

常见题目示例

#### maximum-depth-of-binary-tree

[maximum-depth-of-binary-tree](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

> 给定一个二叉树，找出其最大深度。

思路：分治法

```python
# 二叉树的最大深度
def maxDepth(self, root: TreeNode) -> int:
    # 分治法
    # 终止条件
    if root is None:
        return 0
    # divide
    depth_left = self.maxDepth(root.left)
    depth_right = self.maxDepth(root.right)
    # conquer
    if depth_left < depth_right:
        return depth_right + 1
    else:
        return depth_left + 1
```

#### balanced-binary-tree

[balanced-binary-tree](https://leetcode-cn.com/problems/balanced-binary-tree/)

> 给定一个二叉树，判断它是否是高度平衡的二叉树。

思路：分治法，左边平衡 && 右边平衡 && 左右两边高度 <= 1，
因为需要返回是否平衡及高度，要么返回两个数据，要么合并两个数据，
所以用-1 表示不平衡，>0 表示树高度（二义性：一个变量有两种含义）。

```python
# 平衡二叉树
def isBalanced(self, root: TreeNode) -> bool:
    # 终止条件
    if root is None:
        return True
    # divide
    if self.isBalanced(root.left) and self.isBalanced(root.right):
        depth_left = self.maxDepth(root.left)
        depth_right = self.maxDepth(root.right)
        # conquer
        if abs(depth_left - depth_right) > 1:
            return False
        return True
    else:
        return False
```

注意

> 一般工程中，结果通过两个变量来返回，不建议用一个变量表示两种含义

#### binary-tree-maximum-path-sum

[binary-tree-maximum-path-sum](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

> 给定一个**非空**二叉树，返回其最大路径和。

思路：分治法，分为三种情况：左子树最大路径和最大，右子树最大路径和最大，左右子树最大加根节点最大，需要保存两个变量：一个保存子树最大路径和，一个保存左右加根节点和，然后比较这个两个变量选择最大值即可

```python
# 路径和的最大值
def maxPathSum(self, root: TreeNode) -> int:
    def _path_sum(node, pathSum):
        # 停止条件
        if node is None:
            return 0
        else:
            # divide
            left_path = max(_path_sum(node.left, pathSum), 0)  # 子树小于0则不考虑
            right_path = max(_path_sum(node.right, pathSum), 0)
            # conquer
            sigle_path = max(left_path, right_path) + node.val  # 单边最大值
            path = left_path + right_path + node.val  # 两遍加根结点
            pathSum += [max(path, sigle_path)]
            return sigle_path  # 必须已root结点结尾

    pathSum = []
    _path_sum(root, pathSum)
    return max(pathSum) if len(pathSum) else 0
```

#### lowest-common-ancestor-of-a-binary-tree

[lowest-common-ancestor-of-a-binary-tree](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

> 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

思路：分治法，有左子树的公共祖先或者有右子树的公共祖先，就返回子树的祖先，否则返回根节点

```python
# 最近公共祖先
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    # 我的思路是先找到两个结点所在的字数，再递归那颗字数，时间复杂度太高
    # 用分治法
    # 停止条件
    if root is None or root == p or root == q:
        return root
    # divide
    left_res = self.lowestCommonAncestor(root.left, p, q)
    right_res = self.lowestCommonAncestor(root.right, p, q)
    # conquer
    # 左右结果均不为空,表示两个结点分别位于左右两颗树上
    if left_res and right_res:
        return root
    elif left_res is not None:
        return left_res
    elif right_res is not None:
        return right_res
    # 都为空表示结点不在树上
    # elif left_res is None and right_res is None:
    #     return None
```

### BFS 层次应用

#### binary-tree-level-order-traversal

[binary-tree-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

> 给你一个二叉树，请你返回其按  **层序遍历**  得到的节点值。 （即逐层地，从左到右访问所有节点）

思路：用一个队列记录一层的元素，然后扫描这一层元素添加下一层元素到队列（一个数进去出来一次，所以复杂度 O(logN)）

```python
# 层序遍历
def levelOrder(self, root):
    if root is None:
        return []
    res = []
    queue = [root]
    while queue:
        cur_queue = []
        cur_res = []
        for node in queue:
            cur_res += [node.val]
            cur_queue += [node.left] if node.left else []
            cur_queue += [node.right] if node.right else []
        queue = cur_queue
        res += [cur_res]
    return res
```

#### binary-tree-level-order-traversal-ii

[binary-tree-level-order-traversal-ii](https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/)

> 给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）

思路：在层级遍历的基础上，翻转一下结果即可

```python
# 自底向上的层次遍历
def levelOrderBottom(self, root):
    return self.levelOrder(root)[::-1]
```

#### binary-tree-zigzag-level-order-traversal

[binary-tree-zigzag-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

> 给定一个二叉树，返回其节点值的锯齿形层次遍历。Z 字形遍历

```python
# 锯齿形层次遍历
def zigzagLevelOrder(self, root: TreeNode):
    res = self.levelOrder(root)
    return [res[idx] if idx % 2 == 0 else res[idx][::-1] for idx in range(len(res))]
```

### 二叉搜索树应用

#### validate-binary-search-tree

[validate-binary-search-tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)

> 给定一个二叉树，判断其是否是一个有效的二叉搜索树。

思路 1：中序遍历，检查结果列表是否已经有序

思路 2：分治法，判断左 MAX < 根 < 右 MIN

```python
# 判断其是否是一个有效的二叉搜索树
def isValidBST(self, root: TreeNode) -> bool:
    # 分治法或判断中序遍历结果是否有序
    # 停止条件
    if root is None:
        return True
    # divide and conquer
    # 判断左子树
    if root.left is not None:
        # 左子树是一颗二叉搜索树
        if not self.isValidBST(root.left): return False
        # 左子树的最大(右)结点小于根节点
        node = root.left
        while node.right:
            node = node.right
        if node.val >= root.val: return False
    # 判断右子树
    if root.right is not None:
        # 右子树是一颗二叉搜索树
        if not self.isValidBST(root.right): return False
        # 右子树的最小(左)结点大于根节点
        node = root.right
        while node.left:
            node = node.left
        if node.val <= root.val: return False
    return True
```


#### insert-into-a-binary-search-tree

[insert-into-a-binary-search-tree](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)

> 给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。

思路：找到最后一个叶子节点满足插入条件即可

```python
# 插入二叉搜索树
def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
    if root is None:
        return TreeNode(val)
    if val < root.val:
        # 插入左子树
        root.left = self.insertIntoBST(root.left, val)
    else:
        # 插入右子树
        root.right = self.insertIntoBST(root.right, val)
    return root
```

## 总结

- 掌握二叉树递归与非递归遍历
- 理解 DFS 前序遍历与分治法
- 理解 BFS 层次遍历

## 练习

- [ ] [maximum-depth-of-binary-tree](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)
- [ ] [balanced-binary-tree](https://leetcode-cn.com/problems/balanced-binary-tree/)
- [ ] [binary-tree-maximum-path-sum](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)
- [ ] [lowest-common-ancestor-of-a-binary-tree](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)
- [ ] [binary-tree-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)
- [ ] [binary-tree-level-order-traversal-ii](https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/)
- [ ] [binary-tree-zigzag-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)
- [ ] [validate-binary-search-tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)
- [ ] [insert-into-a-binary-search-tree](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)
