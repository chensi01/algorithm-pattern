# 二叉搜索树

## 定义

- 每个节点中的值必须大于（或等于）存储在其左侧子树中的任何值。
- 每个节点中的值必须小于（或等于）存储在其右子树中的任何值。

## 应用

[validate-binary-search-tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)

> 验证二叉搜索树

```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        #停止条件
        if root is None:
            return True
        #divide and conquer
        #判断左子树
        if root.left is not None:
            #左子树是一颗二叉搜索树
            if not self.isValidBST(root.left):return False
            #左子树的最大(右)结点小于根节点
            node = root.left
            while node.right:
                node = node.right
            if node.val>=root.val:return False
        #判断右子树
        if root.right is not None:
            #右子树是一颗二叉搜索树
            if not self.isValidBST(root.right):return False
            #右子树的最小(左)结点大于根节点
            node = root.right
            while node.left:
                node = node.left
            if node.val<=root.val:return False
        return True
```

[insert-into-a-binary-search-tree](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)

> 给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 保证原始二叉搜索树中不存在新值。

```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if root is None:
            return TreeNode(val)
        if val<root.val:
            #插入左子树
            root.left = self.insertIntoBST(root.left,val)
        else:
            #插入右子树
            root.right = self.insertIntoBST(root.right,val)
        return root
```

[delete-node-in-a-bst](https://leetcode-cn.com/problems/delete-node-in-a-bst/)

> 给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的  key  对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if root is None:
            return None
        if root.val == key:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            else:
                node = root.right
                while node.left:
                    node = node.left
                node.left = root.left
                root = root.right
        elif key>root.val:
            root.right = self.deleteNode(root.right,key)
        elif key<root.val:
            root.left = self.deleteNode(root.left,key)
        return root
```

[balanced-binary-tree](https://leetcode-cn.com/problems/balanced-binary-tree/)

> 给定一个二叉树，判断它是否是高度平衡的二叉树。

```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        #分治法
        #终止条件
        if root is None:
            return 0
        #divide
        depth_left = self.maxDepth(root.left)
        depth_right = self.maxDepth(root.right)
        #conquer
        if depth_left<depth_right:
            return depth_right+1
        else:
            return depth_left+1
    def isBalanced(self, root: TreeNode) -> bool:
        #不平衡返回-1，平衡则返回max_depth
        #---------------------------------
        #终止条件
        if root is None:
            return True
        #divide
        if self.isBalanced(root.left) and self.isBalanced(root.right):
            depth_left = self.maxDepth(root.left)
            depth_right = self.maxDepth(root.right)
            #conquer
            if abs(depth_left-depth_right)>1:
                return False 
            return True
        else:
            return False
```

## 练习

- [ ] [validate-binary-search-tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)
- [ ] [insert-into-a-binary-search-tree](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)
- [ ] [delete-node-in-a-bst](https://leetcode-cn.com/problems/delete-node-in-a-bst/)
- [ ] [balanced-binary-tree](https://leetcode-cn.com/problems/balanced-binary-tree/)
