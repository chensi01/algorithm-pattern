# 链表

## 基本技能

链表相关的核心点

- null/nil 异常处理
- dummy node 哑巴节点
- 快慢指针
- 插入一个节点到排序链表
- 从一个链表中移除一个节点
- 翻转链表
- 合并两个链表
- 找到链表的中间节点

## 常见题型

### [remove-duplicates-from-sorted-list](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

> 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。


```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def deleteDuplicates(self, head: ListNode) -> ListNode:
    node = head
    while node and node.next:
        #node指向node的下一个不重复结点
        while node.next and (node.val == node.next.val):
            node.next = node.next.next
        node = node.next
    return head
```

### [remove-duplicates-from-sorted-list-ii](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

> 给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中   没有重复出现的数字。

思路：链表头结点可能被删除，所以用 dummy node 辅助删除

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def deleteDuplicates(self, head):
    if head is None:
        return head
    #定义哑节点dummy避免处理头节点为空的边界问题(dummy->next=head)
    dummy = ListNode(-1)
    dummy.next = head
    #
    d = dummy
    node = d.next
    while d and node :
        #如果n是重复结点
        if node.next and node.next.val == node.val:
            #找到下一个不与n重复的结点
            while node and node.val== d.next.val:
                node = node.next
            #d指向该结点，继续判断该节点是否重复
            d.next = node
        else:
            #不是重复结点,继续判断n.next
            d = d.next
            node = d.next
    return dummy.next
```

注意点
• A->B->C 删除 B，A.next = C
• 删除用一个 Dummy Node 节点辅助（允许头节点可变）
• 访问 X.next 、X.value 一定要保证 X != nil

### [reverse-linked-list](https://leetcode-cn.com/problems/reverse-linked-list/)

> 反转一个单链表。

思路：用一个 prev 节点保存向前指针，temp 保存向后的临时指针


```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def reverseList_iter(self, head: ListNode) -> ListNode:
    prev,cur= None,head
    while cur:
        nextTemp = cur.next#记录下一个结点
        #反转指针
        cur.next = prev
        prev,cur = cur,nextTemp
    return prev
 
def reverseList_recursive(self, head: ListNode) -> ListNode:
    if head is None or head.next is None:
        return head
    reversed_head = self.reverseList(head.next)
    #reversed_head的尾结点是head.next
    head.next.next = head
    head.next = None
    return reversed_head
```

### [reverse-linked-list-ii](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

> 反转从位置  *m*  到  *n*  的链表。请使用一趟扫描完成反转。

思路：先遍历到 m 处，翻转，再拼接后续，注意指针处理


```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
    #定义哑节点dummy避免处理翻转头节点的边界问题(dummy->next=head)
    dummy = ListNode(-1)
    dummy.next = head
    #遍历到m处
    prev = dummy
    for _ in range(m-1):
        prev = prev.next
    #翻转m->n
    pre,cur = prev.next,prev.next.next
    for _ in range(n-m):
        next_temp = cur.next
        cur.next = pre
        # print(cur.val,'->',pre.val)
        pre,cur = cur,next_temp
    #拼接后续
    prev.next.next = cur
    prev.next = pre
    #
    return dummy.next
```

### [merge-two-sorted-lists](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

> 将两个升序链表合并为一个新的升序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

思路：通过 dummy node 链表，连接各个元素


```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def mergeTwoLists_recursive(self, l1: ListNode, l2: ListNode) -> ListNode:
    # 考虑边界情况,返回非空链表
    if l1 is None:return l2
    if l2 is None:return l1
    # 递归地决定下一个添加到结果里的节点
    if l1.val<l2.val:
        head = l1
        l1 = l1.next
    else:
        head = l2
        l2 = l2.next
    head.next = self.mergeTwoLists(l1,l2)
    return head
    
def mergeTwoLists_iter(self, l1: ListNode, l2: ListNode) -> ListNode:
    # 原地调整链表元素的 next 指针完成合并-->空间代价O(1)
    # 处理合并之后链表的头部:为方便代码书写,设置虚拟的哑结点指向头结点(val属性不保存任何值).合并完后返回它的下一位置
    dummy = ListNode(-1)
    # 需要一个指针 tail 来记录下一个插入位置的前一个位置
    tail  = dummy
    # 需要两个指针p和q来记录未合并部分的第一位
    p,q = l1,l2
    #
    while p and q:
        # 都不为空的时候，取val较小的合并
        if p.val<q.val:
            # 合并时先调整tail的next属性,再将tail和p(q)后移动
            tail.next = p
            tail,p = tail.next,p.next
        else:
            tail.next = q
            tail,q = tail.next,q.next
    #一个链表已经合并完毕,则把另一个链表后面的元素全部合并
    if p:
        tail.next = p
    else:
        tail.next = q
    #
    return dummy.next
```





### [merge-k-sorted-lists](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

> 合并 k 个排序链表，返回合并后的排序链表。分析和描述算法的复杂度。

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def mergeTwoLists_iter(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(-1)
        tail = dummy
        while l1 and l2:
            if l1.val<l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        if l1:
            tail.next = l1
        else:
            tail.next = l2
        return dummy.next
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        # 空间代价O(1):原地调整链表元素的next指针
        # 简化版K=2->
        # --------------------------------
        # (1)逐一合并两条链表, 时间复杂度：O(NK^2)
        # res = None
        # for l in lists:
        #     res = self.mergeTwoLists_iter(res,l)
        # return res
        # --------------------------------
        # (2)使用分治法的思想,两两合并, 时间复杂度O(NKlogK)从K条链表开始两两合并成1条链表，因此每条链表都会被合并logK次,K条链表K*logK次.
        # def _mergeKLists(lists,l,r):
        #     if l==r:
        #         return lists[l]
        #     #divide
        #     mid = (l+r)//2
        #     left = _mergeKLists(lists,l,mid)
        #     right = _mergeKLists(lists,mid+1,r)
        #     #conquer
        #     return self.mergeTwoLists_iter(left,right)
        # if len(lists)==0:
        #     return []
        # return _mergeKLists(lists,0,len(lists)-1)
        # --------------------------------
        # (3)使用优先队列,每次比较k个元素
        # 时间复杂度：O(kn*logk) 堆插入和删除的时间代价为 O(logk),最多有kn个点
        # 空间复杂度：O(k)堆的元素不超过k个
        dummy = ListNode(-1)
        tail = dummy
        # 构建优先队列/最小堆
        import heapq
        prio_heap = []
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(prio_heap,(lists[i].val,i))
        while prio_heap:
            #取堆顶元素
            _,idx = heappop(prio_heap)
            # 修改指针
            tail.next = lists[idx]
            tail = tail.next
            if lists[idx].next :
                lists[idx] = lists[idx].next
                heapq.heappush(prio_heap,(lists[idx].val,idx))
        return dummy.next
```








### [partition-list](https://leetcode-cn.com/problems/partition-list/)

> 给定一个链表和一个特定值 x，对链表进行分隔，使得所有小于  *x*  的节点都在大于或等于  *x*  的节点之前。

思路：将大于 x 的节点，放到另外一个链表，最后连接这两个链表


```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def partition(self, head: ListNode, x: int) -> ListNode:
    head_1 = ListNode(-1)
    head_2 = ListNode(-1)
    #
    p1,p2 = head_1,head_2
    while head:
        # 小于 x 的节点
        if head.val<x:
            p1.next = head
            p1 = p1.next
        # 大于或等于 x 的节点
        else:
            p2.next = head
            p2 = p2.next
        head = head.next
    # 拼接两个列表
    p2.next = None
    p1.next = head_2.next
    return head_1.next
```

哑巴节点使用场景

> 当头节点不确定的时候，使用哑巴节点

### [sort-list](https://leetcode-cn.com/problems/sort-list/)

> 在  *O*(*n* log *n*) 时间复杂度和常数级空间复杂度下，对链表进行排序。

思路：归并排序，找中点和合并操作


```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
# O(nlogn)时间复杂度和常数级空间复杂度
def sortList(self, head: ListNode) -> ListNode:
    if head is None or head.next is None:
        return head
    # 计算链表长度
    p,length = head,0
    while p : p,length = p.next,length+1
    #设置哑结点
    dummy = ListNode(-1)
    dummy.next = head
    #设置切片规模
    intv = 1
    #不断double切片规模并两两归并
    while intv<length:
        prev,current = dummy,dummy.next
        #从当前结点开始找到两个子链表进行归并
        while current:
            #找到要归并的左子链表
            left,residue_length_left = current,intv
            while current and residue_length_left:
                current,residue_length_left = current.next,residue_length_left-1
            # 左子链表长度不足切片规模,表明不存在右子链表,不需要归并.
            if residue_length_left:
                break
            #找到要归并的右子链表
            right,residue_length_right = current,intv
            while current and residue_length_right:
                current,residue_length_right = current.next,residue_length_right-1
            #归并左右有序子链表
            # print(left.val if left else 0,right.val  if right else 0)
            length_left,length_right = intv-residue_length_left,intv-residue_length_right
            while length_left and length_right:
                if left.val<right.val:
                    prev.next = left
                    left = left.next
                    length_left-=1
                else:
                    prev.next = right
                    right = right.next
                    length_right-=1
                prev = prev.next

            prev.next = left if length_left else right
            while length_left>0 or length_right>0:
                prev = prev.next
                length_left,length_right=length_left-1,length_right-1
            prev.next = current
        intv*=2
    return dummy.next



# O(nlogn)时间复杂度和O(logn)空间复杂度
def sortList_recursive(self, head: ListNode) -> ListNode:
    #二分+归并
    #停止条件
    if head is None or head.next is None:
        return head
    # ---------------
    #divide
    # 找中间结点:使用快慢双指针法，奇数个节点找到中点，偶数个节点找到中心左边的节点
    slow,fast = head,head.next
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
    mid,slow.next = slow.next,None
    left,right = self.sortList(head),self.sortList(mid)
    #conquer
    dummy = ListNode(-1)
    n = dummy
    while left and right:
        if left.val<right.val:
            n.next = left
            left = left.next
        else:
            n.next = right
            right = right.next
        n = n.next
    if left:
        n.next=left
    if right:
        n.next = right
    return dummy.next
```

注意点

- 快慢指针 判断 fast 及 fast.Next 是否为 nil 值
- 递归 mergeSort 需要断开中间节点
- 递归返回条件为 head 为 nil 或者 head.Next 为 nil

### [reorder-list](https://leetcode-cn.com/problems/reorder-list/)

> 给定一个单链表  *L*：*L*→*L*→…→*L\_\_n*→*L*
> 将其重新排列后变为： *L*→*L\_\_n*→*L*→*L\_\_n*→*L*→*L\_\_n*→…

思路：找到中点断开，翻转后面部分，然后合并前后两个链表


```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def reorderList(self, head: ListNode) -> None:
    """
    Do not return anything, modify head in-place instead.
    """
    if head is None or head.next is None:
        return head
    #设置快慢指针找到中间结点
    slow,fast = head,head.next
    while fast and fast.next:
        slow,fast = slow.next,fast.next.next
    right = slow.next
    slow.next = None
    #反转后半部分链表
    if right is not None and right.next is not None:
        pre,n = right,right.next
        while n :
            next_tmp = n.next
            n.next = pre
            pre,n = n,next_tmp
        right.next = None
        right = pre

    #交叉合并左右子链表
    left = head
    while right:
        left.next,right.next,left,right= right,left.next,left.next,right.next
```

### [linked-list-cycle](https://leetcode-cn.com/problems/linked-list-cycle/)

> 给定一个链表，判断链表中是否有环。

思路：快慢指针，快慢指针相同则有环，证明：如果有环每走一步快慢指针距离会减 1
![fast_slow_linked_list](https://img.fuiboom.com/img/fast_slow_linked_list.png)


```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def hasCycle(self, head: ListNode) -> bool:
    #方法1:快慢指针，若有环快慢指针会相遇
    if head is None or head.next is None:
        return False
    slow,fast = head,head.next
    while fast and fast.next:
        if slow==fast:
            return True
        slow,fast = slow.next,fast.next.next
    return False
    #----------------
    #方法2:哈希表判断结点是否已访问
    visited = []
    p = head
    while p:
        if p in visited:
            return True
        else:
            visited.append(p)
            p = p.next
    return False
```

### [linked-list-cycle-ii](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

> 给定一个链表，返回链表开始入环的第一个节点。  如果链表无环，则返回  `null`。

思路：快慢指针，快慢相遇之后，慢指针回到头，快慢指针步调一致一起移动，相遇点即为入环点
![cycled_linked_list](https://img.fuiboom.com/img/cycled_linked_list.png)


```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def detectCycle(self, head: ListNode) -> ListNode:
    # #方法1:哈希法
    # if head is None or head.next is None:
    #     return None
    # p,visited = head,[]
    # while p and (p not in visited):
    #     visited+=[p]
    #     p = p.next
    # return p
    #方法2:快慢指针法
    if head is None or head.next is None:
        return None
    #找到相遇结点
    slow,fast = head,head.next
    while fast and fast.next and slow!=fast:
        slow,fast = slow.next,fast.next.next
    #fast 或 fast.next 为None
    if slow!=fast:
        return None
    #根据相遇结点找到环的入口
    #头结点到环入口:a,环入口到相遇点:b,相遇点到环入口:c
    #slow走到相遇点:a+b,fast走到相遇点碰到slow:(a-1)+b+c+b
    #fast的步数是slow的两倍；2(a+b) = (a-1)+b+c+b ==> a=c-1
    p,q = head,slow.next
    while p!=q:
        p,q = p.next,q.next
    return p
```

坑点

- 指针比较时直接比较对象，不要用值比较，链表中有可能存在重复值情况
- 第一次相交后，快指针需要从下一个节点开始和头指针一起匀速移动

另外一种方式是 fast=head,slow=head

这两种方式不同点在于，**一般用 fast=head.Next 较多**，因为这样可以知道中点的上一个节点，可以用来删除等操作。

- fast 如果初始化为 head.Next 则中点在 slow.Next
- fast 初始化为 head,则中点在 slow

### [palindrome-linked-list](https://leetcode-cn.com/problems/palindrome-linked-list/)

> 请判断一个链表是否为回文链表。


```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def isPalindrome(self, head: ListNode) -> bool:
    if head is None or head.next is None:
        return True
    #使用快慢指针找到中间结点
    slow,fast = head,head.next
    while fast and fast.next:
        slow,fast = slow.next,fast.next.next
    right = slow.next
    #断开分为左右子链表
    slow.next = None
    #将右子链表反转
    if not right or right.next:
        p,q = right,right.next
        while q:
            next_q = q.next
            q.next = p
            p,q = q,next_q
        #原头结点的next置为空
        right.next=None
        right = p
    #对比左右子链表是否相同
    left = head
    while right and left:
        print(left.val,right.val)
        if right.val != left.val:
            return False
        left,right = left.next,right.next
    return True   
```

### [copy-list-with-random-pointer](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)

> 给定一个链表，每个节点包含一个额外增加的随机指针，该指针可以指向链表中的任何节点或空节点。
> 要求返回这个链表的 深拷贝。

思路：1、hash 表存储指针，2、复制节点跟在原节点后面


```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def copyRandomList(self, head: 'Node') -> 'Node':
    if head is None:
        return None
    p = head
    #遍历链表,并复制结点到每个结点的next
    while p:
        next_p = p.next
        copy_p = Node(p.val)
        p.next,copy_p.next = copy_p,next_p
        p = next_p
    #遍历链表,将复制节点的random指针指向原结点random指针的next
    p = head
    while p:
        p.next.random = None if p.random is None else p.random.next
        p = p.next.next
    #遍历链表,修正next指针
    copy_head = head.next
    p,q = head,copy_head
    while p and q:
        p.next = q.next
        q.next = None if q.next is None else q.next.next
        p,q = p.next,q.next
    return copy_head

```

## 总结

链表必须要掌握的一些点，通过下面练习题，基本大部分的链表类的题目都是手到擒来~

- null/nil 异常处理
- dummy node 哑巴节点
- 快慢指针
- 插入一个节点到排序链表
- 从一个链表中移除一个节点
- 翻转链表
- 合并两个链表
- 找到链表的中间节点

## 练习

- [ ] [remove-duplicates-from-sorted-list](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)
- [ ] [remove-duplicates-from-sorted-list-ii](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)
- [ ] [reverse-linked-list](https://leetcode-cn.com/problems/reverse-linked-list/)
- [ ] [reverse-linked-list-ii](https://leetcode-cn.com/problems/reverse-linked-list-ii/)
- [ ] [merge-two-sorted-lists](https://leetcode-cn.com/problems/merge-two-sorted-lists/)
- [ ] [partition-list](https://leetcode-cn.com/problems/partition-list/)
- [ ] [sort-list](https://leetcode-cn.com/problems/sort-list/)
- [ ] [reorder-list](https://leetcode-cn.com/problems/reorder-list/)
- [ ] [linked-list-cycle](https://leetcode-cn.com/problems/linked-list-cycle/)
- [ ] [linked-list-cycle-ii](https://leetcode-cn.com/problems/https://leetcode-cn.com/problems/linked-list-cycle-ii/)
- [ ] [palindrome-linked-list](https://leetcode-cn.com/problems/palindrome-linked-list/)
- [ ] [copy-list-with-random-pointer](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)
