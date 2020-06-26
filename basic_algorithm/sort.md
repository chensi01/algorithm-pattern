# 排序

## 常考排序

### 快速排序


### 归并排序

### 堆排序

用数组表示的完美二叉树 complete binary tree

> 完美二叉树 VS 其他二叉树

![image.png](https://img.fuiboom.com/img/tree_type.png)

[动画展示](https://www.bilibili.com/video/av18980178/)

![image.png](https://img.fuiboom.com/img/heap.png)



## 参考

[十大经典排序](https://www.cnblogs.com/onepixel/p/7674659.html)

```python
class Solution:
    def bubbleSort(self,nums):
        #冒泡排序:不断将最大的元素沉到低端
        for i in range(len(nums)-1):
            for j in range(len(nums)-1-i):
                if nums[j]>nums[j+1]:
                    nums[j],nums[j+1] = nums[j+1],nums[j]
    def selectionSort(self,nums):
        #选择排序:n-1次遍历,每次找到当前最小值
        for i in range(len(nums)-1):
            min_idx = i
            for j in range(i+1,len(nums)):
                min_idx = j if nums[j]<nums[min_idx] else min_idx
            nums[min_idx],nums[i] = nums[i],nums[min_idx]
    def insertionSort(self,nums):
        # 插入排序:逐步将i插入(0,i-1)的有序列表
        if len(nums)<2:
            return
        for i in range(1,len(nums)):
            pre_idx = i-1
            tmp = nums[i]
            while pre_idx>=0 and nums[pre_idx]>tmp:
                #向后移动
                nums[pre_idx+1] = nums[pre_idx]
                pre_idx-=1
            nums[pre_idx+1] = tmp
    def shellSort(self,nums):
        # 希尔排序:缩小增量的插入排序
        if len(nums)<2:
            return
        gap = len(nums)//2
        while gap>0:
            for i in range(gap,len(nums)):
                pre_idx = i-gap
                tmp = nums[i]
                while pre_idx>=0 and nums[pre_idx]>tmp:
                    #向后移动
                    nums[pre_idx+gap] = nums[pre_idx]
                    pre_idx-=gap
                nums[pre_idx+gap] = tmp
            gap//=2
    def mergeSort(self,nums):
        # 归并排序:分治法,先使子序列有序，再合并子序列
        if len(nums)<2:
            return
        #divide
        mid = len(nums)//2
        nums_left,nums_right = nums[:mid],nums[mid:]
        self.mergeSort(nums_left)
        self.mergeSort(nums_right)
        #conquer
        i,j = 0,0
        while i<len(nums_left) and j<len(nums_right):
            if nums_left[i]<nums_right[j]:
                nums[i+j] = nums_left[i]
                i+=1
            else:
                nums[i+j] = nums_right[j]
                j+=1
        if i<len(nums_left):
            nums[i+j:] = nums_left[i:]
        if j<len(nums_right):
            nums[i+j:] = nums_right[j:]

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

    def heapSort(self,nums):
        # 堆排序:从无序堆构造最大/小堆，并取出堆顶元素
        # 堆:堆近似完全二叉树,同时最大/小堆子结点大/小于父节点。
        if len(nums)<2:
            return
        def _heapify(nums,n,i):
            largest_root = i
            left,right = 2*i+1,2*i+2
            if left<n and nums[largest_root]<nums[left]:
                largest_root = left
            if right<n and nums[largest_root]<nums[right]:
                largest_root = right
            if largest_root!=i:
                nums[largest_root],nums[i] = nums[i],nums[largest_root]#交换元素
                _heapify(nums,n,largest_root)#递归的构造被交换的元素所处的子堆
        #自下而上构造初始推
        for i in range(len(nums)-1,-1,-1):
            _heapify(nums,len(nums),i)
        #
        for i in range(len(nums)-1,-1,-1):
            #交换堆顶与最右
            nums[0],nums[i] = nums[i],nums[0]
            #自堆顶起重构堆
            _heapify(nums,i,0)

    def countingSort(self,nums):
        # 计数排序:输入的数据是有确定范围的整数,记录该范围内每个数出现的次数。
        # 空间换时间
        if len(nums)<2:
            return
        min_nums,max_nums = min(nums),max(nums)
        num_counts = [0]*(max_nums-min_nums+1)
        for n in nums:
            num_counts[n-min_nums]+=1
        #计算累加和
        for i in range(1,len(nums)):
            num_counts[i]+=num_counts[i-1]
        for i in range(len(num_counts)):
            start = 0 if i==0 else num_counts[i-1]
            end = num_counts[i]
            nums[start:end] = [i+min_nums]*(end-start)

    def bucketSort(self,nums):
        # 桶排序:将元素分到有限数量的桶里,每个桶再分别排序。
        if len(nums)<2:
            return
        #创建空桶
        default_bucket_volume = 5
        bucket_num = (len(nums)//default_bucket_volume)+1
        bucket_list = [[] for _ in range(bucket_num)]
        #将元素分配到桶
        min_nums,max_nums = min(nums),max(nums)
        for n in nums:
            bucket_idx = (n-min_nums)//default_bucket_volume
            bucket_list[bucket_idx].append(n)
        #桶内排序i
        for i in range(bucket_num):
            self.quickSort(bucket_list[i])
        # 产生新的排序后的列表
        idx = 0
        for i in range(bucket_num):
            for j in range(len(bucket_list[i])):
                nums[idx] = bucket_list[i][j]
                idx+=1

    def radixSort(self,nums):
        # 基数排序:按照低位->最高位,排序再收集
        if len(nums)<2:
            return
        #
        max_radix = len(str(max(nums)))
        for i in range(max_radix):
            #初始化桶
            bucket_list = [[] for _ in range(10)]
            #将元素加入到相应位基数的桶
            for n in nums:
                #得到基数
                bucket_idx = int((n // (10**i)) % 10)
                bucket_list[bucket_idx].append(n)
            #将桶中元素放回
            idx = 0
            for bucket_idx in range(10):   # 放回原序列
                for j in range(len(bucket_list[bucket_idx])):
                    nums[idx] = bucket_list[bucket_idx][j]
                    idx+=1
```


[二叉堆](https://labuladong.gitbook.io/algo/shu-ju-jie-gou-xi-lie/er-cha-dui-xiang-jie-shi-xian-you-xian-ji-dui-lie)

## 练习

- [ ] 手写快排、归并、堆排序
