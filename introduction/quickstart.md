# 快速开始

## 数据结构与算法

数据结构是一种数据的表现形式，如链表、二叉树、栈、队列等都是内存中一段数据表现的形式。
算法是一种通用的解决问题的模板或者思路，大部分数据结构都有一套通用的算法模板，所以掌握这些通用的算法模板即可解决各种算法问题。

后面会分专题讲解各种数据结构、基本的算法模板、和一些高级算法模板，每一个专题都有一些经典练习题，完成所有练习的题后，你对数据结构和算法会有新的收获和体会。

先介绍两个算法题，试试感觉~

示例 1

[strStr](https://leetcode-cn.com/problems/implement-strstr/)

> 给定一个  haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从 0 开始)。如果不存在，则返回  -1。

思路：核心点遍历给定字符串字符，判断以当前字符开头字符串是否等于目标字符串

```python
def strStr(self, haystack: str, needle: str) -> int:
    # 1.使用python的find方法
    # return haystack.find(needle)
    # 2.子串逐一比较,O((m-n)*n)
    # m,n = len(haystack),len(needle)
    # for i in range(m-n+1):
    #     if haystack[i:i+n]==needle:
    #         return i
    # return -1
    # 3.利用双指针回溯,空子串返回0
    # m,n = len(haystack),len(needle)
    # if n==0:
    #     return 0
    # pm,pn = 0,0
    # while pm<m-n+1:
    #     while pm<m-n+1 and haystack[pm]!=needle[0]:
    #         pm+=1
    #     while pm+pn<m and pn<n and haystack[pm+pn]==needle[pn]:
    #         pn+=1
    #     if pn==n:
    #         return pm
    #     else:
    #         pm+=1
    #         pn=0
    # return -1
    # 4. Rabin Karp:哈希码对比+滚动哈希
    m, n = len(haystack), len(needle)
    if n == 0:
        return 0
    if n > m:
        return -1
    # 计算哈希码
    modulus = 2 ** 31  # 设置python整数的数值上限来避免溢出
    # hash_code = sum(字母对应的数字*26^(n-1))
    hash_code_m, hash_code_n = 0, 0
    for i in range(n):
        hash_code_m = (hash_code_m * 26 + (ord(haystack[i]) - ord('a'))) % modulus
        hash_code_n = (hash_code_n * 26 + (ord(needle[i]) - ord('a'))) % modulus
    if hash_code_m == hash_code_n:
        return 0
    # print(hash_code_m,hash_code_n)
    #
    t = pow(26, n - 1) % modulus
    for i in range(1, m - n + 1):
        # 滚动更新哈希值
        t1, t2 = ord(haystack[i - 1]) - ord('a'), ord(haystack[i + n - 1]) - ord('a')
        hash_code_m = ((hash_code_m - t1 * t) * 26 + t2) % modulus
        # print(hash_code_m)
        if hash_code_m == hash_code_n:
            return i
    return -1
```

需要注意点

- 循环时，i 不需要到 len-1
- 如果找到目标字符串，len(needle)==j

示例 2

[subsets](https://leetcode-cn.com/problems/subsets/)

> 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

思路：这是一个典型的应用回溯法的题目，简单来说就是穷尽所有可能性，算法模板如下

```go
result = []
func backtrack(选择列表,路径):
    if 满足结束条件:
        result.add(路径)
        return
    for 选择 in 选择列表:
        做选择
        backtrack(选择列表,路径)
        撤销选择
```

通过不停的选择，撤销选择，来穷尽所有可能性，最后将满足条件的结果返回

答案代码

```python
def subsets(self, nums):
    def backtrack(first=0, curr=[]):
        # if the combination is done
        if len(curr) == k:
            output.append(curr[:])

            print(curr)
        for i in range(first, n):
            # add nums[i] into the current combination
            curr.append(nums[i])
            # print(nums[first], nums[i], curr[:-1], '->', curr)
            # use next integers to complete the combination
            backtrack(i + 1, curr)
            # backtrack
            curr.pop()

    output = []
    n = len(nums)
    for k in range(n + 1):
        backtrack()
    return output

def subsets_2(self, nums):
    result = [[]]
    for idx in range(len(nums)):
        # 索引为i时,可生成长度1-i的子串长度
        new_res = []
        for re_ in result:
            new_res += [re_ + [nums[idx]]]
        result += new_res
    return result

def subsets_1(self, nums):
    subset = {0: [[]]}
    subset[len(nums)] = [nums]
    # 遍历可能的子集长度
    for i in range(1, len(nums)):
        subset[i] = []
        # 遍历长度小于1的子集,如不重复则增加元素
        for set_ in subset[i - 1]:
            for j in nums:
                if j not in set_ and sorted([j] + set_) not in subset[i]:
                    subset[i] += [sorted([j] + set_)]

    print(subset)
    #
    result = []
    for i in range(len(nums)):
        result += subset[i]
    #
    return result
```

说明：后面会深入讲解几个典型的回溯算法问题，如果当前不太了解可以暂时先跳过

## 面试注意点

我们大多数时候，刷算法题可能都是为了准备面试，所以面试的时候需要注意一些点

- 快速定位到题目的知识点，找到知识点的**通用模板**，可能需要根据题目**特殊情况做特殊处理**。
- 先去朝一个解决问题的方向！**先抛出可行解**，而不是最优解！先解决，再优化！
- 代码的风格要统一，熟悉各类语言的代码规范。
  - 命名尽量简洁明了，尽量不用数字命名如：i1、node1、a1、b2
- 常见错误总结
  - 访问下标时，不能访问越界
  - 空值 nil 问题 run time error

## 练习

- [ ] [strStr](https://leetcode-cn.com/problems/implement-strstr/)
- [ ] [subsets](https://leetcode-cn.com/problems/subsets/)
