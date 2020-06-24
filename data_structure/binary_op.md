# 二进制

## 常见二进制操作

### 基本操作

a=0^a=a^0

0=a^a

由上面两个推导出：a=a^b^b

### 交换两个数

a=a^b

b=a^b

a=a^b

### 移除最后一个 1

a=n&(n-1)

### 获取最后一个 1

diff=(n&(n-1))^n

## 常见题目

[single-number](https://leetcode-cn.com/problems/single-number/)

> 给定一个**非空**整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

```python
def singleNumber(self, nums) -> int:
    #a^a=0,0^a=a
    res = 0
    for n in nums:
        res^=n
    return res
```

[single-number-ii](https://leetcode-cn.com/problems/single-number-ii/)

> 给定一个**非空**整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。

```python
def singleNumber(self, nums) -> int:
    # 利用异或:由于0^a=a,0^a^a=0,0^a^a^a=a,异或操作可用于检测出现奇数次的数
    # 区分出现1次和3次的元素
    seen_once,seen_twice = 0,0
    # first appearance : add num to seen_once 
    # second appearance : remove num from seen_once,add num to seen_twice
    # third appearance : remove num from seen_twice
    for num in nums:
        seen_once = seen_once^num & (~seen_twice)
        seen_twice = seen_twice^num & (~seen_once)
    return seen_once

    # 时/空间复杂度O(N)
    # return (3*sum(set(nums))-sum(nums))//2
```

[single-number-iii](https://leetcode-cn.com/problems/single-number-iii/)

> 给定一个整数数组  `nums`，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。

```python
def singleNumber(self, nums):
    bitmask = 0
    for num in nums:
        bitmask ^= num
    # 仅保留最右的diff 1,其他bit均为0
    # 假设该bit来自x
    diff = bitmask&(-bitmask)
    x = 0
    for num in nums:
        #通过判断当前num的diff位是否为1排除掉y，是的连续异或的结果为x
        if num&diff:
            x ^= num
    #y=x^y^x
    y = bitmask^x
    return [x,y]
```

[number-of-1-bits](https://leetcode-cn.com/problems/number-of-1-bits/)

> 编写一个函数，输入是一个无符号整数，返回其二进制表达式中数字位数为 ‘1’  的个数（也被称为[汉明重量](https://baike.baidu.com/item/%E6%B1%89%E6%98%8E%E9%87%8D%E9%87%8F)）。

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        #方法1：遍历数字的 32 位。如果某一位是 11 ，将计数器加一
        res = 0
        mask = 1
        for _ in range(32):
            if n&mask!=0:
                res+=1
            mask = mask<<1
        return res
    def hammingWeight_1(self, n: int) -> int:
        #方法2：不断把数字最后一个 1 反转，并把答案加一
        res = 0
        while n:
            res+=1
            #移除最后一个1
            n = n&(n-1)
        return res
```

[counting-bits](https://leetcode-cn.com/problems/counting-bits/)

> 给定一个非负整数  **num**。对于  0 ≤ i ≤ num  范围中的每个数字  i ，计算其二进制数中的 1 的数目并将它们作为数组返回。

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            res+=1
            n = n&(n-1)
        return res
    def countBits(self, num: int) -> List[int]:
        # return [self.hammingWeight(i) for i in range(num+1)]
        #动态规划法
        res = [0]
        for n in range(1,num+1):
            res.append(res[n&(n-1)]+1)
        return res
```

[reverse-bits](https://leetcode-cn.com/problems/reverse-bits/)

> 颠倒给定的 32 位无符号整数的二进制位。

思路：依次颠倒即可

```python
def reverseBits(self, n: int) -> int:
    # 反转索引i处的位,并置于31-i
    res = 0
    pow=31
    while n:
        res+=((n&1)<<pow)
        n = n>>1
        pow-=1
    return res
```

[bitwise-and-of-numbers-range](https://leetcode-cn.com/problems/bitwise-and-of-numbers-range/)

> 给定范围 [m, n]，其中 0 <= m <= n <= 2147483647，返回此范围内所有数字的按位与（包含 m, n 两端点）。

```python
def rangeBitwiseAnd(self, m: int, n: int) -> int:
    # 在某个位上,只要有一个那么结果运算为零
    # 故答案为m,n的公共前缀
    #--------------------------------------
    #方法1:m,n不断右移直到数字被缩减为公共前缀
    shift = 0
    while m!=n:
        m=m>>1
        n=n>>1
        shift+=1
    return m<<shift
    #方法2:不断清除二进制串中最右边的1 直到被缩减为公共前缀
    while m<n:
        n=n&(n-1)
    return n
```

## 练习

- [ ] [single-number](https://leetcode-cn.com/problems/single-number/)
- [ ] [single-number-ii](https://leetcode-cn.com/problems/single-number-ii/)
- [ ] [single-number-iii](https://leetcode-cn.com/problems/single-number-iii/)
- [ ] [number-of-1-bits](https://leetcode-cn.com/problems/number-of-1-bits/)
- [ ] [counting-bits](https://leetcode-cn.com/problems/counting-bits/)
- [ ] [reverse-bits](https://leetcode-cn.com/problems/reverse-bits/)
