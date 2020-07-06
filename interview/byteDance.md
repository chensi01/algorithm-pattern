# leetcode的字节跳动卡片


## 1.字符串

(1)[minimum-path-sum](https://leetcode-cn.com/problems/minimum-path-sum/)

题目概述:给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度

考点:滑动窗口算法+哈希集合的数据结构

>答题模板:
>(1)暴力解法概述：【方法】依次递增地枚举子串的起始位置,找到从该字符起的不重复的最长子串。那么其中最长的字符串就是答案。
>【复杂度】N是字符串的长度，需要遍历字符串作为起始，每次要从该起始位置向后遍历，所以时间复杂度是O(N^2)。空间复杂度为O(1).

>(2)改进思路概述：不重复子串的结束位置是单调递增：假设i,j为不重复子串的左右索引，那么他的子串显然是不重复的。
>当枚举到i+1作为起始位置时，暴力解法从i+2开始判断重复字符，导致重复比较。可以继续增大j，知道右侧出现重复字符。
>这就是【滑动窗口】思想。通过避免重复判断，减小判断次数，从而优化时间复杂度。

>(3)改进方法概述:
>【算法】滑动窗口：使用两个指针指示子串的左右边界（起始和结束位）。每一步，将左指针右移，枚举该字符起始位的子串，不断右移右指针并保证无重复。枚举结束后的最长长度即为答案。
>【数据结构】哈希集合：使用该数据结构判断是否有重复字符。右移指针时添加元素，左移指针时移除元素。
>【复杂度】左指针和右指针分别会遍历整个字符串,时间复杂度为O(N);哈希集合需要占用大小为字符集大小的空间，假设为ascii在0-128的字符,sigma=128，则为O(|sigma|)

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        #两个指针指示子串的左右边界
        left,right = 0,0
        #使用哈希集合判断是否包含重复字符
        hash_set = set()
        #
        res = 0
        for left in range(len(s)):
            # 左指针向右移动一格，移除一个字符
            if left!=0:
                # 注意,先右移再移除，所以移除的是当前left指针的的前一位
                hash_set.remove(s[left-1])
            #不断右移右指针并保证无重复
            while right<len(s) and s[right] not in hash_set:
                hash_set.add(s[right])
                #枚举过程中的最长长度即为答案
                res = max(res,right-left+1)
                print(s[left:right+1],hash_set)
                right+=1
        return res   
```

(2)[longest-common-prefix](https://leetcode-cn.com/problems/longest-common-prefix/)

题目概述:查找字符串数组的最长公共前缀

考点:横/纵向扫描/分治法/数值的二分查找


>答题模板:
>一个更简单的问题是：查找两个字符串的最长公共前缀。依次向右遍历扫描两个字符串，直到某位置上两字符不同。当前列之前的部分为最长公共前缀。
>【复杂度】使用的额外空间为常数所以空间复杂度O(1)；最差需要遍历的字符串长度为N，每次需要比较2个字符，时间复杂度为O(2N)=O(N).
>本题等价于查找M个字符串的最长公共前缀。
>（1）一种最朴素的方法是：依次向右遍历字符串的每一列,每次纵向扫描M个字符，直到某列上M字符不完全同。当前列之前的部分为最长公共前缀。使用常数个额外空间,空间复杂度O(1)。最差需要遍历的字符串长度为N，每次需要比较M个字符，时间复杂度为O(MN).
>（2）一种查找的简单方法是：依次遍历数组中的每个字符串，根据当前最长公共前缀和正在遍历的字符串，更新最长公共前缀。【复杂度】使用的额外空间为常数所以空间复杂度O(1)；最差情况下需要比较M个字符串的N个字符，时间复杂度为O(MN).
>（3）可以使用分治法，把问题分解为两个子问题：求数组中前半部分和后半部分的公共前缀。然后计算两个子问题的解的最长公共前缀。【复杂度】空间复杂度取决于递归调用的层数，层数最大为logM，每层需要大小为N的空间存储返回结果，所以未O(NlogM)。时间复杂度的递推式T(M) = 2*T(M/2)+O(N),得T(M)=O(MN)
>（4）在lcp的长度上做二分查找。初始化lcp的长度边界为0和数组中最短字符串长度L。每次取查找范围的中间值mid，判断长度为mid的前缀是否相同，并且根据结果将查找范围缩小一半：不同则缩小边界为左半部分0->mid，否则缩小到右半部分到mid->L。直到左右边界相同。

```python
class Solution:
    # 方法一：纵向扫描
    def longestCommonPrefix_1(self, strs):
        if len(strs)==0:
            return ""
        length,str_num = len(strs[0]),len(strs)
        # 依次向右遍历字符串的每一列
        for i in range(length):
            c = strs[0][i]
            # 每次纵向扫描M个字符，直到某列上M字符不完全同。
            if any([i>=len(s) or s[i]!=c for s in strs[1:]]):
                # 当前列之前的部分为最长公共前缀
                return strs[0][:i]
        return strs[0]


    # 方法二：横向扫描
    def longestCommonPrefix_2(self, strs):
        if len(strs)==0:
            return ""
        def lcp(s1,s2):
            length = min(len(s1),len(s2))
            #依次向右遍历两个字符串
            for i in range(length):
                #直到某列上两字符不同
                if s1[i]!=s2[i]:
                    #解为当前列之前的部分
                    return s1[:i]
            return s1[:length]
        #遍历数组中的每个字符串
        res = strs[0]
        for i in range(1,len(strs)):
            # 根据当前lcp和字符串更新lcp
            res = lcp(res,strs[i])
        return res


    # 方法三：分治
    def longestCommonPrefix_3(self, strs):
        def lcp(s1,s2):
            length = min(len(s1),len(s2))
            #依次向右遍历两个字符串
            for i in range(length):
                #直到某列上两字符不同
                if s1[i]!=s2[i]:
                    #解为当前列之前的部分
                    return s1[:i]
            return s1[:length]
        
        if len(strs)==0:
            return ""
        if len(strs)==1:
            return strs[0]
        #把数组从中间分为两个较小的子数组
        mid = len(strs)//2
        # 求解两个子数组的最长公共前缀
        res_1 = self.longestCommonPrefix_3(strs[:mid])
        res_2 = self.longestCommonPrefix_3(strs[mid:])
        #求解两个子问题的解的最长公共前缀
        res = lcp(res_1,res_2)
        return res



    # 方法四:在lcp的长度上做二分查找
    def longestCommonPrefix(self, strs):
        if len(strs)==0:
            return ""
        if len(strs)==1:
            return strs[0]
        # 判断长度为length的前缀是否相同
        def isCommon(length):
            s_tmp = strs[0][:length]
            return all([s[:length]==s_tmp for s in strs[1:]])

        #初始化lcp长度边界
        left,right = 0,min(len(s) for s in strs)
        while left<right:
            #每次取查找范围的中间值
            # mid = (left+right)//2
            mid = (right-left+1)//2+left
            # 判断长度为mid的前缀是否相同，并根据结果将查找范围缩小一半
            if isCommon(mid):
                left = mid
            else:
                right = mid-1
        return strs[0][:left]
        


```

(3)[permutation-in-string](https://leetcode-cn.com/problems/permutation-in-string/)

题目概述:判断 s2 是否包含 s1 的排列,第一个字符串的排列之一是第二个字符串的子串。

考点:滑动窗口算法+哈希字典或数组的数据结构

注意:每次窗口滑动，只改变边界头尾，所以尝试在常数时间内完成窗口内数据的更新以优化时间复杂度

>答题模板:
>【做法】最简单的方法是生成s1的所有排列，并检查它是否是s2的子串。
为了生成所有可能的排列,首先将字符串中的第一个元素与每个后面的其他元素交换，再递归的将第二个元素与每个后面的其他元素交换，每次置换到最后一个元素都能得到一个新的排列。【复杂度分析】这个过程的时间复杂度为O(n!)-读作n的全排列。【做法】比较一个字符串是否为另一字符串的子串可以通过依次遍历较长字符串的每个索引作为起始位置，每次向后比较子串长度次，查看是否相同。【复杂度分析】最大时间复杂度是O(MN),如果将子串哈希，那么时间复杂度可缩短至O(N)。综合时间复杂度为O(N!)【复杂度分析】空间复杂度依赖于递归的深度N和每次递归保存的排列长度N，所以空间复杂度为O(N^2)
【分析复杂的根本原因】生成字符串的所有排列是该方法复杂的根本原因。

> 要优化时间复杂度，可以不生成所有排列，转化为比较s2中每个长度与s1相同的窗口内的子串是否是s1的一个排列。如果他们每个字母的频率完全匹配，那么一个字符串就是另一个字符串的排列。

> 首先需要一个数据结构来存储两个字符串中每个字母的频率，可以使用哈希映射表，或更简单的，长度为26的数组。
在比较s2的子串时，可以依次向右遍历s2的元素作为始位置，计算频率并进行比较。
【优化点】由于长度固定,所以每次更新滑动窗口两个边界位置的元素的频率。【复杂度】需要在s1上计算频率O(l1),然后在(l2-l1)个起始位置上计算频率并做比较，计算频率O(1)，每次比较26个元素的频率O(26*(l2-l1))，所以时间复杂度为O(l1+26*(l2-l1))。
【优化点】上个解法的问题在于每次滑动窗口只有两个元素的频率改变但需要比较26个字符。要针对这点进行优化，可以维护一个 count变量，记录26个字符中，有多少字符的频率匹配。每次滑动窗口时更新count的值。值为26表示所有字符的频率完全匹配。【复杂度】时间复杂度为O(l1+(l2-l1))
【复杂度】使用常数个额外空间所以空间复杂度为O(1)

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        #维护两个长度为26的数组,分别记录每个字符的频率
        map1,map2 = [0]*26,[0]*26
        for i in range(len(s1)):
            c1 = ord(s1[i])-ord('a')
            map1[c1]+=1
            if i>=len(s2):
                return False
            c2 = ord(s2[i])-ord('a')
            map2[c2]+=1
        # 维护一个 count变量,记录26个字符中，有多少字符的频率匹配。
        count = 0
        for i in range(26):
            if map1[i]==map2[i]:
                count+=1
        # 比较s2中每个长度与s1相同的窗口内的子串是否是s1的一个排列
        left_i,right_i = 0,len(s1)-1
        while right_i<len(s2)-1:
            # print(map1,map2,count,right_i)
            # 值为26表示所有字符的频率完全匹配
            if count==26:
                return True
            # 滑动窗口,数据更新
            # 右移右侧边界
            right_i+=1
            c2_right = ord(s2[right_i])-ord('a')
            map2[c2_right]+=1
            if map2[c2_right]==map1[c2_right]:
                count+=1
            elif map2[c2_right]-1==map1[c2_right]:
                count-=1
            # 收缩左侧边界
            c2_left = ord(s2[left_i])-ord('a')
            map2[c2_left]-=1
            if map2[c2_left]==map1[c2_left]-1:
                count-=1
            elif map2[c2_left]==map1[c2_left]:
                count+=1
            left_i+=1
        return count==26
```



(4)[add-strings](https://leetcode-cn.com/problems/add-strings/)

题目概述:字符串相加

考点:进位/边界处理

注意:进位/边界处理

>答题模板:【算法】要解决字符串相加，可以模拟人工加法的竖式相加思路。从后向前把对应位置的数字相加。需要特别处理的是1.根据逢十进一的原则处理进位2.索引yi'chu3.最高位进位处理。
【数据结构】首先设置两个变量作为指针，指向字符串中正在被处理的位置，初始化为字符串尾部。再设置一个变量作为进位标记，初始化为0。
【算法流程】计算两个字符串对应位置的和，并且加上上一次的进位。这个值除以10的结果作为新的进位，余数作为当前位的结果；接着将两个指针向前移动一位。当指针已经走过某个数字的首位，则默认为0。遍历完两个字符串后跳出循环，并根据进位的值决定是否在头部添加进位1。
【复杂度分析】这个方法需要遍历两个字符串，所以时间复杂度为O(N),N为较长的字符串长度。空间复杂度为O(N)。

```python
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        #指针
        i,j = len(num1)-1,len(num2)-1
        #进位标记
        carry = 0
        res = ""
        #处理最高位进位
        while i>=0 or j>=0 or carry>0:
            #处理索引移除
            n1 = int(num1[i]) if i>=0 else 0
            n2 = int(num2[j]) if j>=0 else 0
            #算法基本流程
            tmp = n1+n2+carry
            carry = tmp//10
            res = str(tmp%10)+res
            i,j = i-1,j-1
        return res
```


(5)[multiply-strings](https://leetcode-cn.com/problems/multiply-strings/)

题目概述：字符串乘法

考点:算法+数据结构

注意:翻译成代码逻辑,相乘结果对应结果的哪个索引位

>答题模板:【算法】要解决字符串相乘，可以模拟人工乘法的竖式相乘思路。从后向前把s1的每一位与s2相乘，并把结果错位相加。
【数据结构】首先设置两个变量作为指针，指向字符串中正在被处理的位置，初始化为字符串尾部。再设置一个长度为M+N的变量作为临时结果，初始化为全0。
>【算法流程】翻译成代码逻辑：指针i从后向前逐位遍历s1,对每个i将指针j从后向前遍历，并计算i,j对应元素的乘积，再加上进位。i,j在结果中对应的索引是i+j+1。所以这个值除以10的结果作为进位与i+j位相加，余数作为当前位的结果与i+j+1位相加
遍历完后跳出循环，删去结果中可能存在的前缀0。

```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        # 指针
        i,j = len(num1)-1,len(num2)-1
        # 临时结果
        res = [0]*(len(num1)+len(num2))
        #从后向前扫描
        for i in range(len(num1)-1,-1,-1):
            for j in range(len(num2)-1,-1,-1):
                # ii,jj = len(num1)-1-i,len(num2)-1-j
                tmp = (ord(num1[i])-ord('0'))*(ord(num2[j])-ord('0'))+res[i+j+1]
                res[i+j+1] = tmp%10
                res[i+j] += tmp//10
        #删去前缀0
        while len(res)>1 and res[0]==0:
            res = res[1:]
        return ''.join([str(c) for c in res])
```

(6)[reverse-words-in-a-string](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

题目概述:翻转字符串里的单词

考点:双指针算法+数据结构；注意:边界处理/python字符串不可变

> 答题模板:要翻转字符串里的单词，最简单的方法是【使用语言特性】。很多语言对字符串提供了 split（拆分），reverse（翻转）和 join（连接）等方法，因此可以调用内置的 API 完成操作：（1）使用 split 将字符串按空格分割成字符串数组；（2）使用 reverse 将字符串数组进行反转；（3）使用 join 方法将字符串数组拼成一个字符串。【复杂度分析】时间空间复杂度位O(N)

> 【自己编写对应的函数】首先去除多余的空格，再翻转字符串，最后翻转每个单词。要去除多余的空格，所以要删除字符串头尾的空格，以及中间前置位同为空格的空格；要翻转字符串，可以使用双指针的方法，设置两个变量作为头尾指针交换头尾值,并逐渐把指针向中间收敛。翻转每个单词需要遍历字符串，根据空格的位置确定单词位置，同样使用双指针的方法进行翻转。
> 【复杂度分析】需要遍历字符串常数次，所以时间复杂度O(N)；python字符串不可变，所以需要O(N)的空间存储结果字符串。

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        # 使用内置API
        # return ' '.join([i for i in s.split(' ')][::-1])
        # 自己编写
        #1.去除多余空格
        res = []
        left,right = 0,len(s)-1
        while left<=right and s[left]==' ':
            left+=1
        while left<=right and s[right]==' ':
            right-=1
        for i in range(left,right+1):
            if i==left or (s[i]!=' ' or s[i-1]!=' '):
                res+=[s[i]]
        # print(''.join(res))

        # 2.翻转字符串
        def _reverse(s,left,right):
            while left<right:
                s[left],s[right] = s[right],s[left]
                left,right = left+1,right-1
        _reverse(res,0,len(res)-1)
        # print(''.join(res))
        # 3.翻转字符串里的每个单词
        left = 0
        for i in range(len(res)):
            if res[i] == ' ':
                right = i-1
                _reverse(res,left,right)
                # print(''.join(res),)
                left = i+1
        _reverse(res,left,len(res)-1)
        # print(''.join(res))
        return ''.join(res)
```





(7)[simplify-path](https://leetcode-cn.com/problems/simplify-path/)

题目概述：简化路径

考点:分割字符串+栈数据结构

>答题模板:要简化路径，可以使用分割字符串+栈的思路。将地址字符串按照斜杠分隔成地址数组，遍历地址数组并通过维护一个地址栈处理特殊字符：
（1）遇到表示当前目录的一个点（.）或者空地址，忽略，不处理；（2） 表示上一级目录的两个点（..）且栈不为空时，将栈顶地址出栈（3）合法地址入栈
使用斜杠字符拼接地址栈，得到简化后的路径

```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        # 地址栈
        stack = []
        # 分割为地址数组
        for i in path.split('/'):
            # 忽略
            if len(i)==0 or i == '.':
                continue
            # 出栈
            if  i== '..':
                if len(stack):
                    stack.pop()
            # 合法地址入栈
            else:
                stack.append(i)
        return '/'+'/'.join(stack)
```



(8) [restore-ip-addresses](https://leetcode-cn.com/problems/restore-ip-addresses/)

> 复原IP地址
>这个问题最朴素的解法是暴力法：遍历点可能的所有位置并判断合法性。需要11×10×9=990 次检查。
可以通过约束规划+回溯来减少检查次数，优化时间复杂度。
约束规划：对每个点的放置设置限制。每个点的放置位置只有 3 种可能：上个点的1/2/3个数字之后。只需要检测3×3×3=27种情况。
回溯：当已经放置的点使得无法摆放其他点来生成有效IP地址时，回到之前，改变上一个摆放点的位置。并试着继续。
要实现这个方法，我们需要一个地址栈来保存临时结果。
需要一个回溯函数，以（1）上一个点的位置（2）待放置点的数量（3）当前地址栈 为参数。再把回溯翻译成代码逻辑：如果还有需要放置的点，我们遍历三个有效位置，并判断两个位置之间的部分是否是有效整数，是则（1）将结果压入地址栈，递归调用函数，继续放下一个点；（2）递归返回后，移除栈顶元素，进行回溯。
当点放置完，检查剩余部分是否是有效整数，是则将结果添加到输出列表。

```python
class Solution:
    def restoreIpAddresses(self, s: str):
        def backtrack(prev_pos = -1, dots = 3,cur_res = []):
            if dots==0:
                cur_seg = s[prev_pos+1:]
                if int(cur_seg) not in range(256) or (len(cur_seg)>1 and cur_seg[0]=='0'):
                    return
                res.append('.'.join(cur_res+[cur_seg]))
            else:
                # 每个点的放置位置只有 3 种可能
                for cur_pos in range(prev_pos+1,min(prev_pos+4,len(s))):
                    cur_seg = s[prev_pos+1:cur_pos+1]
                    # 不在0-255或有多余的前缀0
                    if int(cur_seg) not in range(256) or (len(cur_seg)>1 and cur_seg[0]=='0'):
                        continue
                    cur_res+=[cur_seg]
                    backtrack(cur_pos,dots-1,cur_res)
                    cur_res.pop()
        n = len(s)
        res = []
        backtrack()
        return res
                
    def restoreIpAddresses_(self, s: str):
        def _back_track(cur_s=s,cur_res=[]):
            if len(cur_res)==4:
                if len(cur_s)==0:
                    # 增加结果
                    res.append('.'.join(cur_res))
                return
            #剪枝:剩下的字符无法构成ip地址
            if len(cur_s) not in range(1*(4-len(cur_res)),3*(4-len(cur_res))+1):
                return
            for i in range(1,min(4,1+len(cur_s))):
                # 剪枝:当前字符无法构成ip地址的一部分
                if not cur_s[:i].isdigit() or int(cur_s[:i]) not in range(256):
                    break
                if i>1 and cur_s[0]=='0':
                    break
                #增加下一个
                _back_track(cur_s[i:],cur_res+[cur_s[:i]])

        res = []
        _back_track()
        return res
```


(n)[题目名称](https://leetcode-cn.com/problems/minimum-path-sum/)

题目概述

考点:算法+数据结构

注意:

>答题模板:

```python

```