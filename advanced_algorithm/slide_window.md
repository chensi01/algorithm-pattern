# 滑动窗口

每次窗口滑动，只改变边界头尾，所以尝试在常数时间内完成窗口内数据的更新以优化时间复杂度

## 模板

```cpp
/* 滑动窗口算法框架 */
void slidingWindow(string s, string t) {
    unordered_map<char, int> need, window;
    for (char c : t) need[c]++;

    int left = 0, right = 0;
    int valid = 0;
    while (right < s.size()) {
        // c 是将移入窗口的字符
        char c = s[right];
        // 右移窗口
        right++;
        // 进行窗口内数据的一系列更新
        ...

        /*** debug 输出的位置 ***/
        printf("window: [%d, %d)\n", left, right);
        /********************/

        // 判断左侧窗口是否要收缩
        while (window needs shrink) {
            // d 是将移出窗口的字符
            char d = s[left];
            // 左移窗口
            left++;
            // 进行窗口内数据的一系列更新
            ...
        }
    }
}
```

需要变化的地方

- 1、右指针右移之后窗口数据更新
- 2、判断窗口是否要收缩
- 3、左指针右移之后窗口数据更新
- 4、根据题意计算结果

## 示例

[minimum-window-substring](https://leetcode-cn.com/problems/minimum-window-substring/)

> 给你一个字符串 S、一个字符串 T，请在字符串 S 里面找出：包含 T 所有字母的最小子串

```python
#只需包含所有字符,不需要保证顺序
def minWindow(self, s: str, t: str) -> str:
    left_i,right_i = 0,-1
    #维护两个哈希表,分别记录子串中每个字符的(1)已出现次数(2)应出现次数
    need = {}
    have = {}
    for c in t:
        need[c] = need.get(c,0)+1
        have[c] = 0
    #count表示已满足子串t的字符个数
    count = 0
    res = s*2
    while right_i<len(s)-1:
        # 右移窗口
        right_i+=1
        # 数据更新
        c = s[right_i]
        if have.get(c,0)<need.get(c,0):
            count+=1
        if c in have:
            have[c] += 1
        # 收缩左侧窗口
        while count == len(t):
            if (right_i-left_i+1)<len(res):
                res = s[left_i:right_i+1]
            #收缩左侧窗口,更新收缩后结果
            c = s[left_i]
            left_i+=1
            if c in need :
                have[c] -=1
                #当前解不满足,继续向右寻找
                if have[c]<need[c]:
                    count-=1
    return res if res!=(s*2) else ""

```

[permutation-in-string](https://leetcode-cn.com/problems/permutation-in-string/)

> 给定两个字符串  **s1**  和  **s2**，写一个函数来判断  **s2**  是否包含  **s1 **的排列。

```python
def checkInclusion(self, s1: str, s2: str) -> bool:
    #维护两个哈希表,分别记录子串中每个字符的(1)已出现次数(2)应出现次数
    need,have = {},{}
    for c in s1:
        need[c] = need.get(c,0)+1
    left_i,right_i = 0,-1
    while right_i<len(s2)-1:
        # 右移窗口,数据更新
        right_i+=1
        c = s2[right_i]
        have[c] = have.get(c,0)+1
        # 收缩左侧边界
        while have.get(c,0)>need.get(c,0):
            have[s2[left_i]]-=1
            left_i+=1
        if (right_i-left_i+1) == len(s1):
            return True
    return False
```

[find-all-anagrams-in-a-string](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)

> 给定一个字符串  **s **和一个非空字符串  **p**，找到  **s **中所有是  **p **的字母异位词的子串，返回这些子串的起始索引。

```python
def findAnagrams(self, s: str, p: str) -> List[int]:
    #维护两个哈希表,分别记录子串中每个字符的(1)已出现次数(2)应出现次数
    need,have = {},{}
    for c in p:
        need[c] = need.get(c,0)+1
    left_i,right_i = 0,-1
    res = []
    while right_i<len(s)-1:
        # 右移窗口,数据更新
        right_i+=1
        c = s[right_i]
        have[c] = have.get(c,0)+1
        # 收缩左侧边界
        while have.get(c,0)>need.get(c,0):
            have[s[left_i]]-=1
            left_i+=1
        if (right_i-left_i+1) == len(p):
            res+=[left_i]
            #
            have[s[left_i]]-=1
            left_i+=1
    return res
```

[longest-substring-without-repeating-characters](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

> 给定一个字符串，请你找出其中不含有重复字符的   最长子串   的长度。
> 示例  1:
>
> 输入: "abcabcbb"
> 输出: 3
> 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

```python
def lengthOfLongestSubstring(self, s: str) -> int:
    res,left_i,right_i = 0,0,0
    have = {}
    while right_i<len(s):
        have[s[right_i]]=have.get(s[right_i],0)+1
        if have[s[right_i]]==1:
            res = max(right_i-left_i+1,res)
        else:
            while left_i<=right_i and have[s[right_i]]>1:
                have[s[left_i]]-=1
                left_i+=1
        right_i+=1
    return res
```

## 总结

- 和双指针题目类似，更像双指针的升级版，滑动窗口核心点是维护一个窗口集，根据窗口集来进行处理
- 核心步骤
  - right 右移
  - 收缩
  - left 右移
  - 求结果

## 练习

- [ ] [minimum-window-substring](https://leetcode-cn.com/problems/minimum-window-substring/)
- [ ] [permutation-in-string](https://leetcode-cn.com/problems/permutation-in-string/)
- [ ] [find-all-anagrams-in-a-string](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)
- [ ] [longest-substring-without-repeating-characters](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)
