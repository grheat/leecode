import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.Stack;

public class dp {

    public String longestPalindrome(String s) {
        if (s == null || s.length() < 2) {
            return s;
        }
        int strLen = s.length();
        int maxStart = 0;  //最长回文串的起点
        int maxEnd = 0;    //最长回文串的终点
        int maxLen = 1;  //最长回文串的长度

        boolean[][] dp = new boolean[strLen][strLen];

        for (int r = 1; r < strLen; r++) {
            for (int l = 0; l < r; l++) {
                if (s.charAt(l) == s.charAt(r) && (r - l <= 2 || dp[l + 1][r - 1])) {
                    dp[l][r] = true;
                    if (r - l + 1 > maxLen) {
                        maxLen = r - l + 1;
                        maxStart = l;
                        maxEnd = r;

                    }
                }

            }

        }
        return s.substring(maxStart, maxEnd + 1);

    }

    public String longestPalindrome2(String s) {
        if (s == null || s.length() < 1) return "";
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    private int expandAroundCenter(String s, int left, int right) {
        int L = left, R = right;
        while (L >= 0 && R < s.length() && s.charAt(L) == s.charAt(R)) {
            L--;
            R++;
        }
        return R - L - 1;
    }

    public String longestPalindrome3(String s) {
        int len = s.length();
        if (len == 0 || len == 1)
            return s;
        int[][] dp = new int[len][len]; //定义二位数组存储值，dp值为1表示true，为0表示false
        int start = 0;  //回文串的开始位置
        int max = 1;   //回文串的最大长度
        for (int i = 0; i < len; i++) {  //初始化状态
            dp[i][i] = 1;
            if (i < len - 1 && s.charAt(i) == s.charAt(i + 1)) {
                dp[i][i + 1] = 1;
                start = i;
                max = 2;
            }
        }

        for (int l = 3; l <= len; l++) {  //l表示检索的子串长度，等于3表示先检索长度为3的子串
            for (int i = 0; i + l - 1 < len; i++) {
                int j = l + i - 1;  //终止字符位置
                if (s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1] == 1) {  //状态转移
                    dp[i][j] = 1;  //是一，不是字母l
                    start = i;
                    max = l;
                }
            }
        }
        return s.substring(start, start + max);   //获取最长回文子串
    }

    /*
    如果 p.charAt(j) == s.charAt(i) : dp[i][j] = dp[i-1][j-1]；
    如果 p.charAt(j) == '.' : dp[i][j] = dp[i-1][j-1]；
    如果 p.charAt(j) == '*'：
    如果 p.charAt(j-1) != s.charAt(i) : dp[i][j] = dp[i][j-2] //in this case, a* only counts as empty
    如果 p.charAt(i-1) == s.charAt(i) or p.charAt(i-1) == '.'：
    dp[i][j] = dp[i-1][j] //in this case, a* counts as multiple a
    or dp[i][j] = dp[i][j-1] // in this case, a* counts as single a
    or dp[i][j] = dp[i][j-2] // in this case, a* counts as empty

     */
    public boolean isMatchNo10(String s, String p) {
        int row = s.length();
        int col = p.length();
        boolean[][] dp = new boolean[row + 1][col + 1];
        dp[0][0] = true;
        for (int i = 1; i <= col; i++) {
            dp[0][i] = i > 1 && dp[0][i - 2] && p.charAt(i - 1) == '*';
        }
        for (int i = 1; i <= row; i++) {
            dp[i][0] = false;
        }
        for (int i = 1; i <= row; i++) {
            for (int j = 1; j <= col; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                }
                if (p.charAt(j - 1) == '*') {
                    if (s.charAt(i - 1) != p.charAt(j - 2)) {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i][j - 2] || dp[i - 1][j - 2] || dp[i - 1][j];
                    }
                }
            }
        }
        return dp[row][col];
    }
    /*
    Leecode 44  https://leetcode-cn.com/problems/wildcard-matching/
    通配符匹配 dp[i][j]表示s到i位置,p到j位置是否匹配!

初始化:

dp[0][0]:什么都没有,所以为true
第一行dp[0][j],换句话说,s为空,与p匹配,所以只要p开始为*才为true
第一列dp[i][0],当然全部为False
动态方程:

如果(s[i] == p[j] || p[j] == "?") && dp[i-1][j-1] ,有dp[i][j] = true

如果p[j] == "*" && (dp[i-1][j] = true || dp[i][j-1] = true) 有dp[i][j] = true

​ note:

​ dp[i][j-1],表示*代表是空字符,例如ab,ab*

​ dp[i-1][j],表示*代表非空任何字符,例如abcd,ab*

     */

    public boolean isMatch(String s, String p) {
        int row = s.length();
        int col = p.length();
        boolean[][] dp = new boolean[row + 1][col + 1];
        dp[0][0] = true;
        for (int i = 1; i <= col; i++) {
            dp[0][i] = dp[0][i - 1] && p.charAt(i - 1) == '*';
        }
        for (int i = 1; i <= row; i++) {
            dp[i][0] = false;
        }

        for (int i = 1; i <= row; i++) {
            for (int j = 1; j <= col; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '?') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
                }
            }
        }
        return dp[row][col];
    }


    public boolean isInterleave(String s1, String s2, String s3) {
        int row = s1.length();
        int col = s2.length();
        boolean[][] dp = new boolean[row + 1][col + 1];
        dp[0][0] = true;
        for (int i = 1; i <= col; i++) {
            dp[0][i] = dp[0][i - 1] && s2.charAt(i - 1) == s3.charAt(i - 1);
        }
        for (int i = 1; i <= row; i++) {
            dp[i][0] = dp[i - 1][0] && s1.charAt(i - 1) == s3.charAt(i - 1);
        }

        for (int i = 1; i <= row; i++) {
            for (int j = 1; j <= col; j++) {
                dp[i][j] = (dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1)) || (dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1));
            }
        }
        return dp[row][col];
    }

    /*
    判断S有多少种方式可以得到T。但其实还是动态规划，我们一个定义二维数组dp，dp[i][j]为字符串s(0,i)变换到t(0,j)的变换方法的个数。

如果S[i]==T[j]，那么dp[i][j] = dp[i-1][j-1] + dp[i-1][j]。意思是：如果当前S[i]==T[j]，那么当前这个字符即可以保留也可以抛弃，所以变换方法等于保留这个字符的变换方法加上不用这个字符的变换方法， dp[i-1][j-1]为保留这个字符时的变换方法个数，dp[i-1][j]表示抛弃这个字符时的变换方法个数。
如果S[i]!=T[i]，那么dp[i][j] = dp[i-1][j]，意思是如果当前字符不等，那么就只能抛弃当前这个字符。
     */
    public int numDistinct(String s, String t) {
        int row = s.length();
        int col = t.length();
        int[][] dp = new int[row + 1][col + 1];
        dp[0][0] = 1;
        for (int i = 1; i <= row; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= col; i++) {
            dp[0][i] = 0;
        }
        for (int i = 1; i <= row; i++) {
            for (int j = 1; j <= col; j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[row][col];
    }


    /*
    用dp数组记录以当前字符为尾的最长有效括号，显然只有在当前字符是右括号)时才有可能有有效子串，所以分两种情况：
1.当前字符s[i]是右括号)，且前一个符号是左括号(。这种情况字符串是这样的“…()”，所以dp[i]=dp[i−2]+2，即前面最长连续有效括号加现在的两个。
2.当前字符s[i]是右括号)，且前一个符号也是右括号)。这种情况字符串是这样的“…))”，这时要判断以前一个右括号为尾的最长有效括号之前的一个字符是否是左括号(，如果是左括号，即“…((…))”，则dp[i]应该等于它和它对应左括号中间的有效括号数dp[i−1]加现在的两个，再加对应左括号之前的最长有效括号数dp[i−dp[i−1]−2]，即dp[i]=dp[i−1]+dp[i−dp[i−1]−2]+2；如果是右括号，即“…)(…))”，则dp[i]是0.

     */

    public int longestValidParentheses(String s) {
        int[] dp = new int[s.length()];
        int res = 0;
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == ')') {
                if (s.charAt(i - 1) == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                } else if (i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                    dp[i] = dp[i - 1] + (i - dp[i - 1] >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
                res = Math.max(res, dp[i]);
            }
        }
        return res;
    }

    /*
不同路径
 */
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            dp[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    /*
    最小路径和
     */
    public int minPathSum(int[][] grid) {
        int row = grid.length;
        int col = grid[0].length;
        int[][] dp = new int[row][col];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < row; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int i = 1; i < col; i++) {
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        }
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                dp[i][j] = Math.min(dp[i - 1][j] + grid[i][j], dp[i][j - 1] + grid[i][j]);
            }
        }
        return dp[row - 1][col - 1];
    }
    /*
    优化为一维数组
     */

    public int minPathSum2(int[][] grid) {
        int row = grid.length;
        int col = grid[0].length;
        int[] dp = new int[col];

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (i == 0 && j != 0) {
                    dp[j] = dp[j - 1] + grid[0][j];
                } else if (j == 0 && i != 0) {
                    dp[j] = dp[j] + grid[i][0];
                } else if (i != 0 && j != 0) {
                    dp[j] = Math.min(dp[j - 1] + grid[i][j], dp[j] + grid[i][j]);
                } else {
                    dp[j] = grid[i][j];
                }
            }
        }
        return dp[col - 1];
    }

    public int climbStairs(int n) {

        if (n == 1) {
            return 1;
        }
        if (n == 2) {
            return 2;
        }
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    /*
    编辑距离

    dp[i][j] 代表 word1 到 i 位置转换成 word2 到 j 位置需要最少步数

所以，

当 word1[i] == word2[j]，dp[i][j] = dp[i-1][j-1]；

当 word1[i] != word2[j]，dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1

其中，dp[i-1][j-1] 表示替换操作，dp[i-1][j] 表示删除操作，dp[i][j-1] 表示插入操作。


     */

    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        dp[0][0] = 0;
        for (int i = 1; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int i = 1; i <= n; i++) {
            dp[0][i] = i;
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1], dp[i][j - 1]), dp[i - 1][j]) + 1;
                }
            }
        }
        return dp[m][n];

    }

    public int maxProfit(int[] prices) {
        int n = prices.length;
        int dp = 0;
        int max = 0;
        for (int i = 1; i < n; i++) {
            int num = prices[i] - prices[i - 1];
            dp = Math.max(dp + num, num);
            max = Math.max(max, dp);
        }
        return max;
    }

    /*
    139. 单词拆分
     */

    public boolean wordBreak(String s, List<String> wordDict) {
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                dp[i] = dp[j] && wordDict.contains(s.substring(j, i));
                if (dp[i]) {
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    /*
    先定义一个数组 dpMax，用 dpMax[i] 表示以第 i 个元素的结尾的子数组，乘积最大的值，也就是这个数组必须包含第 i 个元素。

那么 dpMax[i] 的话有几种取值。

当 nums[i] >= 0 并且dpMax[i-1] > 0，dpMax[i] = dpMax[i-1] * nums[i]
当 nums[i] >= 0 并且dpMax[i-1] < 0，此时如果和前边的数累乘的话，会变成负数，所以dpMax[i] = nums[i]
当 nums[i] < 0，此时如果前边累乘结果是一个很大的负数，和当前负数累乘的话就会变成一个更大的数。所以我们还需要一个数组 dpMin 来记录以第 i 个元素的结尾的子数组，乘积最小的值。
当dpMin[i-1] < 0，dpMax[i] = dpMin[i-1] * nums[i]
当dpMin[i-1] >= 0，dpMax[i] = nums[i]
当然，上边引入了 dpMin 数组，怎么求 dpMin 其实和上边求 dpMax 的过程其实是一样的。

按上边的分析，我们就需要加很多的 if else来判断不同的情况，这里可以用个技巧。

我们注意到上边dpMax[i] 的取值无非就是三种，dpMax[i-1] * nums[i]、dpMin[i-1] * nums[i] 以及 nums[i]。

所以我们更新的时候，无需去区分当前是哪种情况，只需要从三个取值中选一个最大的即可。
     */
    public int maxProduct(int[] nums) {
        int n = nums.length;
        if (n == 0) {
            return 0;
        }
        int dpMax = nums[0];
        int dpMin = nums[0];
        int max = nums[0];
        for (int i = 1; i < n; i++) {
            int preMax = dpMax;
            dpMax = Math.max(dpMin * nums[i], Math.max(dpMax * nums[i], nums[i]));
            dpMin = Math.min(dpMin * nums[i], Math.min(preMax * nums[i], nums[i]));
            max = Math.max(dpMax, max);
        }
        return max;
    }

    /*

     */
    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 0) {
            return 0;
        }
        if (n == 1) {
            return nums[0];
        }
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = nums[0];
        int max = dp[1];
        for (int i = 2; i <= n; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i - 1], dp[i - 1]);
        }
        return dp[n];
    }




}
