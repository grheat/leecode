import com.sun.scenario.effect.impl.sw.sse.SSEBlend_SRC_OUTPeer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class SlideWindow {


    public String minWindow(String s, String t) {
        int[] map = new int[128];
        for (int i = 0; i < t.length(); i++) {
            char char_i = t.charAt(i);
            map[char_i]++;
        }
        int start = 0, minLen = Integer.MAX_VALUE;
        int left = 0, right = 0;
        int count = t.length();
        while (right < s.length()) {
            char charRight = s.charAt(right);
            map[charRight]--;
            if (map[charRight] >= 0) {
                count--;
            }
            while (count == 0) {
                if (right - left + 1 < minLen) {
                    start = left;
                    minLen = right - left + 1;
                }
                char charLeft = s.charAt(left);
                map[charLeft]++;
                if (map[charLeft] > 0) {
                    count++;
                }
                left++;
            }
            right++;
        }
        return s.substring(start, start + minLen);
    }

    public List<Integer> findAnagrams(String s, String t) {
        int[] map = new int[128];
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < t.length(); i++) {
            char char_i = t.charAt(i);
            map[char_i]++;
        }
        int left = 0, right = 0;
        int count = t.length();
        while (right < s.length()) {
            char charRight = s.charAt(right);
            map[charRight]--;
            if (map[charRight] >= 0) {
                count--;
            }
            while (count == 0) {
                if (right - left + 1 == t.length()) {
                    res.add(left);
                }
                char charLeft = s.charAt(left);
                map[charLeft]++;
                if (map[charLeft] > 0) {
                    count++;
                }
                left++;
            }
            right++;
        }
        return res;
    }

    public int lengthOfLongestSubstring(String s) {
        int left = 0, right = 0;
        int[] map = new int[128];
        int res = 0;
        while (right < s.length()) {
            char charRight = s.charAt(right);
            map[charRight]++;
            while (map[charRight] > 1) {
                char charLeft = s.charAt(left);
                map[charLeft]--;
                left++;
            }
            right++;
            res = Math.max(res, right - left);
        }
        return res;
    }


    /*
    No.325 Maximum Sum Subarray of Size K
     */
    public int maxSubArrayLen(int[] nums, int k) {
        int left = 0, right = 0;
        int sum = 0;
        int maxLen = 0;
        while (right < nums.length) {
            sum += nums[right];
            while (sum == k) {
                sum -= nums[left];
                maxLen = Math.max(maxLen, right - left + 1);
                left++;
            }
            right++;
        }
        return maxLen;
    }


    public static void main(String[] args) {
        System.out.println(new SlideWindow().maxSubArrayLen(new int[]{-2, -1, 2, 1}, 1));

    }
}
