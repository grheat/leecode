import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Test {

    public static int[] change(int[] arr) {
        int n = arr.length;
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        int index = 0;
        int len = 0;
        for (int i = 0; i < dp.length; i++) {
            for (int j = 0; j < i; j++) {
                if (arr[j] < arr[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            if (dp[i] > len) {
                len = Math.max(dp[i], len);
                index = i;
            }

        }
        int[] res = new int[len];
        res[--len] = arr[index];
        for (int i = index; i >= 0; i--) {
            if (dp[i] == dp[index] - 1 && arr[i] < arr[index]) {
                res[--len] = arr[i];
                index = i;
            }
        }
        return res;
    }


    public List<Integer> preorderTraversal1(Tree.TreeNode root) {
        ArrayDeque<Tree.TreeNode> stack = new ArrayDeque<>();
        List<Integer> res = new ArrayList<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            stack.pop();
            res.add(root.val);
            stack.add(root.right);
            stack.add(root.left);

        }
        return res;
    }

    public List<Integer> inOrder(Tree.TreeNode root) {
        ArrayDeque<Tree.TreeNode> stack = new ArrayDeque<>();
        List<Integer> res = new ArrayList<>();
        Tree.TreeNode cur = root;
        while (cur != null && !stack.isEmpty()) {
            stack.push(cur);
            cur = cur.left;
        }
        cur = stack.pop();
        res.add(cur.val);
        cur = cur.right;
        return res;
    }

    public static char[] reverseString(char[] string) {
        if (string == null || string.length == 0) {
            return null;
        }
        int left = 0, right = string.length - 1;
        while (left < right) {
            char tmp = string[left];
            string[left] = string[right];
            string[right] = tmp;
            left++;
            right--;
        }
        return string;
    }

    public static int reverseInteger(int n) {
        if (n == 0) {
            return 0;
        }
        String s = String.valueOf(Math.abs(n));
        String res = new String(reverseString(s.toCharArray()));
        long l = Long.parseLong(res);
        if (n > 0) {
            if (l > Integer.MAX_VALUE) {
                return 0;
            } else {
                return (int) l;
            }
        } else {
            if (-l < Integer.MIN_VALUE) {
                return 0;
            } else {
                return (int) -l;
            }
        }
    }

    private int count = 0;
    private final Object lock1 = new Object();
    private final Object lock2 = new Object();
    private final Object lock3 = new Object();

    public void multiTest() {
        Thread t1 = new Thread(() -> {
            while (count <= 99) {
                synchronized (lock2) {
                    synchronized (lock1) {
                        System.out.println("t1: " + count);
                        count++;
                        lock1.notifyAll();
                    }
                    try {
                        if (count <= 99) {
                            lock2.wait();
                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        });
        Thread t2 = new Thread(() -> {
            while (count <= 99) {
                synchronized (lock3) {
                    synchronized (lock2) {
                        System.out.println("t2: " + count);
                        count++;
                        lock2.notifyAll();
                    }
                    try {
                        if (count <= 99) {
                            lock3.wait();
                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        });
        Thread t3 = new Thread(() -> {
            while (count <= 99) {
                synchronized (lock1) {
                    synchronized (lock3) {
                        System.out.println("t3: " + count);
                        count++;
                        lock3.notifyAll();
                    }
                    try {
                        if (count <= 99) {
                            lock1.wait();
                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        });
        t1.start();
        t2.start();
        t3.start();
    }



    public static void main() {
        System.out.println("main 2");
    }
}
