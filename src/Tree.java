import java.util.*;
import java.util.LinkedList;

public class Tree {
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    /*
    中序遍历 递归
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        getRes(res, root);
        return res;
    }

    private void getRes(List<Integer> res, TreeNode node) {
        if (node == null) {
            return;
        }
        getRes(res, node.left);
        res.add(node.val);
        getRes(res, node.right);
    }

    /*
    栈
    1、申请一个栈stack，初始时令cur=head

2、先把cur压入栈中，依次把左边界压入栈中，即不停的令cur=cur.left，重复步骤2

3、不断重复2，直到为null，从stack中弹出一个节点，记为node，打印node的值，并令cur=node.right,重复步骤2

4、当stack为空且cur为空时，整个过程停止。
     */
    public List<Integer> inorderTraversal1(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()) {
            //节点不为空一直压栈
            while (cur != null) {
                stack.push(cur);
                cur = cur.left; //考虑左子树
            }
            //节点为空，就出栈
            cur = stack.pop();
            //当前值加入
            ans.add(cur.val);
            //考虑右子树
            cur = cur.right;
        }
        return ans;
    }

    /**
     * 前序遍历：递归
     */
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        getRes1(res, root);
        return res;
    }

    private void getRes1(List<Integer> res, TreeNode node) {
        if (node == null) {
            return;
        }
        res.add(node.val);
        getRes1(res, node.left);
        getRes1(res, node.right);
    }

    /**
     * 前序遍历：非递归
     */
    public List<Integer> preorderTraversal1(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        ArrayDeque<TreeNode> stack = new ArrayDeque();
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()) {
            if (cur != null) {
                res.add(cur.val);
                stack.push(cur);
                cur = cur.left;
            } else {
                cur = stack.pop();
                cur = cur.right;
            }
        }
        return res;
    }

    /**
     * 前序遍历：非递归
     * 1、申请一个栈stack，然后将头节点压入stack中。
     * <p>
     * 2、从stack中弹出栈顶节点，打印，再将其右孩子节点（不为空的话）先压入stack中，最后将其左孩子节点（不为空的话）压入stack中。
     * <p>
     * 3、不断重复步骤2，直到stack为空，全部过程结束。
     */
    public List<Integer> preorderTraversal2(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        ArrayDeque<TreeNode> stack = new ArrayDeque();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode cur = stack.pop();
            if (cur == null) {
                continue;
            }
            res.add(cur.val);
            stack.push(cur.right);
            stack.push(cur.left);

        }
        return res;
    }

    /**
     * 后序遍历非递归
     */
    public List<Integer> postorderTraversal(TreeNode head) {
        List<Integer> list = new ArrayList<Integer>();
        Stack<TreeNode> stack1 = new Stack<TreeNode>();
        Stack<TreeNode> stack2 = new Stack<TreeNode>();
        if (head != null) {
            stack1.push(head);
            while (!stack1.empty()) {
                head = stack1.pop();
                stack2.push(head);
                if (head.left != null) {
                    stack1.push(head.left);
                }
                if (head.right != null) {
                    stack1.push(head.right);
                }
            }
            while (!stack2.empty()) {
                list.add(stack2.pop().val);
            }
        }
        return list;
    }

    /**
     * No.95 不同的二叉搜索树 II : 递归
     *
     * @param n
     * @return
     */
    public List<TreeNode> generateTrees(int n) {
        List<TreeNode> res = new ArrayList<>();
        if (n == 0) {
            return res;
        }
        return getAns(1, n);
    }

    private List<TreeNode> getAns(int start, int end) {
        List<TreeNode> res = new ArrayList<>();
        if (start > end) {
            res.add(null);
            return res;
        }
        if (start == end) {
            TreeNode treeNode = new TreeNode(start);
            res.add(treeNode);
            return res;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> leftTrees = getAns(start, i - 1);
            List<TreeNode> rightTrees = getAns(i + 1, end);

            for (TreeNode leftTree : leftTrees) {
                for (TreeNode rightTree : rightTrees) {
                    TreeNode root = new TreeNode(i);
                    root.left = leftTree;
                    root.right = rightTree;
                    res.add(root);
                }
            }
        }
        return res;
    }


    /**
     * 96. 不同的二叉搜索树：动态规划
     */
    public int numTrees(int n) {
        if (n == 0) {
            return 0;
        }
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for (int len = 1; len <= n; len++) {
            for (int root = 1; root <= len; root++) {
                int left = root - 1;
                int right = len - root;
                dp[len] += dp[left] * dp[right];
            }
        }
        return dp[n];
    }

    /**
     * 98. 验证二叉搜索树:递归
     */
    public boolean isValidBST(TreeNode root) {
        return helper(root, null, null);
    }

    public boolean helper(TreeNode node, Integer lower, Integer upper) {
        if (node == null) {
            return true;
        }
        int val = node.val;
        if (lower != null && val <= lower) return false;
        if (upper != null && val >= upper) return false;
        return helper(node.right, val, upper) && helper(node.left, lower, val);

    }
    /**
     * 98. 验证二叉搜索树:DFS->树的中序遍历：递增
     */
    public boolean isValidBST1(TreeNode root) {
        if (root == null) {
            return true;
        }
        ArrayDeque<TreeNode> stack = new ArrayDeque();
        TreeNode pre = null;
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (pre != null && root.val <= pre.val) return false;
            pre = root;
            root = root.right;
        }
        return true;

    }

    /**
     * 100.相同的树
     *
     * @param p
     * @param q
     * @return
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        // p and q are both null
        if (p == null && q == null) return true;
        // one of p and q is null
        if (q == null || p == null) return false;
        if (p.val != q.val) return false;
        return isSameTree(p.right, q.right) &&
                isSameTree(p.left, q.left);
    }

    /**
     * 101.对称二叉树:递归
     */
    public boolean isSymmetric(TreeNode root) {
        return isMirror(root, root);
    }

    public boolean isMirror(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) return true;
        if (t1 == null || t2 == null) return false;
        return (t1.val == t2.val) && isMirror(t1.right, t2.right) && isMirror(t1.left, t2.left);
    }

    /**
     * 101.对称二叉树:递归:BFS(迭代)
     */
    public boolean isSymmetric1(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode t1 = queue.poll();
            TreeNode t2 = queue.poll();
            if (t1 == null && t2 == null) continue;
            if (t1 == null || t2 == null) return false;
            if (t1.val != t2.val) return false;
            queue.add(t1.left);
            queue.add(t2.right);
            queue.add(t1.right);
            queue.add(t2.left);
        }
        return true;
    }

    /**
     * 102. 二叉树的层次遍历：递归
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        DFS(root, res, 0);
        return res;
    }

    private void DFS(TreeNode node, List<List<Integer>> res, int level) {
        if (node == null) {
            return;
        }
        if (res.size() <= level) {
            res.add(new ArrayList<>());
        }
        res.get(level).add(node.val);
        DFS(node.left, res, level + 1);
        DFS(node.right, res, level + 1);
    }

    /**
     * 102. 二叉树的层次遍历：队列
     */
    public List<List<Integer>> levelOrder1(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int level = 0;
        while (!queue.isEmpty()) {
            res.add(new ArrayList<>());
            int count = queue.size();
            for (int i = 0; i < count; i++) {
                TreeNode node = queue.remove();
                res.get(level).add(node.val);
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            level++;
        }
        return res;
    }

    /*
    No.103 二叉树的锯齿形层次遍历
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int level = 0;
        while (!queue.isEmpty()) {
            res.add(new ArrayList<>());
            int count = queue.size();
            for (int i = 0; i < count; i++) {
                TreeNode node = queue.remove();
                if (level % 2 == 0)
                    res.get(level).add(node.val);
                else
                    res.get(level).add(0, node.val);

                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            level++;
        }
        return res;
    }

    /**
     * 104. 二叉树的最大深度:递归
     *
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    public int maxDepth1(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int level = 0;
        while (!queue.isEmpty()) {
            int count = queue.size();
            for (int i = 0; i < count; i++) {
                TreeNode node = queue.poll();
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);

            }
            level++;
        }
        return level;
    }

    /**
     * 105. 从前序与中序遍历序列构造二叉树
     *
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }
        return helper(preorder, 0, preorder.length, inorder, 0, inorder.length, map);
    }

    private TreeNode helper(int[] preorder, int p_start, int p_end, int[] inorder, int i_start, int i_end,
                            HashMap<Integer, Integer> map) {
        if (p_start == p_end) {
            return null;
        }
        int root_val = preorder[p_start];
        TreeNode root = new TreeNode(root_val);
        int i_root_index = map.get(root_val);
        int leftNum = i_root_index - i_start;
        root.left = helper(preorder, p_start + 1, p_start + leftNum + 1, inorder, i_start, i_root_index, map);
        root.right = helper(preorder, p_start + leftNum + 1, p_end, inorder, i_root_index + 1, i_end, map);
        return root;
    }

    /*
    二叉树展开为链表
     */
    public void flatten(TreeNode root) {
        while (root != null) {
            //左子树为 null，直接考虑下一个节点
            if (root.left == null) {
                root = root.right;
            } else {
                // 找左子树最右边的节点
                TreeNode pre = root.left;
                while (pre.right != null) {
                    pre = pre.right;
                }
                //将原来的右子树接到左子树的最右边节点
                pre.right = root.right;
                // 将左子树插入到右子树的地方
                root.right = root.left;
                root.left = null;
                // 考虑下一个节点
                root = root.right;
            }
        }
    }

    int max = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        int max = helper1(root);
        return max;
    }

    private int helper1(TreeNode root) {
        if (root == null) return 0;
        int left = Math.max(helper1(root.left), 0);
        int right = Math.max(helper1(root.right), 0);
        max = Math.max(max, root.val + left + right);
        return root.val + Math.max(left, right);
    }

    /**
     * 128. 最长连续序列
     */
    public int longestConsecutive(int[] nums) {
        Arrays.sort(nums);
        int longestCnt = 1;
        int curCnt = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1]) {
                if (nums[i - 1] + 1 == nums[i]) {
                    curCnt++;
                } else {
                    longestCnt = Math.max(longestCnt, curCnt);
                    curCnt = 1;
                }
            }
        }
        return Math.max(longestCnt, curCnt);
    }


    public int longestConsecutive1(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) set.add(num);
        int longestCnt = 0;
        for (int num : nums) {
            if (!set.contains(num - 1)) {
                int curNum = num;
                int curCnt = 1;
                while (set.contains(curNum + 1)) {
                    curNum++;
                    curCnt++;
                }
                longestCnt = Math.max(longestCnt, curCnt);
            }
        }
        return longestCnt;
    }


    public int singleNumber(int[] nums) {
        int ans = 0;
        for (int i = 0; i < nums.length; i++) {
            ans ^= nums[i];
        }
        return ans;
    }

    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int nr = grid.length;
        int nc = grid[0].length;
        int num_islands = 0;
        for (int r = 0; r < nr; r++) {
            for (int c = 0; c < nc; c++) {
                if (grid[r][c] == '1') {
                    ++num_islands;
                    dfs(grid, r, c);
                }
            }
        }
        return num_islands;
    }

    void dfs(char[][] grid, int r, int c) {
        int nr = grid.length;
        int nc = grid[0].length;
        if (r < 0 || c < 0 || r >= nr || c >= nc || grid[r][c] == '0') {
            return;
        }
        grid[r][c] = '0';
        dfs(grid, r - 1, c);
        dfs(grid, r + 1, c);
        dfs(grid, r, c - 1);
        dfs(grid, r, c + 1);

    }


    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int[] indegrees = new int[numCourses];
        for (int[] cp :
                prerequisites) {
            indegrees[cp[0]]++;
        }
        LinkedList<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegrees[i] == 0) queue.addLast(i);
        }
        while (!queue.isEmpty()) {
            Integer pre = queue.removeFirst();
            numCourses--;
            for (int[] req :
                    prerequisites) {
                if (req[1] != pre) continue;
                if (--indegrees[req[0]] == 0) queue.add(req[0]);
            }
        }
        return numCourses == 0;
    }

    public boolean canFinish1(int numCourses, int[][] prerequisites) {
        int[][] adjacency = new int[numCourses][numCourses];
        int[] flags = new int[numCourses];
        for (int[] cp :
                prerequisites) {
            adjacency[cp[1]][cp[0]] = 1;
        }
        for (int i = 0; i < numCourses; i++) {
            if (!dfs(adjacency, flags, i)) return false;
        }
        return true;
    }

    boolean dfs(int[][] adjacency, int[] flags, int i) {
        if (flags[i] == 1) return false;
        if (flags[i] == -1) return true;
        flags[i] = 1;
        for (int j = 0; j < adjacency.length; j++) {
            if (adjacency[i][j] == 1 && !dfs(adjacency, flags, j)) return false;
        }
        flags[i] = -1;
        return true;
    }

    class Trie {
        class TrieNode {
            TrieNode[] children;
            boolean flag;

            public TrieNode() {
                children = new TrieNode[26];
                flag = false;
                for (int i = 0; i < 26; i++) {
                    children[i] = null;
                }
            }
        }

        TrieNode root;

        public Trie() {
            root = new TrieNode();
        }

        public void insert(String word) {
            char[] array = word.toCharArray();
            TrieNode cur = root;
            for (int i = 0; i < array.length; i++) {
                if (cur.children[array[i] - 'a'] == null) {
                    cur.children[array[i] - 'a'] = new TrieNode();
                }
                cur = cur.children[array[i] - 'a'];
            }
            cur.flag = true;
        }

        public boolean search(String word) {
            char[] array = word.toCharArray();
            TrieNode cur = root;
            for (int i = 0; i < array.length; i++) {
                if (cur.children[array[i] - 'a'] == null) {
                    return false;
                }
                cur = cur.children[array[i] - 'a'];
            }
            return cur.flag;
        }

        public boolean startsWith(String prefix) {
            char[] array = prefix.toCharArray();
            TrieNode cur = root;
            for (int i = 0; i < array.length; i++) {
                if (cur.children[array[i] - 'a'] == null) {
                    return false;
                }
                cur = cur.children[array[i] - 'a'];
            }
            return true;
        }

    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode right = invertTree(root.right);
        TreeNode left = invertTree(root.left);
        root.left = right;
        root.right = left;
        return root;
    }

    public TreeNode invertTree1(TreeNode root) {
        if (root == null) return null;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode cur = queue.poll();
            TreeNode temp = cur.left;
            cur.left = cur.right;
            cur.right = temp;
            if (cur.left != null) queue.add(cur.left);
            if (cur.right != null) queue.add(cur.right);
        }
        return root;
    }

    /**
     * 二叉树最近的公共祖先
     * 若p和q要么分别位于左右子树中，那么对左右子结点调用递归函数，会分别返回p和q结点的位置，而当前结点正好就是p和q的最小共同父结点，直接返回当前结点即可。
     * <p>
     * 若p和q同时位于左子树，这里有两种情况，一种情况是left会返回p和q中较高的那个位置，而right会返回空，所以我们最终返回非空的left即可。还有一种情况是会返回p和q的最小父结点，就是说当前结点的左子树中的某个结点才是p和q的最小父结点，会被返回。
     * <p>
     * 若p和q同时位于右子树，同样这里有两种情况，一种情况是right会返回p和q中较高的那个位置，而left会返回空，所以我们最终返回非空的right即可，还有一种情况是会返回p和q的最小父结点，就是说当前结点的右子树中的某个结点才是p和q的最小父结点，会被返回。
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) return root;
        return left != null ? left : right;
    }


    /*
    非递归
     */
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }
        List<TreeNode> pPath = findPath(root, p);
        List<TreeNode> qPath = findPath(root, q);
        TreeNode common = null;
        for (int i = 0, j = 0; i < pPath.size() && j < qPath.size(); i++, j++) {
            if (pPath.get(i) == qPath.get(j)) {
                common = pPath.get(i);
            }
        }
        return common;
    }

    private List<TreeNode> findPath(TreeNode root, TreeNode node) {
        List<TreeNode> path = new ArrayList<>();
        dfs(root, node, new ArrayList<>(), path);
        return path;
    }

    private void dfs(TreeNode root, TreeNode node, List<TreeNode> tmp, List<TreeNode> path) {
        if (root == null) {
            return;
        }
        tmp.add(root);
        if (root == node) {
            path.addAll(new ArrayList<>(tmp));
        }
        dfs(root.left, node, tmp, path);
        dfs(root.right, node, tmp, path);
        tmp.remove(tmp.size() - 1);


    }


    /**
     * leetcode297. 二叉树的序列化与反序列化
     */
    public class Codec {
        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            if (root == null) {
                return "#!";
            }
            String res = root.val + "!";
            res += serialize(root.left);
            res += serialize(root.right);
            return res;
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            String[] values = data.split("!");
            Queue<String> queue = new LinkedList<>();
            for (int i = 0; i < values.length; i++) {
                queue.offer(values[i]);
            }
            return deserialize(queue);
        }

        private TreeNode deserialize(Queue<String> queue) {
            String value = queue.poll();
            if (value.equals("#")) {
                return null;
            }
            TreeNode node = new TreeNode(Integer.valueOf(value));
            node.left = deserialize(queue);
            node.right = deserialize(queue);
            return node;
        }
    }

    /*
    t1是否包含t2的全部拓扑结构
     */
    public boolean contains(TreeNode t1, TreeNode t2) {
        return check(t1, t2) || contains(t1.left, t2) || contains(t1.right, t2);
    }

    public boolean check(TreeNode t, TreeNode t2) {
        if (t2 == null) return true;
        if (t.val != t2.val) return false;
        return check(t.left, t2.left) && check(t.right, t2.right);
    }


    /*
    判断是否是平衡二叉树
     */
    public boolean res = true;

    public boolean isBalance(TreeNode root) {
        getHeight(root);
        return res;
    }

    public int getHeight(TreeNode root) {
        if (root == null) return 0;
        int l = getHeight(root.left);
        int r = getHeight(root.right);
        if (Math.abs(l - r) > 1) {
            res = false;
        }
        return Math.max(l, r) + 1;
    }

    /*
    打家劫舍3
     */
    public int rob(TreeNode root) {
        int[] res = dp(root);
        return Math.max(res[0], res[1]);
    }

    public int[] dp(TreeNode root) {
        if (root == null) {
            return new int[]{0, 0};
        }
        int[] left = dp(root.left);
        int[] right = dp(root.right);
        int rob = root.val + left[0] + right[0];
        int not_rob = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        return new int[]{not_rob, rob};
    }

    /*
    No.112 路径总和
     */
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null)
            return false;

        sum -= root.val;
        if ((root.left == null) && (root.right == null))
            return (sum == 0);
        return hasPathSum(root.left, sum) || hasPathSum(root.right, sum);
    }

    /*
    No.113 路径总和2
     */
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> res = new ArrayList<>();
        pathSumHelper(res, new ArrayList<>(), root, sum);
        return res;
    }

    public void pathSumHelper(List<List<Integer>> result, List<Integer> curPath, TreeNode curNode, int sum) {
        if (curNode == null) {
            return;
        }
        curPath.add(curNode.val);
        if (curNode.val == sum && curNode.left == null && curNode.right == null) {
            result.add(new ArrayList<>(curPath));
        } else {
            pathSumHelper(result, curPath, curNode.left, sum - curNode.val);
            pathSumHelper(result, curPath, curNode.right, sum - curNode.val);

        }
        curPath.remove(curPath.size() - 1);

    }

    /*
    No.129 求根到叶子节点数字之和
     */
    public int sumNumbers(TreeNode root) {
        return sumNumbersHelper(root, 0);
    }

    public int sumNumbersHelper(TreeNode root, int sum) {
        if (root == null) {
            return 0;
        }
        sum = sum * 10 + root.val;
        if (root.left == null && root.right == null) {
            return sum;
        }
        return sumNumbersHelper(root.left, sum) + sumNumbersHelper(root.right, sum);
    }

/*
No.257 二叉树的所有路径
 */
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<String>();
        helper(res, "", root);
        return res;
    }

    public void helper(List<String> res, String curPath, TreeNode root) {
        if (root == null) {
            return;
        }
        curPath += root.val;
        if (root.left == null && root.right == null) {
            res.add(curPath);
        } else {
            curPath += "->";
            helper(res, curPath, root.left);
            helper(res, curPath, root.right);
        }
    }


    public static void main(String[] args) {
        int[][] chars = {{1, 0}, {0, 1}};
        boolean i = new Tree().canFinish1(2, chars);
        System.out.println(i);
    }
}
