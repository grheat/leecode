import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class BackTrack {
    /*
    全排列
     */
    List<List<Integer>> res = new LinkedList();

    List<List<Integer>> permute(int[] nums) {

        LinkedList<Integer> track = new LinkedList<>();
        backtrack(nums, track);
        return res;
    }

    void backtrack(int[] nums, LinkedList<Integer> track) {
        if (track.size() == nums.length) {
            res.add(track);
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (track.contains(nums[i]))
                continue;
            track.add(nums[i]);
            backtrack(nums, track);
            track.removeLast();
        }
    }


    /**
     * N皇后
     */

    public List<List<String>> solveNQueens(int n) {
        List<List<String>> ans = new ArrayList<>();
        backtrack(new ArrayList<Integer>(), ans, 0);
        return ans;
    }

    private void backtrack(List<Integer> currentQueen, List<List<String>> ans, int n) {
        if (currentQueen.size() == n) {
            List<String> temp = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                char[] t = new char[n];
                Arrays.fill(t, '.');
                t[currentQueen.get(i)] = 'Q';
                temp.add(new String(t));
            }
            ans.add(temp);
            return;
        }
        for (int col = 0; col < n; col++) {
            if (!currentQueen.contains(col)) {
                if (isDiagonalAttack(currentQueen, col)) {
                    continue;
                }
                currentQueen.add(col);
                backtrack(currentQueen, ans, n);
                currentQueen.remove(currentQueen.size() - 1);
            }
        }

    }
    private boolean isDiagonalAttack(List<Integer> currentQueen, int i) {
        // TODO Auto-generated method stub
        int current_row = currentQueen.size();
        int current_col = i;
        //判断每一行的皇后的情况
        for (int row = 0; row < currentQueen.size(); row++) {
            //左上角的对角线和右上角的对角线，差要么相等，要么互为相反数，直接写成了绝对值
            if (Math.abs(current_row - row) == Math.abs(current_col - currentQueen.get(row))) {
                return true;
            }
        }
        return false;
    }


    class Nqueue {
        public List<List<String>> res = new ArrayList<>();
        public List<List<String>> solveNQueens(int n) {
            int[][] board = new int[n][n];
            dfs(n, 0, board);
            return res;
        }

        private void dfs(int n, int row, int[][] board) {
            if (row == n) {
                res.add(track(board, n));
                return;
            }
            for (int col = 0; col < n; col++) {
                if (isValid(board, row, col)) {
                    board[row][col] = 1;
                    dfs(n, row + 1, board);
                    board[row][col] = 0;
                }
            }
        }

        private boolean isValid(int[][] board, int row, int col) {
            int n = board.length;
            for (int i = 0; i < row; i++) {
                if (board[i][col] == 1) return false;
            }
            for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--) {
                if (board[i][j] == 1) return false;
            }
            for (int i = row - 1, j = col - 1;
                 i >= 0 && j >= 0; i--, j--) {
                if (board[i][j] == 'Q')
                    return false;
            }
            return true;
        }
        private List<String> track(int[][] board, int n) {
            List<String> list=new ArrayList<>();
            for (int i = 0; i < n; i++) {
                StringBuilder temp=new StringBuilder();
                for (int j = 0; j < n; j++) {
                    if (board[i][j]==0)temp.append('.');
                    else temp.append('Q');
                }
                list.add(temp.toString());
            }
            return list;
        }


    }

}
