package answer;

//import java.math.BigDecimal;
//import java.math.RoundingMode;
//import java.text.NumberFormat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;

public class Solution {
	
	public int singleNumber(int[] A) {
      int result = 0;
      for(int i=0; i<A.length; i++){
      	result = result ^ A[i];
      }
      return result;
  }
	
	 public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
	        ListNode c1 = l1;
	        ListNode c2 = l2;
	        ListNode sentinel = new ListNode(0);
	        ListNode d = sentinel;
	        int sum = 0;
	        while (c1 != null || c2 != null) {
	            sum /= 10;
	            if (c1 != null) {
	                sum += c1.val;
	                c1 = c1.next;
	            }
	            if (c2 != null) {
	                sum += c2.val;
	                c2 = c2.next;
	            }
	            d.next = new ListNode(sum % 10);
	            d = d.next;
	        }
	        if (sum / 10 == 1)
	            d.next = new ListNode(1);
	        return sentinel.next;
	    }

	 public int findMin(int[] num) {
		for(int i=0; i<num.length-1; i++){
			if(num[i]>num[i+1])
				return num[i+1];
		}
		return num[0];
		
/*		 int start = 0;
	        int end = num.length - 1;
	        while(start < end) {
	            int mid = (start + end) / 2;
	            if(num[mid] >= num[end]) {
	                start = mid;
	                if(end-mid == 1) return num[end];
	            } else if(num[start] >= num[mid]) {
	                end = mid;
	                if(mid-start == 1) return num[mid];
	            } else {
	                return num[start];
	            }
	        }
	        return num[end];*/
  }
	 
	 public static int titleToNumber(String s) {
		char c = 0;
		int num = 0;
		int total = 0;
		int power = 1;
		for(int i=s.length()-1; i>=0; i--){
			c = s.charAt(i);
			num = c - 'A' + 1;
			total += num*power;
			power *= 26;
		}
      return total;
  }
	 
	 public static List<Integer> grayCode(int n) {
		 List<Integer> list = new ArrayList<Integer>();
		 if(n==0){
			 list.add(0);
			 return list;
		 }
		 else if(n >= 1){
			 List<Integer> list1 = grayCode(n - 1);
			 for(int i=0; i<list1.size(); i++){
				 list.add(list1.get(i));
			 }
			 for(int i=list.size()-1; i>=0; i--){
				 list.add(list1.get(i) + (1<<(n-1)));
			 }
		 }
		return list;
  }
	 
	 public static int numDecodings(String s) {
		 	int len = s.length();
		    char[] nums = s.toCharArray();
		    if (len == 0 || nums[0] == '0') return 0;
		    int[] ways = new int[len];
		    ways[0] = 1;

		    for (int i=1; i<len; i++) {
		        // Add result calculated step back
		        ways[i] = ways[i-1];

		        // "00", "30"..."90" are not accepted
		        if (nums[i] == '0' && (nums[i-1] < '1' || nums[i-1] > '2')) return 0;

		        // Add result calculated two steps back
		        // if "11"..."19" or "21...26" but not like "110" or "210"
		        if (nums[i-1] == '1' && nums[i] != '0' && ((i+1 < len && nums[i+1] != '0') || i+1 == len)) {
		            ways[i] += i-2 > 0 ? ways[i-2] : 1;
		        }
		        if (nums[i-1] == '2' && nums[i] > '0' && nums[i] < '7' && ((i+1 < len && nums[i+1] != '0') || i+1 == len)) {
		            ways[i] += i-2 > 0 ? ways[i-2] : 1;
		        }
		    }
		    return ways[len-1];
	    }

	 
	 public int climbStairs(int n) {
		 if(n==0) {
			 return 0;
		 }
		 
		 int[] ways = new int[n];
		 ways[0] = 1;
		 for(int i=1; i<n; i++){
			 if(i==1) {
				 ways[i] = 2;
			 }
			 else {
				 ways[i] = ways[i-1] + ways[i-2];
			 }
		 }
		 return ways[n-1];
	 } 
	 
	 public static boolean isValidSudoku(char[][] board) {
	       int count = 9;
	       for (int i=0; i<count; i++) {
	           boolean[] rowExist = new boolean[count+1];
	           boolean[] colExist = new boolean[count+1];
	           boolean[] matrixExist = new boolean[count+1];
	           for (int j=0; j<count; j++) {
	               int rowNum = board[i][j] == '.' ? -1 : board[i][j] - '0';
	               int colNum = board[j][i] == '.' ? -1 : board[j][i] - '0';
	               int mtxRowIdx = 3*(i/3);
	               int mtxColIdx = 3*(i%3);
	               int matrixNum = board[mtxRowIdx + j/3][mtxColIdx + j%3] == '.' ? 
	                                   -1 : board[mtxRowIdx + j/3][mtxColIdx + j%3] - '0';
	               if (rowNum > 0 && rowExist[rowNum] ||
	                   colNum > 0 && colExist[colNum] || 
	                   matrixNum > 0 && matrixExist[matrixNum]) {
	                   return false;        
	               }
	               if (rowNum > 0){
	                   rowExist[rowNum] = true;
	               }
	               if (colNum > 0){
	                   colExist[colNum] = true;
	               }
	               if (matrixNum > 0){
	                   matrixExist[matrixNum] = true;
	               }
	           }
	       }
	       return true;
	 }
	 
	 public static boolean isValid(String s) {
		 Stack<Character> stack = new Stack<Character>();
	        // Iterate through string until empty
	        for(int i = 0; i<s.length(); i++) {
	            // Push any open parentheses onto stack
	            if(s.charAt(i) == '(' || s.charAt(i) == '[' || s.charAt(i) == '{')
	                stack.push(s.charAt(i));
	            // Check stack for corresponding closing parentheses, false if not valid
	            else if(s.charAt(i) == ')' && !stack.empty() && stack.peek() == '(')
	                stack.pop();
	            else if(s.charAt(i) == ']' && !stack.empty() && stack.peek() == '[')
	                stack.pop();
	            else if(s.charAt(i) == '}' && !stack.empty() && stack.peek() == '{')
	                stack.pop();
	            else
	                return false;
	        }
	        // return true if no open parentheses left in stack
	        return stack.empty();
	 }
	 
	 public static int sqrt(int x) {
		 //return (int)Math.sqrt(x);
		   if (x==0) 
	        {
	          return 0;
	        }
	       int begin = 1;
	       int end = x;
	       int result=0;
	       while (begin <= end) {
	          int mid = (begin+end) / 2;
	          if (mid <= x/mid ) {
	               begin = mid + 1;
	               result = mid;
	                } else {
	                 end = mid - 1;
	          }
	      }
	      return result;
	 } 
	 
	 public static int reverse(int x) {
		 int result = 0;
	    while (x != 0){
	        int tail = x % 10;
	        int newResult = result * 10 + tail;
	        if ((newResult - tail) / 10 != result)
	        			{ return 0; }
	        result = newResult;
	        x = x / 10;
	    }
	    return result;
	 }
	 
	 public static String convertToTitle(int n) {
	     StringBuilder buffer = new StringBuilder();
	     char c = 0;
	     while(n != 0){
	    	 int i = n%26;
	    	 if(i == 0) {
	    		 c = 'Z';
	    		 n -= 26;
	    	 }
	    	 else {
	    		 c = (char)('A' + i - 1);
	    	 }
	    	 buffer.insert(0, c);
	    	 n /= 26;
	     }
	     return buffer.toString();
   }
	 
	 public int minPathSum(int[][] grid) {
       // DP
       if(grid.length==0){return 0;}
       // base case
       for(int i = 1;i < grid.length;i++)
           grid[i][0] += grid[i-1][0];
       for(int j = 1;j < grid[0].length;j++)
           grid[0][j] += grid[0][j-1];
       // iteration
       for(int i = 1;i < grid.length;i++)
           for(int j = 1;j < grid[0].length;j++)
               grid[i][j] += Math.min(grid[i-1][j],grid[i][j-1]);
       return grid[grid.length-1][grid[0].length-1];
   }
	 
	 public static int candy(int[] ratings) {
		 if (ratings == null || ratings.length == 0) return 0;
	        int total = 1, prev = 1, countDown = 0;
	        for (int i = 1; i < ratings.length; i++) {
	            if (ratings[i] >= ratings[i-1]) {
	                if (countDown > 0) {
	                    total += countDown*(countDown+1)/2; // arithmetic progression
	                    if (countDown >= prev) total += countDown - prev + 1;
	                    countDown = 0;
	                    prev = 1;
	                }
	                prev = ratings[i] == ratings[i-1] ? 1 : prev+1;
	                total += prev;
	            } else countDown++;
	        }
	        if (countDown > 0) { // if we were descending at the end
	            total += countDown*(countDown+1)/2;
	            if (countDown >= prev) total += countDown - prev + 1;
	        }
	        return total;
	 }
	 
	 public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
		    ListNode head = new ListNode(0);
	        ListNode headcopy = head;
	        while(l1 != null && l2 != null){
	            if(l1.val < l2.val){
	                headcopy.next = l1;
	                l1 = l1.next;
	            }
	            else if(l2.val <= l1.val){
	                headcopy.next = l2;
	                l2 = l2.next;
	            }
	            headcopy = headcopy.next;
	        }

	        headcopy.next = (l1 == null ? l2 : l1);   
	        return head.next;
	 }
	 
/*	 3 steps:
   1. Reverse find first number which breaks descending order.
	 2. Exchange this number with the least number that's greater than this number.
	 3. Reverse sort the numbers after the exchanged number.*/
	 public void nextPermutation(int[] num) {
       int i = num.length - 2;
       for(; i >= 0 && num[i] >= num[i+1]; i--) 
           ;

       if(i >= 0) {
           int j = i + 1;
           for(; j<num.length && num[i] < num[j]; j++) 
               ;
           swap(num, i, j-1);
       }

       i ++ ; 
       int k = num.length - 1;
       for(; i<k; i++, k--)
           swap(num, i, k);
   }
	 
	 public List<List<Integer>> permute(int[] num) {
		List<List<Integer>> ans = new ArrayList<List<Integer>>();
	    if (num.length ==0) return ans;
	    List<Integer> l0 = new ArrayList<Integer>();
	    l0.add(num[0]);
	    ans.add(l0);
	    for (int i = 1; i< num.length; ++i){
	        List<List<Integer>> new_ans = new ArrayList<List<Integer>>(); 
	        for (int j = 0; j<=i; ++j){            
	           for (List<Integer> l : ans){
	               List<Integer> new_l = new ArrayList<Integer>(l);
	               new_l.add(j,num[i]);
	               new_ans.add(new_l);
	           }
	        }
	        ans = new_ans;
	    }
	    return ans;
  }
	 
	 public static ListNode insertionSortList(ListNode head) {
		   if( head == null ){
	            return head;
	        }

	        ListNode helper = new ListNode(0); //new starter of the sorted list
	        ListNode cur = head; //the node will be inserted
	        ListNode pre = helper; //insert node between pre and pre.next
	        ListNode next = null; //the next node will be inserted
	        //not the end of input list
	        while( cur != null ){
	            next = cur.next;
	            //find the right place to insert
	            while( pre.next != null && pre.next.val <= cur.val ){
	                pre = pre.next;
	            }
	            //insert between pre and pre.next
	            cur.next = pre.next;
	            pre.next = cur;
	            pre = helper;
	            cur = next;
	        }

	        return helper.next;
	 }
	 
	 public static String longestPalindrome(String s) {
		 StringBuilder longest = new StringBuilder("");
		  if (s.length() <= 1) return s;

		    for (int i = 0; i < s.length(); i++) {
		        expand(s, longest, i, i); //odd
		        expand(s, longest, i, i + 1); //even
		    }

		    return longest.toString();
	 }
	 
	 private static void expand(String s, StringBuilder longest, int i, int j) {
	    while (i >= 0 && j < s.length()) {
	        if (s.charAt(i) == s.charAt(j)) {
	            if (j - i + 1 > longest.length()) {
	                longest.delete(0, longest.length());
	                longest.append(s.substring(i, j + 1));
	            }
	            i--;
	            j++;
	        }
	        else
	            break;
	    }
	 }
	 
	 public static int singleNumber2(int[] A) {
		    int[] result = new int[32];
		    for (int i = 0; i < 32; i++) {
		        int temp = 0;
		        for (int j = 0; j < A.length; j++) {
		            temp += (A[j] & (1 << i)) >> i;
		        }
		        result[i] = (temp) % 3;
		    }
		    int finalResult = 0;
		
		    for (int i = 0; i < 32; i++) {
		        finalResult |= result[i] << i;
		    }
		
		    return finalResult;
	 }
	 
	 public static int findPeakElement(int[] num) {
		    int n = num.length;
	        // check first element & last element
	        if (n == 1 || num[0] > num[1]) {
	            return 0;
	        } else if (num[n-1] > num[n-2]) {
	            return n-1;
	        }
	    
	        int l = 1;
	        int h = n-2;
	        while(l < h) {
	            int mid = (l+h)/2;
	            if (num[mid] > num[mid-1] && num[mid] > num[mid+1]) {
	                return mid;
	            } else if (num[mid] < num[mid+1]) {
	                l = mid+1;
	            } else {
	                h = mid-1;
	            }
	        }
	        return l;
	 }
	 
	public ListNode deleteDuplicates(ListNode head) {
		if(head == null || head.next == null) return head;
	      ListNode pre = head;
	      ListNode cur = head.next;
	      while(cur != null){
	    	  if(cur.val == pre.val){
	    		  pre.next = cur.next;
	    		  cur = cur.next;
	    	  }
	    	  else{
	    		  pre = pre.next;
	    		  cur = cur.next;
	    	  }
	      }
	      return head;
  }
	
	 public static void connect(TreeLinkNode root) {
		   if(root == null) return;
		   root.next = null;
		   if(root.left == null) return;
		   
		   TreeLinkNode parent = root;
		   TreeLinkNode left = root.left;
		   TreeLinkNode right = root.right;
		   TreeLinkNode firstLeaf = left;
		   
		   while(parent.next != null || left.left != null){
			   left.next = right;
			   if(parent.next == null){
				   right.next = null;
				   parent = firstLeaf;
				   left = firstLeaf.left;
				   right = firstLeaf.right;
				   firstLeaf = left;
			   }else{
				   right.next = parent.next.left;
				   parent = parent.next;
				   left = parent.left;
				   right = parent.right;
			   }
		   }
		   
		   left.next = right;
		   right.next = null;
	 }
	 
	 public static ListNode rotateRight(ListNode head, int n) {
		 if(n == 0 || head == null || head.next == null) return head;
		 int size = 1;
		 ListNode current = head;
		 while(current.next != null){
			 current = current.next;
			 size++;
		 }
		 int m = size - n%size;
		 if(m == size) return head;
		 current.next = head;
		 current = head;
		 while(m > 1){
			 current = current.next;
			 m--;
		 }
		 ListNode newHead = current.next;
		 current.next = null;
	     return newHead;   
	  }
	 
	 public void sortColors(int[] A) {
       int second = A.length -1, zero = 0;
       for (int i=0; i<=second; i++) {
           while (A[i]==2 && i<second) swap(A, i, second--);
           while (A[i]==0 && i>zero) swap(A, i, zero++);
       }
	 }
	 
	 public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
		 if(headA == null || headB == null){
	            return null;
	        }
	        int length1=0;
	        int length2=0;

	        ListNode head1 = headA;
	        ListNode head2 = headB;
	        while(headA!=null){
	            length1++;
	            headA = headA.next;
	        }
	        while(headB!=null){
	            length2++;
	            headB = headB.next;
	        }
	        int minus = length1-length2;
	        int abs = Math.abs(minus);
	        if(minus<0){
	            int step=abs;
	            while(step>0){
	                head2 = head2.next;
	                step--;
	            }
	        }else{
	            int step=abs;
	            while(step>0){
	                head1 = head1.next;
	                step--;
	            }
	        }
	        if(head1==head2){
	            return head1;
	        }
	        while(head1!=null&&head2!=null&&head1!=head2){
	            head1= head1.next;
	            head2= head2.next;
	        }

	        return head1;


	 }
	 
	 public String largestNumber(int[] num) {
		 if(num==null || num.length==0)
		        return "";
	    String[] Snum = new String[num.length];
	    for(int i=0;i<num.length;i++)
	        Snum[i] = num[i] + "";

	    Comparator<String> comp = new Comparator<String>(){
	        @Override
	        public int compare(String str1, String str2){
	            String s1 = str1+str2;
	            String s2 = str2+str1;
	            return s1.compareTo(s2);
	        }
	    };

	    Arrays.sort(Snum, comp);
	    if(Snum[Snum.length-1].charAt(0)=='0')
	        return "0";

	    StringBuilder sb = new StringBuilder();

	    for(String s: Snum)
	        sb.insert(0, s);

	    return sb.toString();
	 }
	 
	 public List<List<Integer>> combinationSum(int[] candidates, int target) {
	        Arrays.sort(candidates);
	        List<List<Integer>> result = new ArrayList<List<Integer>>();
	        getResult(result, new ArrayList<Integer>(), candidates, target, 0);

	        return result;
	    }

	    private void getResult(List<List<Integer>> result, List<Integer> cur, int candidates[], int target, int start){
	        if(target > 0){
	            for(int i = start; i < candidates.length && target >= candidates[i]; i++){
	                cur.add(candidates[i]);
	                getResult(result, cur, candidates, target - candidates[i], i);
	                cur.remove(cur.size() - 1);
	            }//for
	        }//if
	        else if(target == 0 ){
	            result.add(new ArrayList<Integer>(cur));
	        }//else if
	    }
	    
  public boolean isSymmetric(TreeNode root) {
  	if(root== null){
  		return true;
  	}
  	else return areSymmetric(root.left, root.right);
  }
  
  private boolean areSymmetric(TreeNode node1, TreeNode node2){
  	if(node1 == null && node2 == null) return true;
  	if((node1 == null && node2 != null) || (node1 != null && node2 == null))
  		return false;
  	else{
  		if(node1.val != node2.val) return false;
  		else{
  			if(node1.left == null && node1.right == null && 
  					node2.left == null && node2.right == null)
  				return true;
  			if(areSymmetric(node1.left, node2.right) && 
  					areSymmetric(node1.right, node2.left))
  				return true;
  			else return false;
  		}
  	}
  }
	
 public static String countAndSay(int n) {
	   if(n == 1) return "1";
	   String str = "1";
	   StringBuilder sb = new StringBuilder();
	   char current = 0;
	   char last = 0;
	   int count = 1;
	   
	   for(int i=0; i<n-1; i++){
		   last = str.charAt(0); 
		   if(str.length() == 1){
			   sb.append('1');
			   sb.append(last);
		   }
		   for(int j=1; j<str.length(); j++){
			   current = str.charAt(j);
			   if(current != last){
				   sb.append(count).append(last);
				   count = 1;
			   }else {
				   count++; 
			   }
			   last = current;
			   if(j == str.length()-1){
				   sb.append(count).append(current);
				   count = 1;
			   }
		   }
		   str = sb.toString();
		   sb.delete(0, sb.length());
	   }
	   return str;
 }
 
 public TreeNode sortedListToBST(ListNode head) {
	   if(head==null) return null;
	    if(head.next==null) return new TreeNode(head.val);

	    ListNode slow = head, fast = head.next;
	    while(fast.next!=null&&fast.next.next!=null){
	        fast=fast.next.next;
	        slow=slow.next;
	    }

	    TreeNode root = new TreeNode(slow.next.val);

	    ListNode second=slow.next.next;
	    slow.next=null;

	    root.left = sortedListToBST(head);
	    root.right = sortedListToBST(second);

	    return root;
 }
 
 
 TreeNode current = null;  //定义一个全局变量
 public void flatten(TreeNode root) {
     if(root == null) return;
     flateenHelper(root);
 }
 
 private void flateenHelper(TreeNode currentNode){
	   if(currentNode == null) return;
	   current = currentNode;
	   TreeNode rightNode = currentNode.right;
	   TreeNode leftNode = currentNode.left;
	   currentNode.left = null;
	   if(leftNode != null){
		   currentNode.right = leftNode;
		   flateenHelper(leftNode);
	   }
	   current.right = rightNode;
     flateenHelper(rightNode);
 }
 
 public int longestValidParentheses(String s) {
	   Stack<Integer> stack = new Stack<Integer>();
	    int max=0;
	    int left = -1;
	    for(int j=0;j<s.length();j++){
	        if(s.charAt(j)=='(') stack.push(j);            
	        else {
	            if (stack.isEmpty()) left=j;
	            else{
	                stack.pop();
	                if(stack.isEmpty()) max=Math.max(max,j-left);
	                else max=Math.max(max,j-stack.peek());
	               }
	            }
	        }
	    return max;
 }
 
 public static int lengthOfLongestSubstring(String s) {
     if (s.length()==0) return 0;
     HashMap<Character, Integer> map = new HashMap<Character, Integer>();
     int max=0;
     for (int i=0, j=0; i<s.length(); ++i){
         if (map.containsKey(s.charAt(i))){
             j = Math.max(j,map.get(s.charAt(i))+1);
         }
         map.put(s.charAt(i),i);
         max = Math.max(max,i-j+1);
     }
     return max;
 }
 
 public static int maxProfit(int[] prices) {
	    int maxPro = 0;
	    int minPrice = Integer.MAX_VALUE;
	    for(int i = 0; i < prices.length; i++){
	        minPrice = Math.min(minPrice, prices[i]);
	        maxPro = Math.max(maxPro, prices[i] - minPrice);
	    }
	    return maxPro;
 }
 
 public List<List<Integer>> combinationSum2(int[] candidates, int target) {
	   Arrays.sort(candidates);
     List<List<Integer>> result = new ArrayList<List<Integer>>();
     getResult2(result, new ArrayList<Integer>(), candidates, target, 0);
     return result;
 }
 
 private void getResult2(List<List<Integer>> result, List<Integer> cur, int[] candidates, int target, int start){
	   if(target > 0){
		   for(int i=start; i<candidates.length && target >= candidates[i]; i++){
			   cur.add(candidates[i]);
			   getResult2(result, cur, candidates, target - candidates[i], i+1);
			   cur.remove(cur.size()-1);
			   
			   // eliminate duplicates
		        while (i < candidates.length - 1 && candidates[i] == candidates[i+1]) ++i;
		   }
	   }else if(target == 0){
		   result.add(new ArrayList<Integer>(cur));
	   }
 }
 
 public int ladderLength(String start, String end, Set<String> dict) {
     if(start==null||end==null||dict==null||start.isEmpty()||end.isEmpty()||dict.isEmpty()) return 0;
     if(start==end) return 1;
     Set<String> current=new HashSet<String>();
     Set<String> past=new HashSet<String>();
     past.add(end);                      //ladder from the end to the start
     dict.add(start);                      //add start into dict, so that when start shows up in the laddered output, the loop can end
     int count=1;
     while(!past.isEmpty()){
         if(past.contains(start)) return count;
         current=new HashSet<String>();
         for(String s:past){
             current=convert(s,dict,current);
         }
         past=current;
         count++;
     }
     return 0;
 }
 private Set<String> convert(String s,Set<String> dict,Set<String> current){  //convert the words in the past set into next laddered set according to dictionary
     for(String word:dict){
         if(convertable(s,word)){
             current.add(word);
         }
     }
     for(String word:current){
         dict.remove(word);
     }
     return current;
 }

 private boolean convertable(String s1,String s2){
     if(s1==null||s2==null||s1.length()!=s2.length()||s1.equals(s2)) return false;
     int diff=0;
     for(int i=s1.length()-1;i>=0;i--){
         if(s1.charAt(i)!=s2.charAt(i)) diff++;
         if(diff>1) return false;
     }
     return true;
 }
 
 public boolean isMatch(String str, String pattern) {
     int s = 0, p = 0, match = 0, starIdx = -1;            
     while (s < str.length()){
         // advancing both pointers
         if (p < pattern.length()  && (pattern.charAt(p) == '?' || str.charAt(s) == pattern.charAt(p))){
             s++;
             p++;
         }
         // * found, only advancing pattern pointer
         else if (p < pattern.length() && pattern.charAt(p) == '*'){
             starIdx = p;
             match = s;
             p++;
         }
        // last pattern pointer was *, advancing string pointer
         else if (starIdx != -1){
             p = starIdx + 1;
             match++;
             s = match;
         }
        //current pattern pointer is not star, last patter pointer was not *
       //characters do not match
         else return false;
     }

     //check for remaining characters in pattern
     while (p < pattern.length() && pattern.charAt(p) == '*')
         p++;

     return p == pattern.length();
 }
 
 public static String multiply(String num1, String num2) {
	   // max (m + n) digits
	   int[] product = new int[num1.length() + num2.length()];
	   
	   // reverse for ease of calc
	   char[] num1Array = new char[num1.length()];
	   char[] num2Array = new char[num2.length()];
	   for(int i=num1.length()-1; i>=0; i--){
		   num1Array[num1.length()-i-1] = num1.charAt(i);
	   }
	   for(int i=num2.length()-1; i>=0; i--){
		   num2Array[num2.length()-i-1] = num2.charAt(i);
	   }
	   
	   // digit i * digit j contributes to digit i + j
	   for(int i=0; i<num1.length(); i++){
		   for(int j=0; j<num2.length(); j++){
			   product[i+j] += (num1Array[i] - '0')*(num2Array[j] - '0');
			   product[i+j+1] += product[i+j]/10;
			   product[i+j] %= 10;
		   }
	   }
	   
	   // remove leading 0; keep last 0 if all 0
	   int i = product.length - 1;
	   while(i>0 && 0 == product[i]){
		   i--;
	   }
	   
	   StringBuilder sb = new StringBuilder();
	   while(i>=0){
		   sb.append(product[i]);
		   i--;
	   }
	   return sb.toString();
 }
 
 public static List<List<Integer>> threeSum(int[] num) {
	   Arrays.sort(num);
	    List<List<Integer>> res = new ArrayList<List<Integer>>(); 
	    for (int i = 0; i < num.length-2; i++) {
	        if (i == 0 || (i > 0 && num[i] != num[i-1])) {
	            int lo = i+1, hi = num.length-1, sum = 0 - num[i];
	            while (lo < hi) {
	                if (num[lo] + num[hi] == sum) {
	                    res.add(Arrays.asList(num[i], num[lo], num[hi]));
	                    while (lo < hi && num[lo] == num[lo+1]) lo++;
	                    while (lo < hi && num[hi] == num[hi-1]) hi--;
	                    lo++; hi--;
	                } else if (num[lo] + num[hi] < sum) lo++;
	                else hi--;
	           }
	        }
	    }
	    return res;
 }
 
 public ListNode removeNthFromEnd(ListNode head, int n) {
	   ListNode fast=head;
     ListNode slow=head;
     while(n>0){
         fast=fast.next;
         n--;
     }

     if(fast==null) return slow.next;

     while(fast.next!=null){
         fast=fast.next;
         slow=slow.next;
     }

     slow.next=slow.next.next;
     return head;
 }
 
 public static int atoi(String str) {
	    if (str.isEmpty()) return 0;
	    int sign = 1, base = 0, i = 0;
	    while (str.charAt(i) == ' ')
	        i++;
	    if (str.charAt(i) == '-' || str.charAt(i) == '+')
	        sign = str.charAt(i++) == '-' ? -1 : 1;
	    while (i < str.length() && str.charAt(i) >= '0' && str.charAt(i) <= '9') {
	        if (base > Integer.MAX_VALUE / 10 || (base == Integer.MAX_VALUE / 10 && str.charAt(i) - '0' > 7)) {
	            return (sign == 1) ? Integer.MAX_VALUE : Integer.MIN_VALUE;
	        }
	        base = 10 * base + (str.charAt(i++) - '0');
	    }
	    return base * sign;
 }
 
 public int search(int[] nums, int target) {
	    int minIdx = findMinIdx(nums);
	    if (target == nums[minIdx]) return minIdx;
	    int m = nums.length;
	    int start = (target <= nums[m - 1]) ? minIdx : 0;
	    int end = (target > nums[m - 1]) ? minIdx - 1 : m - 1;

	    while (start <= end) {
	        int mid = start + (end - start) / 2;
	        if (nums[mid] == target) return mid;
	        else if (target > nums[mid]) start = mid + 1;
	        else end = mid - 1;
	    }
	    return -1;
	}

	private int findMinIdx(int[] nums) {
	    int start = 0, end = nums.length - 1;
	    while (start < end) {
	        int mid = start + (end -  start) / 2;
	        if (nums[mid] > nums[end]) start = mid + 1;
	        else end = mid;
	    }
	    return start;
	}
 
 public static int uniquePaths(int m, int n) {
     int N = n + m - 2;// how much steps we need to do
     int k = m - 1; // number of steps that need to go down
     double res = 1;
     // here we calculate the total possible path number 
     // Combination(N, k) = n! / (k!(n - k)!)
     // reduce the numerator and denominator and get
     // C = ( (n - k + 1) * (n - k + 2) * ... * n ) / k!
     for (int i = 1; i <= k; i++)
         res = res * (N - k + i) / i;
     return (int)res;
 }
 
 public static boolean isPalindrome(String s) {
	   if(s == null || s.equals("")) return true;
	   int left = 0;
	   int right = s.length() - 1;
	   while(left < right){
		   while(left < right && (!(s.charAt(left) >= 'a' && s.charAt(left) <= 'z')) &&
				   !(s.charAt(left) >= 'A' && s.charAt(left) <= 'Z') && 
				   !(s.charAt(left) >= '0' && s.charAt(left) <= '9')){
			   left++;
		   }
		   while(left < right && (!(s.charAt(right) >= 'a' && s.charAt(right) <= 'z')) &&
				   !(s.charAt(right) >= 'A' && s.charAt(right) <= 'Z') && 
				   !(s.charAt(right) >= '0' && s.charAt(right) <= '9')){
			   right--;
		   }
		   if(s.charAt(left) == s.charAt(right) || s.charAt(left) - s.charAt(right)
				   == 'A' - 'a' || s.charAt(left) - s.charAt(right)
				   == 'a' - 'A' ){
			   left++;
			   right--;
		   }
		   else{
			   return false;
		   }
	   }
     return true;
 }
 
 public static int removeDuplicates(int[] nums) {
	    int i = 0;
	    for (int n : nums)
	        if (i == 0 || n > nums[i-1])
	            nums[i++] = n;
	    return i;
 }
 
 public static int removeDuplicates2(int[] A) {
	   if(A.length < 3) return A.length;
		  int j = 0;
		  boolean doubled = false;
	      for(int i=1; i<A.length; i++){
	    	 if(A[j] != A[i]){
	    		  A[++j] = A[i];
	    		  doubled = false;
	    	  }else if(!doubled){
	    		  A[++j] = A[i];
	    		  doubled = true;
	    	  }
	      }
	      return ++j;
 }
 
 public int findMin2(List<Integer> num) {
     for(int i=0; i<num.size()-1; i++){
			if(num.get(i)>num.get(i+1))
				return num.get(i+1);
		}
		return num.get(0);
 }
 
 public boolean isSameTree(TreeNode p, TreeNode q) {
	   if(p == null && q == null) return true;
	   if((p == null && q != null) || (p != null && q == null)) return false;
	   if(p.val != q.val) return false;
	   return (isSameTree(p.left, q.left) && isSameTree(p.right, q.right));
 }
 
 private int factorial(int n){
	   int result = 1;
	   for(int i=n; i>0; i--){
		   result *= i;
	   }
	   return result;
 }
 
 public String getPermutation(int n, int k) {
	   if(n == 1) return "1";
	   
	   List<Integer> list = new ArrayList<Integer>();
	   for(int i=1; i<=n; i++){
		   list.add(i);
	   }
	   
	   StringBuilder sb = new StringBuilder();
	   int divisor = factorial(n-1);
	   int consult = 0;
	   int temp = n-1;
	   for(int i=0; i<n-1; i++){
		   consult = (int)Math.ceil((double)k/divisor);
		   sb.append(list.get(consult-1));
		   list.remove(consult-1);
		   k -= (consult-1)*divisor;
		   divisor /= temp;
		   temp--;
	   }
	   sb.append(list.get(0));
     return sb.toString();
 }
 
 public int numTrees(int n) {
	    int [] G = new int[n+1];
	    G[0] = G[1] = 1;

	    for(int i=2; i<=n; ++i) {
	        for(int j=1; j<=i; ++j) {
	            G[i] += G[j-1] * G[i-j];
	        }
	    }

	    return G[n];
	}
 
 public int trailingZeroes(int n) {
	   if(n<5) return 0;
	   int zeros = 0;
	   double div = 5;
	   int power = 1;
	   while(div <= n){
		   zeros += n/div;
		   power++;
		   div = Math.pow(5, power);
	   }
	   return zeros;
 }
 
 public boolean isNumber(String s) {
	   //trim whitespace front and end
     s = s.trim();

     if (s.length()<1) { //if no digits
         return false; 
     }


     if (s.charAt(s.length()-1) == 'f' || s.charAt(s.length()-1) == 'D'){ //last digit check
         return false;   //if last digit is f or D, fail
     }

     try{
     Double.parseDouble(s);   //parse as a double
     return true;

     } catch (Exception e1){
         return false;
     }
 }
 
 public int compareVersion(String version1, String version2) {
	   String[] str1 = version1.split("\\.");
	   String[] str2 = version2.split("\\.");
	   int length = Math.min(str1.length, str2.length);
	   int i1 = 0;
	   int i2 = 0;
	   for(int i=0; i<length; i++){
		   i1 = Integer.parseInt(str1[i]);
		   i2 = Integer.parseInt(str2[i]);
		   if(i1>i2) return 1;
		   else if(i1<i2) return -1;
	   }
	   if(str1.length > str2.length){
		   for(int i= length; i<str1.length; i++){
			   i1 = Integer.parseInt(str1[i]);
			   if(i1 != 0) return 1;
		   }
	   }else if(str1.length < str2.length) {
		   for(int i= length; i<str2.length; i++){
			   i2 = Integer.parseInt(str2[i]);
			   if(i2 != 0) return -1;
		   }
	   }
	   return 0;
 }
 
 public int countBitDiff(int m, int n){
		// 请在此添加代码
		int dif = m ^ n;
		int i = 1;
		int count = 0;
		for(int j = 0; j<32; j++){
			if((dif & i) != 0){
				count++;
			}
			i <<= 1;
		}
		return count;
	}
 
 public int sumNumbers(TreeNode root) {
	   if(root == null) return 0;
     int sum = 0;
     StringBuffer sb = new StringBuffer();
     sb.append(root.val);
     TreeNode cur = root;
     Stack<Object> stack = new Stack<Object>();
     stack.push(root);
     stack.push(Integer.parseInt(sb.toString()));
     while(cur.left != null){
		   cur = cur.left;
		   sb.append(cur.val);
		   stack.push(cur);
		   stack.push(Integer.parseInt(sb.toString()));
	   }
	   while(!stack.isEmpty()){
		   sb = new StringBuffer(String.valueOf((Integer)stack.pop()));
		   cur = (TreeNode)stack.pop();
		   if(cur.left ==null && cur.right == null){
			   	sum += Integer.parseInt(sb.toString());
			   	continue;
		   }
		   if(cur.right == null) continue;
		   else if(cur.right != null){
			   cur = cur.right;
			   sb.append(cur.val);
			   stack.push(cur);
			   stack.push(Integer.parseInt(sb.toString()));
		   }
		   while(cur.left != null){
			   cur = cur.left;
			   sb.append(cur.val);
			   stack.push(cur);
			   stack.push(Integer.parseInt(sb.toString()));
		   }
	   }
	   return sum;
 }
 
 public void rotate(int[][] matrix) {
	   int n = matrix.length;
	   int level = n/2;
     for(int i=0; i<level; i++){
  	   for(int j=0; j<n-1-i*2; j++){
  		   int temp = matrix[i][i+j];
  		   matrix[i][i+j]= matrix[n-1-i-j][i];
  		   matrix[n-1-i-j][i] = matrix[n-1-i][n-1-i-j];
  		   matrix[n-1-i][n-1-i-j] = matrix[i+j][n-1-i];
  		   matrix[i+j][n-1-i] = temp;
  	   }
     }
 }
 
 public int lengthOfLastWord(String s) {
	   String str = s.trim();
	   String lastWord = str.substring(str.lastIndexOf(" ") + 1);
     return lastWord.length();
 }
 
 public int maxProfit2(int[] prices) {
	   int in = 0;
	   int out = 0;
	   boolean isBought = false;
	   int totalProfit = 0;
	   for(int i=0; i<prices.length-1; i++){
		   if(prices[i] < prices[i+1]){
			   if(isBought) continue;
			   else {
				   in = i;
				   isBought = true;
				   continue;
			   }
		   }else if(prices[i] > prices[i+1]){
			   if(isBought) {
				   out = i;
				   totalProfit += prices[out] - prices[in];
				   isBought = false;
				   continue;
			   }
			   else continue;
		   }else{
			   continue;
		   }
	   }
	   if(isBought){
		   totalProfit += prices[prices.length-1] - prices[in];
	   }
     return totalProfit;
 }
 
 public String longestCommonPrefix(String[] strs) {
     if(strs.length==0) return "";
     StringBuilder lcp=new StringBuilder();
     for(int i=0;i<strs[0].length();i++){
         char c=strs[0].charAt(i);
         for(String s:strs){
             if(s.length()<i+1||c!=s.charAt(i)) return lcp.toString();
         }
         lcp.append(c);
     }
     return lcp.toString();
 }
 
 List<List<String>> result = new ArrayList<List<String>>();
 public List<List<String>> partition(String s) {
     helper(s, new ArrayList<String>());
     return result;
 }  
 
 private void helper(String s, List<String> cur){                 //DFS every combinations
     if(s.length() == 0){result.add(cur); return;}        
     for(int i = 1; i <= s.length(); i++){
         String sub = s.substring(0,i);
         if(isPalindromeString(sub)){
             List<String> newList = new ArrayList<String>(cur);
             newList.add(sub);
             helper(s.substring(i,s.length()), newList);
         }
         else continue;                                    //not palindrome, ignore it
     }        
 }        
 
  private boolean isPalindromeString(String s){
  	for(int i=0; i<s.length()/2; i++){
  		if(s.charAt(i) != s.charAt(s.length()-1-i))
  			return false;
  	}
  	return true;
  }
  
  public ListNode reverseList(ListNode head) {
		ListNode newHead = null;
		while(head != null){
			ListNode next = head.next;
			head.next = newHead;
			newHead = head;
			head = next;
		}
		return newHead;
	}
  
  public int[] twoSum(int[] numbers, int target) {
      int[] result = new int[2];
      Map<Integer, Integer> map = new HashMap<>();
      for (int i = 0; i < numbers.length; i++) {
          if (map.containsKey(target - numbers[i])) {
              result[1] = i + 1;
              result[0] = map.get(target - numbers[i]);
              return result;
          }
          map.put(numbers[i], i + 1);
      }
      return result;
  }
  
  public int removeElement(int[] A, int elem) {
  	int m = 0;    
	   for(int i = 0; i < A.length; i++){
	       if(A[i] != elem){
	           A[m] = A[i];
	           m++;
	       }
	   }
  	return m;
  }
  
  public double findMedianSortedArrays(int A[], int B[]) {
      int length=A.length+B.length;
      if(length%2==1)
          return findMedianSortedArrays(A, 0, A.length, B, 0, B.length, length/2+1);
      else
          return (findMedianSortedArrays(A, 0, A.length, B, 0, B.length, length/2)+findMedianSortedArrays(A, 0, A.length, B, 0, B.length, length/2+1))/2;
  }

  public double findMedianSortedArrays(int A[],int aStart,int aEnd, int B[],int bStart,int bEnd,int length){
      if((aEnd-aStart)>(bEnd-bStart))
          return findMedianSortedArrays(B,bStart,bEnd,A,aStart,aEnd,length);
      if(aEnd-aStart==0)
          return B[bStart+length-1];
      if(length==1)
          return A[aStart]<B[bStart]?A[aStart]:B[bStart];
      int pa=(aEnd-aStart)<(length/2)?(aEnd-aStart):(length/2);
      int pb=length-pa;
      if(A[aStart+pa-1]==B[bStart+pb-1])
          return A[aStart+pa-1];
      else if(A[aStart+pa-1]<B[bStart+pb-1])
          return findMedianSortedArrays(A, aStart+pa, aEnd, B, bStart, bEnd, length-pa);
      else
          return findMedianSortedArrays(A, aStart, aEnd, B, bStart+pb, bEnd, length-pb);
  }

  public String convert(String s, int nRows) {
      char[] c = s.toCharArray();
      int len = c.length;
      StringBuilder[] sb = new StringBuilder[nRows];
      for (int i = 0; i < sb.length; i++) sb[i] = new StringBuilder();

      int i = 0;
      while (i < len) {
          for (int idx = 0; idx < nRows && i < len; idx++) // vertically down
              sb[idx].append(c[i++]);
          for (int idx = nRows-2; idx >= 1 && i < len; idx--) // obliquely up
              sb[idx].append(c[i++]);
      }
      for (int idx = 1; idx < sb.length; idx++)
          sb[0].append(sb[idx]);
      return sb[0].toString();
  }
  
  public boolean isPalindrome(int x) {
      if (x<0 || (x!=0 && x%10==0)) return false;
      int rev = 0;
      while (x>rev){
          rev = rev*10 + x%10;
          x = x/10;
      }
      return (x==rev || x==rev/10);
  }
  
  public boolean isMatch2(String s, String p) {
      if (p.isEmpty()) {
      return s.isEmpty();
  }

  if (p.length() == 1 || p.charAt(1) != '*') {
      if (s.isEmpty() || (p.charAt(0) != '.' && p.charAt(0) != s.charAt(0))) {
          return false;
      } else {
          return isMatch(s.substring(1), p.substring(1));
      }
  }

  //P.length() >=2 && p.charAt(1) == '*'
  while (!s.isEmpty() && (s.charAt(0) == p.charAt(0) || p.charAt(0) == '.')) {  
      if (isMatch(s, p.substring(2))) { 
          return true;                     
      }                                    
      s = s.substring(1);
  }

  return isMatch(s, p.substring(2));
  }
  
  public int maxArea(int[] height) {
  	 int maxWater=0, left=0, right=height.length-1;
       while(left<right) {
           maxWater = Math.max(maxWater,(right-left)*Math.min(height[left], height[right]));
           if(height[left]<height[right]) left++;
           else right--;
       }
       return maxWater;
  }
  
  public String intToRoman(int num) {
      String M[] = {"", "M", "MM", "MMM"};
      String C[] = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
      String X[] = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
      String I[] = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
      return M[num/1000] + C[(num%1000)/100] + X[(num%100)/10] + I[num%10];
  }
  
  public int romanToInt(String s) {
      int sum=0;
      if(s.indexOf("IV")!=-1){sum-=2;}
      if(s.indexOf("IX")!=-1){sum-=2;}
      if(s.indexOf("XL")!=-1){sum-=20;}
      if(s.indexOf("XC")!=-1){sum-=20;}
      if(s.indexOf("CD")!=-1){sum-=200;}
      if(s.indexOf("CM")!=-1){sum-=200;}

      char c[]=s.toCharArray();
      
     for(int count=0;count<s.length();count++){
         if(c[count]=='M') sum+=1000;
         if(c[count]=='D') sum+=500;
         if(c[count]=='C') sum+=100;
         if(c[count]=='L') sum+=50;
         if(c[count]=='X') sum+=10;
         if(c[count]=='V') sum+=5;
         if(c[count]=='I') sum+=1;
     }

     return sum;
  }
  
  public int threeSumClosest(int[] nums, int target) {
  	Arrays.sort(nums);
      int result = nums[0] + nums[1] + nums[nums.length-1];
      int abs = Math.abs(target-result);
      for(int i = 0; i<nums.length-2; i++){
      	int lo = i+1, hi = nums.length-1;
      	while(lo < hi){
      		int tempResult = nums[i] + nums[lo] + nums[hi];
      		int tempAbs = Math.abs(target-tempResult);
      		if(tempAbs < abs){
      			result = tempResult;
      			abs = tempAbs;
      		}
      		if(tempResult < target) lo++;
      		else if(tempResult > target) hi--;
      		else if(tempResult == target) return result;
      	}
       }
      return result;
  }
  
  public List<String> letterCombinations(String digits) {
      LinkedList<String> ans = new LinkedList<String>();
      if(digits == null || digits.isEmpty()) return ans;
      String[] mapping = new String[] {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
      ans.add("");
      for(int i =0; i<digits.length();i++){
          int x = Character.getNumericValue(digits.charAt(i));
          while(ans.peek().length()==i){
              String t = ans.remove();
              for(char s : mapping[x].toCharArray())
                  ans.add(t+s);
          }
      }
      return ans;
  }
  
  public List<List<Integer>> fourSum(int[] nums, int target) {
      List<List<Integer>> result = new ArrayList<List<Integer>>();
      if(nums.length < 4) return result;
      Arrays.sort(nums);
      int i = 0;
      while(i<nums.length-3){
      	while(i>0 && nums[i] == nums[i-1] && i<nums.length-3) i++;
      	int j = i+1;
      	while(j<nums.length-2){
      		while(j>i+1 && nums[j] == nums[j-1] && j<nums.length-2) j++;
      		int lo = j+1, hi = nums.length-1, sum = target - nums[i] - nums[j];
      		 while (lo < hi) {
                   if (nums[lo] + nums[hi] == sum) {
                       result.add(Arrays.asList(nums[i],nums[j],nums[lo], nums[hi]));
                       while (lo < hi && nums[lo] == nums[lo+1]) lo++;
                       while (lo < hi && nums[hi] == nums[hi-1]) hi--;
                       lo++; hi--;
                   } else if (nums[lo] + nums[hi] < sum) lo++;
                   else hi--;
              }
      		j++;
      	}
      	i++;
      }
      return result;
  }
  
  public List<String> generateParenthesis(int n) {
      List<String> ans = new ArrayList<String>();
      if (n == 0)
          return ans;
      addPair(ans, "", n, 0);
      return ans;
  }

  private void addPair(List<String> ans, String s, int n, int m) {
      if (n == 0 && m == 0) {
          ans.add(s);
          return;
      }
      if (m > 0) {
          addPair(ans, s + ")", n, m-1);
      }
      if (n > 0) {
          addPair(ans, s + "(", n-1, m+1);
      }
  }
  
  public ListNode mergeKLists(ListNode[] lists) {
  	if(lists.length == 0) return null;
      Queue<ListNode> queue = new PriorityQueue<ListNode>(lists.length,new Comparator<ListNode>(){
          @Override
          public int compare(ListNode l1, ListNode l2) {
              if (l1.val < l2.val)
                    return -1;
              else if (l1.val == l2.val)
                    return 0;
              else
                    return 1;
        }
      });
      ListNode head = new ListNode(0);
      ListNode p = head;
      for(ListNode node : lists) {
          if(node != null)
              queue.offer(node);
      }
      while(!queue.isEmpty()) {
          ListNode node = queue.poll();
          p.next = node;
          if(node.next != null)
              queue.offer(node.next);
          p = p.next;
      }
      return head.next;
  }
  
  public ListNode swapPairs(ListNode head) {
  	if(head==null)
          return null;
      ListNode dummy = new ListNode(-1);
      dummy.next = head;
      ListNode pre = dummy;
      ListNode cur = head;
      ListNode next = null;
      while (cur!=null && cur.next != null) {
        next = cur.next;
        
        pre.next = next;
        cur.next = next.next;
        next.next = cur;
        
        pre= cur;
        cur= cur.next;
      }
    return dummy.next;
  }
  
  public ListNode reverseKGroup(ListNode head, int k) {
      ListNode curr = head;
      int count = 0;
      while (curr != null && count != k) { // find the k+1 node
          curr = curr.next;
          count++;
      }
      if (count == k) { // if k+1 node is found
          curr = reverseKGroup(curr, k); // reverse list with k+1 node as head
          // head - head-pointer to direct part, 
          // curr - head-pointer to reversed part;
          while (count-- > 0) { // reverse current k-group: 
              ListNode tmp = head.next; // tmp - next head in direct part
              head.next = curr; // preappending "direct" head to the reversed list 
              curr = head; // move head of reversed part to a new node
              head = tmp; // move "direct" head to the next node in direct part
          }
          head = curr;
      }
      return head;
  }
  
  public int strStr(String haystack, String needle) {
  	for (int i = 0; ; i++) {
  	    for (int j = 0; ; j++) {
  	      if (j == needle.length()) return i;
  	      if (i + j == haystack.length()) return -1;
  	      if (needle.charAt(j) != haystack.charAt(i + j)) break;
  	    }
  	  }
  }
  
  public List<Integer> findSubstring(String s, String[] words) {
  	List<Integer> result = new ArrayList<Integer>();
      int size = words[0].length();
      if (words.length == 0 || size==0 || words.length*size > s.length()) 
          return result;
      Map<String, Integer> hist = new HashMap<String, Integer>();
      for (String w : words) {
          hist.put(w, !hist.containsKey(w) ? 1 : hist.get(w)+1);
      }
      for (int i = 0; i+size*words.length <= s.length(); i++) {
          if (hist.containsKey(s.substring(i, i+size))) {
              Map<String, Integer> currHist = new HashMap<String, Integer>();
              for (int j = 0; j < words.length; j++) {
                  String word = s.substring(i+j*size, i+(j+1)*size);
                  if(!hist.containsKey(word)) break;
	                    currHist.put(word, !currHist.containsKey(word) ? 
	                            1 : currHist.get(word)+1);
              }
              if (currHist.equals(hist)) result.add(i);
          }
      }
      return result;
  }
  
  public int[] searchRange(int[] nums, int target) {
      int[] result = new int[2];
      Arrays.fill(result, -1);
      int lo = 0;
      int hi = nums.length - 1;
      // Search for the left one
      while(lo < hi){
      	int mid = (lo + hi)/2;
      	if(nums[mid] < target) lo = mid + 1;
      	else hi = mid;
      }
      if(nums[lo] != target) return result;
      else result[0] = lo;
      hi = nums.length - 1; // We don't have to set lo to 0 the second time.
      // Search for the right one
      while(lo < hi){
      	int mid = (lo + hi)/2 + 1;   // Make mid biased to the right
      	if(nums[mid] > target) hi = mid - 1;
      	else lo = mid;			  // So that this won't make the search range stuck.
      }
      result[1] = hi;
      return result;
  }
  
  public int searchInsert(int[] nums, int target) {
  	 int low = 0, high = nums.length-1;
       while(low<=high){
           int mid = (low+high)/2;
           if(nums[mid] == target) return mid;
           else if(nums[mid] > target) high = mid-1;
           else low = mid+1;
       }
       return low;
  }
  
  public void solveSudoku(char[][] board) {
      if(board == null || board.length == 0)
          return;
      solve(board);
  }

  public boolean solve(char[][] board){
      for(int i = 0; i < board.length; i++){
          for(int j = 0; j < board[0].length; j++){
              if(board[i][j] == '.'){
                  for(char c = '1'; c <= '9'; c++){//trial. Try 1 through 9 for each cell
                      if(isValid(board, i, j, c)){
                          board[i][j] = c; //Put c for this cell

                          if(solve(board))
                              return true; //If it's the solution return true
                          else
                              board[i][j] = '.'; //Otherwise go back
                      }
                  }
                  return false;
              }
          }
      }
      return true;
  }

  public boolean isValid(char[][] board, int i, int j, char c){
      //Check colum
      for(int row = 0; row < 9; row++)
          if(board[row][j] == c)
              return false;

      //Check row
      for(int col = 0; col < 9; col++)
          if(board[i][col] == c)
              return false;

      //Check 3 x 3 block
      for(int row = (i / 3) * 3; row < (i / 3) * 3 + 3; row++)
          for(int col = (j / 3) * 3; col < (j / 3) * 3 + 3; col++)
              if(board[row][col] == c)
                  return false;
      return true;
  }
  
  public int firstMissingPositive2(int[] nums) {
  	if(nums == null || nums.length == 0) return 1;
  	for(int i=0; i<nums.length; i++){
  		if(nums[i]<=0) nums[i] = nums.length + 2;
  	}
  	for(int i=0; i<nums.length; i++){
  		if(Math.abs(nums[i]) < nums.length + 1){
  			int cur = Math.abs(nums[i]) - 1;
  			nums[cur] = -Math.abs(nums[cur]);
  		}
  	}
  	for(int i=0; i<nums.length; i++){
  		if(nums[i] > 0) return i+1;
  	}
      return nums.length+1;
  }
  
  public int firstMissingPositive(int[] A) {
      int i = 0;
      while(i < A.length){
          if(A[i] == i+1 || A[i] <= 0 || A[i] > A.length) i++;
          else if(A[A[i]-1] != A[i]) swap(A, i, A[i]-1);
          else i++;
      }
      i = 0;
      while(i < A.length && A[i] == i+1) i++;
      return i+1;
  }

  private void swap(int[] A, int i, int j){
      int temp = A[i];
      A[i] = A[j];
      A[j] = temp;
  }
  
  public int trap(int[] height) {
      int left=0; int right=height.length-1;
      int res=0;
      int maxleft=0, maxright=0;
      while(left<=right){
          if(height[left]<=height[right]){
              if(height[left]>=maxleft) maxleft=height[left];
              else res+=maxleft-height[left];
              left++;
          }
          else{
              if(height[right]>=maxright) maxright= height[right];
              else res+=maxright-height[right];
              right--;
          }
      }
      return res;
  }
  
  public int jump(int[] nums) {
      int result = 0;
      int curMax = 0;
      int curRch = 0;
      for(int i=0; i<nums.length; i++){
      	if(curRch < i){
      		result ++;
      		curRch = curMax;
      	}
      	curMax = Math.max(curMax, i+nums[i]);
      }
  	return result;
  }
  
  public boolean canJump(int[] nums) {
  	int curMax =0;
  	for(int i=0; i<nums.length; i++){
  		if(curMax < i) return false;
  		curMax = Math.max(curMax, i+nums[i]);
  	}
    return true;  
  }
  
  public List<List<Integer>> permuteUnique(int[] num) {
      List<List<Integer>> returnList = new ArrayList<List<Integer>>();
      returnList.add(new ArrayList<Integer>());

	 	for (int i = 0; i < num.length; i++) {
	 		Set<ArrayList<Integer>> currentSet = new HashSet<ArrayList<Integer>>();
	 		for (List<Integer> l : returnList) {
	 			for (int j = 0; j < l.size() + 1; j++) {
	 				l.add(j, num[i]);
	 				ArrayList<Integer> T = new ArrayList<Integer>(l);
	 				l.remove(j);
	 				currentSet.add(T);
	 			}
	 		}
	 		returnList = new ArrayList<List<Integer>>(currentSet);
	 	}
	 	return returnList;
   }
  
  public List<List<String>> groupAnagrams(String[] strs) {
      List<List<String>> lists = new ArrayList<List<String>>();
      if (strs.length == 0) return lists;
      Arrays.sort(strs);
      HashMap<String, List<String>> map = new HashMap<String, List<String>>();
      for (String str : strs) {
          char[] charArray = str.toCharArray();
          Arrays.sort(charArray);
          String sorted = new String(charArray);
          if (map.containsKey(sorted)) {
              List<String> list = map.get(sorted);
              list.add(str);
          }
          else {
              List<String> list = new ArrayList<String>();
              list.add(str);
              map.put(sorted, list);
          }
      }

      for (List<String> list : map.values()) {
          lists.add(list);
      }
      return lists;
  }
  
  public double myPow(double x, int n) {
      if(n==0) return 1;
      if(n==Integer.MIN_VALUE) return 1/(myPow(x,Integer.MAX_VALUE)*x);
      if(n < 0) return 1/myPow(x,Math.abs(n));
      if(x==1) return 1;
      else{
      	double result = myPow(x, n>>1);
      	result *= result;
      	if((n & 1) != 0){
      		result *= x;
      	}
      	return result;
      }
  }
  
  public List<String[]> solveNQueens(int n) {
  	ArrayList<String[]> ret = new ArrayList<String[]>();  
      int[] queenList = new int[n];  
      placeQueen(queenList, 0, n, ret);  
      return ret;
  }
  
  // 递归回溯8皇后，关键记录下到达了哪一行了  
  public void placeQueen(int[] queenList, int row, int n, ArrayList<String[]> ret){  
      // Base Case, 已经完成任务了  
      if(row == n){  
          StringBuilder[] sol = new StringBuilder[n];  
             
          // 对数组内每一个对象都要new出其对象  
          for(int i=0; i<n; i++){  
              sol[i] = new StringBuilder();  
              for(int j=0; j<n; j++){  
                  sol[i].append('.');  
              }  
          }  
          // 在相应的地方放置queen  
          for(int i=0; i<n; i++){  
              sol[i].setCharAt(queenList[i], 'Q');  
          }  
          String[] ss = new String[n];  
          for (int i=0; i<n; i++) {  
              ss[i] = sol[i].toString();  
          }  
          ret.add(ss);  
          return;  
      }  
         
      // 开始这一行的查找  
      // 遍历第row行的所有列，测试哪一个位置是安全的  
      for(int col=0; col<n; col++){  
          if(isSafe(queenList, row, col)){  
              queenList[row] = col;  
              placeQueen(queenList, row+1, n, ret);  
          }  
      }  
  }
  
  // 判断是否坐标(row,col)的位置是安全的（检查行，列，正反对角线）  
  // queenList里面存放行，列坐标pair，即queenList[row] = col  
  public boolean isSafe(int[] queenList, int row, int col){  
      for(int preRow=0; preRow<row; preRow++){  
          int preCol = queenList[preRow];  
          if(preRow == row){      // 理论上不必检查，因为preRow是总是小于row的  
              return false;  
          }  
          if(preCol == col){          // 检查是否在同一列  
              return false;  
          }  
          if(row-preRow == col-preCol){       // 反对角线  
              return false;  
          }  
          if(row-preRow == preCol-col){       // 正对角线  
              return false;  
          }  
      }  
      return true;  
  }
  
  int nQueensCount = 0;
  public int totalNQueens(int n) {
      int[] queenList = new int[n];  
      placeQueen(queenList, 0, n);
      return nQueensCount;
  }
  
// 递归回溯8皇后，关键记录下到达了哪一行了  
  public void placeQueen(int[] queenList, int row, int n){  
      // Base Case, 已经完成任务了  
      if(row == n){  
      	nQueensCount++;
          return;  
      }  
         
      // 开始这一行的查找  
      // 遍历第row行的所有列，测试哪一个位置是安全的  
      for(int col=0; col<n; col++){  
          if(isSafe(queenList, row, col)){  
              queenList[row] = col;  
              placeQueen(queenList, row+1, n);  
          }  
      }  
  }
  
  public int maxSubArray(int[] nums) {
      if(nums == null || nums.length == 0) return 0;
      int result = 0;
      int curSum = 0;
      int numMax = nums[0];
      for(int i=0; i<nums.length; i++){
      	numMax = Math.max(numMax, nums[i]);
      	int temp = curSum + nums[i];
      	if(temp<=0){
      		curSum = 0;
      	}else{
      		curSum = temp;
      	}
      	result = Math.max(curSum, result);
      }
      if(numMax<0) return numMax;
  	return result;
  	
/*    	if (nums.length == 0){
          return 0;
      }
      int prev = nums[0];
      int cur = nums[0];
      int max = nums[0];
      for (int i = 1; i < nums.length; i++){
          if (prev > 0){
              cur = prev + nums[i];
          }else{
              cur = nums[i];
          }
          max = Math.max(max, cur);
          prev = cur; 
      }
      return max;*/
  }
  
  public ListNode deleteDuplicates2(ListNode head) {
  	ListNode pre_head = new ListNode(0);
  	pre_head.next = head;
  	ListNode pre = pre_head;
  	ListNode cur = head;
      boolean dup = false;
      while(cur != null){
          ListNode aft = cur.next;
          if((aft != null && aft.val != cur.val) || aft == null){
              if(!dup){
                  pre = cur;
                  cur = aft;
                  continue;
              }else{
                  if(pre == pre_head) pre_head.next = aft;
                  pre.next = aft;
                  cur = aft;
                  dup = false;
                  continue;
              }
          }else{
              cur = aft;
              dup = true;
              continue;
          }
      }
      return pre_head.next;
  }
  
  public List<List<Integer>> subsetsWithDup(int[] nums) {
  	Arrays.sort(nums);
  	Set<List<Integer>> result = new HashSet<List<Integer>>();
  	List<Integer> temp = new ArrayList<Integer>();
  	subsetsWithDupHelper(nums, result, temp, 0);
  	return new ArrayList<List<Integer>>(result);
  }
  
  private void subsetsWithDupHelper(int[]nums, Set<List<Integer>> result, List<Integer> temp, int index){
  	if(index == nums.length) {
  		result.add(temp);
  		return;
  	}
  	subsetsWithDupHelper(nums, result, new ArrayList<Integer>(temp), index+1);
  	temp.add(nums[index]);
  	subsetsWithDupHelper(nums, result, temp, index+1);
  }
  
  public ListNode reverseBetween(ListNode head, int m, int n) {
      if(head == null) return null;
      ListNode dummy = new ListNode(0); // create a dummy node to mark the head of this list
      dummy.next = head;
      ListNode pre = dummy; // make a pointer pre as a marker for the node before reversing
      for(int i = 0; i<m-1; i++) pre = pre.next;

      ListNode start = pre.next; // a pointer to the beginning of a sub-list that will be reversed
      ListNode then = start.next; // a pointer to a node that will be reversed

      // 1 - 2 -3 - 4 - 5 ; m=2; n =4 ---> pre = 1, start = 2, then = 3
      // dummy-> 1 -> 2 -> 3 -> 4 -> 5

      for(int i=0; i<n-m; i++)
      {
          start.next = then.next;
          then.next = pre.next;
          pre.next = then;
          then = start.next;
      }

      // first reversing : dummy->1 - 3 - 2 - 4 - 5; pre = 1, start = 2, then = 4
      // second reversing: dummy->1 - 4 - 3 - 2 - 5; pre = 1, start = 2, then = 5 (finish)

      return dummy.next;
  }
  
  public List<String> restoreIpAddresses(String s) {
      List<String> res = new ArrayList<String>();
      int len = s.length();
      for(int i = 1; i<4 && i<len-2; i++){
          for(int j = i+1; j<i+4 && j<len-1; j++){
              for(int k = j+1; k<j+4 && k<len; k++){
                  String s1 = s.substring(0,i), s2 = s.substring(i,j), s3 = s.substring(j,k), s4 = s.substring(k,len);
                  if(isValidIP(s1) && isValidIP(s2) && isValidIP(s3) && isValidIP(s4)){
                      res.add(s1+"."+s2+"."+s3+"."+s4);
                  }
              }
          }
      }
      return res;
  }
  
  private boolean isValidIP(String s){
      if(s.length()>3 || s.length()==0 || (s.charAt(0)=='0' && s.length()>1) || Integer.parseInt(s)>255)
          return false;
      return true;
  }
  
  public List<Integer> inorderTraversal(TreeNode root) {
  	 List<Integer> list = new ArrayList<Integer>();
	    Stack<TreeNode> stack = new Stack<TreeNode>();
	    TreeNode cur = root;
	    while(cur!=null || !stack.isEmpty()){
	        if(cur!=null){
	            stack.push(cur);
	            cur = cur.left;
	        }else{
		        cur = stack.pop();
		        list.add(cur.val);
		        cur = cur.right;
	        }
	    }
	    return list;  
  }
  
	 public List<Integer> preorderTraversal(TreeNode node) {
		    List<Integer> list = new ArrayList<Integer>();
		    Stack<TreeNode> stack = new Stack<TreeNode>();
		    while(node != null || !stack.empty()){
		    	if(node != null){
		    		list.add(node.val);
		    		stack.push(node.right);
		    		node = node.left;
		    	}else{
		    		node = stack.pop();
		    	}
		    }
		    return list;
	 }
	 
	  public List<Integer> postorderTraversal(TreeNode node) {
		    List<Integer> list = new ArrayList<Integer>();
		    Stack<TreeNode> stack = new Stack<TreeNode>();
		    while(node != null || !stack.empty()){
		    	if(node != null){
		    		list.add(node.val);
		    		stack.push(node.left);
		    		node = node.right;
		    	}else{
		    		node = stack.pop();
		    	}
		    }
		    Collections.reverse(list);
		    return list;
	  }
  
  public List<TreeNode> generateTrees(int n) {
      return genTreeList(1,n);
  }
  
  private List<TreeNode> genTreeList (int start, int end) {
      List<TreeNode> list = new ArrayList<TreeNode>(); 
      if (start > end) {
          list.add(null);
      }
      for(int idx = start; idx <= end; idx++) {
          List<TreeNode> leftList = genTreeList(start, idx - 1);
          List<TreeNode> rightList = genTreeList(idx + 1, end);
          for (TreeNode left : leftList) {
              for(TreeNode right: rightList) {
                  TreeNode root = new TreeNode(idx);
                  root.left = left;
                  root.right = right;
                  list.add(root);
              }
          }
      }
      return list;
  }
  
  public boolean isInterleave(String s1, String s2, String s3) {
  	if(s1.length() + s2.length() != s3.length()) return false;
  	return isInterleaveHelper(s1, s2, s3, 0, 0, 0);
  }
  
  private boolean isInterleaveHelper(String s1, String s2, String s3, int i1, int i2, int i3){
  	if(i1 == s1.length()) return s2.substring(i2).equals(s3.substring(i3));
  	if(i2 == s2.length()) return s1.substring(i1).equals(s3.substring(i3));
  	
  	if(s1.charAt(i1) != s3.charAt(i3) && s2.charAt(i2) != s3.charAt(i3)) return false;
  	else if(s1.charAt(i1) == s3.charAt(i3) && s2.charAt(i2) != s3.charAt(i3)) 
  		return isInterleaveHelper(s1, s2, s3, i1+1, i2, i3+1);
  	else if(s1.charAt(i1) != s3.charAt(i3) && s2.charAt(i2) == s3.charAt(i3)) 
  		return isInterleaveHelper(s1, s2, s3, i1, i2+1, i3+1);
  	else return isInterleaveHelper(s1, s2, s3, i1+1, i2, i3+1) || isInterleaveHelper(s1, s2, s3, i1, i2+1, i3+1);
  }

  public boolean isValidBST(TreeNode root) {
      if(root == null) return true;
  	if(!isValidBSTHelper(root.left, root.val, true))
  		return false;
  	if(!isValidBSTHelper(root.right, root.val, false))
  		return false;
      if(!isValidBST(root.left) || !isValidBST(root.right)) 
      	return false;
      return true;
  }
  
  private boolean isValidBSTHelper(TreeNode root, int val, boolean testLess) {
  	if(root == null) return true;
  	if(testLess){
      	if(root.val >= val) return false;
      }else{
      	if(root.val <= val) return false;
      }
	    return isValidBSTHelper(root.left, val, testLess) && isValidBSTHelper(root.right, val, testLess);  
 }
  
  public void moveZeroes(int[] nums) {
      int pZero = 0, pNonZero = 0;
      while(pZero < nums.length && nums[pZero] != 0) pZero++;	//第一个0
      pNonZero = pZero + 1;
      while(pNonZero < nums.length && nums[pNonZero] == 0) pNonZero++; //0后面的第一个非0
      while(pZero < nums.length && pNonZero < nums.length){
      	nums[pZero] = nums[pNonZero];
      	nums[pNonZero] = 0;
      	pZero++;
      	while(pZero < nums.length && nums[pZero] != 0) pZero++;
      	pNonZero = pZero + 1;
      	while(pNonZero < nums.length && nums[pNonZero] == 0) pNonZero++; 
      }
  }
  
  public int minCut(String s) {
  	int len = s.length();
  	boolean[][] pair = new boolean[len][len];
  	int[] result = new int[len+1];
  	result[0] = 0;
  	
  	for(int i=0; i<len; i++){
  		result[i+1] = Integer.MAX_VALUE;
  		for(int left=0; left<=i; left++){
  			if(s.charAt(left) == s.charAt(i) && (i-left<=1 || pair[left+1][i-1])){
  				pair[left][i] = true;
  				if(left == 0) result[i+1] = 0;
  				else{
  					result[i+1] = Math.min(result[left] + 1, result[i+1]);
  				}
  			}
  		}
  	}
  	return result[len];
  }
  
  public int missingNumber(int[] nums) {
      int n = nums.length;
      int res = 0;
      for(int i=0; i<n; i++){
      	res = res ^ nums[i];
      }
      for(int i=0; i<=n; i++){
      	res = res ^ i;
      }
      return res;
  }
  
  public List<List<Integer>> generate(int numRows) {
  	List<List<Integer>> res = new ArrayList<List<Integer>>();
  	if(numRows == 0) return res;
  	List<Integer> list = new ArrayList<Integer>();
  	list.add(1);
  	res.add(list);
  	if(numRows == 1) return res;
  	else if(numRows > 1) {
  		for(int i=2; i<=numRows; i++){
  			generateHelper(i, res);
  		}
  	}
  	return res;
  }
  
  private void generateHelper(int n, List<List<Integer>> res){
  	List<Integer> list = res.get(n-2);
  	List<Integer> newList = new ArrayList<Integer>();
  	newList.add(1);
  	for(int i=0; i<n-2; i++){
  		int sum = list.get(i) + list.get(i+1);
  		newList.add(sum);
  	}
  	newList.add(1);
  	res.add(newList);
  }
  
  public List<Integer> getRow(int rowIndex) {
  	List<Integer> res = new ArrayList<Integer>();
  	for(int i=0; i<=rowIndex; i++){
  		res.add(0, 1);
  		for(int j=1; j<res.size()-1; j++){
  			res.set(j, res.get(j) + res.get(j+1));
  		}
  	}
  	return res;
  }
  
  public int[] singleNumber3(int[] nums) {
      int[] result = new int[2];
      int s = 0;
      for(int i=0; i<nums.length; i++){
      	s ^= nums[i];
      }
      int k = 0;
      int tmp = s;
      while((tmp & 1) == 0){
      	tmp = tmp >> 1;
      	k++;
      }
      int s2 = 0;
      for(int i=0; i<nums.length; i++){
      	if(((nums[i]>>k) & 1) != 0 )
      		s2 ^= nums[i];
      }
     result[0] = s2;
     result[1] = s ^ s2;
     return result;
  }
  
  public boolean containsDuplicate(int[] nums) {
  	Set<Integer> set = new HashSet<Integer>();
      for(int i : nums){
      	if(set.contains(i)) return true;
      	else{
      		set.add(i);
      	}
      }
      return false;
  }
  
  public boolean containsNearbyDuplicate(int[] nums, int k) {
      Set<Integer> set = new HashSet<Integer>();		//长度为k+1的滑动窗
      for(int i = 0; i < nums.length; i++){
          if(i > k) set.remove(nums[i-k-1]);
          if(!set.add(nums[i])) return true;
      }
      return false;
  }
  
  public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
      if (k < 1 || t < 0) return false;
      Map<Long, Long> map = new HashMap<Long, Long>();
      for (int i = 0; i < nums.length; i++) {
          long remappedNum = (long) nums[i] - Integer.MIN_VALUE;
          long bucket = remappedNum / ((long) t + 1);
          if (map.containsKey(bucket)
                  || (map.containsKey(bucket - 1) && remappedNum - map.get(bucket - 1) <= t)
                      || (map.containsKey(bucket + 1) && map.get(bucket + 1) - remappedNum <= t))
                          return true;
          if (i >= k) {
              long lastBucket = ((long) nums[i - k] - Integer.MIN_VALUE) / ((long) t + 1);
              map.remove(lastBucket);
          }
          map.put(bucket, remappedNum);
      }
      return false;
  }
  
  public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
      while ((root.val - p.val) * (root.val - q.val) > 0)
          root = p.val < root.val ? root.left : root.right;
      return root;
  }
  
  public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
      if (root == null || root == p || root == q) return root;
      TreeNode left = lowestCommonAncestor(root.left, p, q);
      TreeNode right = lowestCommonAncestor(root.right, p, q);
      return left == null ? right : right == null ? left : root;
  }
  
  public TreeNode sortedArrayToBST(int[] nums) {
      return sortedArrayToBSTHelper(nums, 0, nums.length-1);
  }
  
  private TreeNode sortedArrayToBSTHelper(int[] nums, int begin, int end){
  	if(begin > end) return null;
  	int mid = (begin + end)/2;
  	TreeNode root = new TreeNode(nums[mid]);
  	root.left = sortedArrayToBSTHelper(nums, begin, mid-1);
  	root.right = sortedArrayToBSTHelper(nums, mid+1, end);
  	return root;
  }
  
  public void deleteNode(ListNode node) {
      node.val = node.next.val;
      node.next = node.next.next;
  }
  
  public int minDistance(String word1, String word2) {
      int m = word1.length(), n = word2.length();
      int[][] dp = new int[m+1][n+1];
      for (int i = 1; i <= m; i++)
          dp[i][0] = i;
      for (int j = 1; j <= n; j++)
          dp[0][j] = j;
      for (int i = 1; i <= m; i++) {
          for (int j = 1; j <= n; j++) {
              if (word1.charAt(i-1) == word2.charAt(j-1))
                  dp[i][j] = dp[i - 1][j - 1];
              else dp[i][j] = Math.min(dp[i - 1][j - 1] + 1, 
              		Math.min(dp[i][j - 1] + 1, dp[i - 1][j] + 1));
          }
      }
      return dp[m][n];
  }
  
  public boolean isBalanced(TreeNode root) {
      return dfsHeight(root) != -1;
  }
  
  private int dfsHeight (TreeNode root) {
      if (root == null) return 0;

      int leftHeight = dfsHeight (root.left);
      if (leftHeight == -1) return -1;
      int rightHeight = dfsHeight (root.right);
      if (rightHeight == -1) return -1;

      if (Math.abs(leftHeight - rightHeight) > 1)  return -1;
      return Math.max(leftHeight, rightHeight) + 1;
  }
  
  public int divide(int dividend, int divisor) {
  	 if (divisor == 0 || (dividend == Integer.MIN_VALUE && divisor == -1))
           return Integer.MAX_VALUE;
       int sign = ((dividend < 0) ^ (divisor < 0)) ? -1 : 1;
       long dvd = Math.abs((long)dividend);
       long dvs = Math.abs((long)divisor);
       int res = 0;
       while (dvd >= dvs) { 
           long temp = dvs, multiple = 1;
           while (dvd >= (temp << 1)) {
               temp <<= 1;
               multiple <<= 1;
           }
           dvd -= temp;
           res += multiple;
       }
       return sign == 1 ? res : -res; 
  }
  
  public boolean isPowerOfTwo(int n) {
	  if(n <= 0) return false;
      return (n & (n-1)) == 0;
  }
  
  public int maximalSquare(char[][] matrix) {
	  if(matrix.length == 0) return 0;
	    int m = matrix.length, n = matrix[0].length, result = 0;
	    int[][] b = new int[m+1][n+1];
	    for (int i = 1 ; i <= m; i++) {
	        for (int j = 1; j <= n; j++) {
	            if(matrix[i-1][j-1] == '1') {
	                b[i][j] = Math.min(Math.min(b[i][j-1] , b[i-1][j-1]), b[i-1][j]) + 1;
	                result = Math.max(b[i][j], result); // update result
	            }
	        }
	    }
	    return result*result;
  }
  
  public int maximalRectangle(char[][] matrix) {
	  if (matrix==null||matrix.length==0||matrix[0].length==0)
          return 0;
      int cLen = matrix[0].length;    // column length
      int rLen = matrix.length;       // row length
      // height array 
      int[] h = new int[cLen+1];
      h[cLen]=0;
      int max = 0;
      for (int row=0;row<rLen;row++) {
          Stack<Integer> s = new Stack<Integer>();
          for (int i=0;i<cLen+1;i++) {
              if (i<cLen)
                  if(matrix[row][i]=='1')
                      h[i]+=1;
                  else h[i]=0;

              if (s.isEmpty()||h[s.peek()]<=h[i])
                  s.push(i);
              else {
                  while(!s.isEmpty()&&h[i]<h[s.peek()]){
                      int top = s.pop();
                      int area = h[top]*(s.isEmpty()?i:(i-s.peek()-1));
                      if (area>max)
                          max = area;
                  }
                  s.push(i);
              }
          }
      }
      return max;
  }
  
  public int maxProfit3(int[] prices) {
      int hold1 = Integer.MIN_VALUE, hold2 = Integer.MIN_VALUE;
      int release1 = 0, release2 = 0;
      for(int i:prices){                              // Assume we only have 0 money at first
          release2 = Math.max(release2, hold2+i);     // The maximum if we've just sold 2nd stock so far.
          hold2    = Math.max(hold2,    release1-i);  // The maximum if we've just buy  2nd stock so far.
          release1 = Math.max(release1, hold1+i);     // The maximum if we've just sold 1nd stock so far.
          hold1    = Math.max(hold1,    -i);          // The maximum if we've just buy  1st stock so far. 
      }
      return release2; ///Since release1 is initiated as 0, so release2 will always higher than release1.
  }
  
  TreeNode firstElement = null;
  TreeNode secondElement = null;
  // The reason for this initialization is to avoid null pointer exception in the first comparison when prevElement has not been initialized
  TreeNode prevElement = new TreeNode(Integer.MIN_VALUE);
  
  public void recoverTree(TreeNode root) {
	  // In order traversal to find the two elements
      traverse(root);

      // Swap the values of the two nodes
      int temp = firstElement.val;
      firstElement.val = secondElement.val;
      secondElement.val = temp; 
  }
  
  private void traverse(TreeNode root) {
      if (root == null)
          return;
      traverse(root.left);
      // Start of "do some business", 
      // If first element has not been found, assign it to prevElement
      if (firstElement == null && prevElement.val >= root.val) {
          firstElement = prevElement;
      }
      // If first element is found, assign the second element to the roo
      if (firstElement != null && prevElement.val >= root.val) {
          secondElement = root;
      }        
      prevElement = root;
      traverse(root.right);
  }
  
  public boolean isPowerOfThree(int n) {
	    if(n>1)
	        while(n%3==0) n /= 3;
	    return n==1;
  }
  
  public String reverseWords(String s) {
      String[] parts = s.trim().split("\\s+");
      StringBuilder out = new StringBuilder();
      if (parts.length > 0) {
          for (int i = parts.length - 1; i > 0; i--) {
              out.append(parts[i]).append(" ");
          }
          out.append(parts[0]);
      }
      return out.toString();
  }
  
  public int coinChange(int[] coins, int amount) {
	    if(amount<1) return 0;
	    return coinChangehelper(coins, amount, new int[amount]);
	}

	private int coinChangehelper(int[] coins, int rem, int[] count) { // rem: remaining coins after the last step; count[rem]: minimum number of coins to sum up to rem
	    if(rem<0) return -1; // not valid
	    if(rem==0) return 0; // completed
	    if(count[rem-1] != 0) return count[rem-1]; // already computed, so reuse
	    int min = Integer.MAX_VALUE;
	    for(int coin : coins) {
	        int res = coinChangehelper(coins, rem-coin, count);
	        if(res>=0)
	            min = Math.min(min, res+1);
	    }
	    count[rem-1] = (min==Integer.MAX_VALUE) ? -1 : min;
	    return count[rem-1];
	}

	public int[] productExceptSelf(int[] nums) {
		int n = nums.length;
		int[] res = new int[n];
		res[0] = 1;
		for (int i = 1; i < n; i++) {
			res[i] = res[i - 1] * nums[i - 1];
		}
		int right = 1;
		for (int i = n - 1; i >= 0; i--) {
			res[i] *= right;
			right *= nums[i];
		}
		return res;
	}

	public boolean isAnagram(String s, String t) {
		Map<Character, Integer> map = new HashMap<>();
		if(s.length() != t.length()) return false;
		for(int i=0; i<s.length(); i++){
			Character c = s.charAt(i);
			if(map.containsKey(c)){
				map.put(c, map.get(c) + 1);
			}else{
				map.put(c, 1);
			}
		}
		for(int j=0; j<t.length(); j++){
			Character d = t.charAt(j);
			if(!map.containsKey(d)) return false;
			else{
				if(map.get(d) == 0) return false;
				else map.put(d, map.get(d) - 1);
			}
		}
		return true;
	}

	// you need treat n as an unsigned value
	public int reverseBits(int n) {
        int result = 0;
        for (int i = 0; i < 32; i++) {
            result += n & 1;
            n >>>= 1;   // CATCH: must do unsigned shift
            if (i < 31) // CATCH: for last digit, don't shift!
                result <<= 1;
        }
        return result;
	}

	public List<String> wordBreak(String s, List<String> wordDict) {
		return DFS(s, wordDict, new HashMap<>());
	}

	// DFS function returns an array including all substrings derived from s.
	List<String> DFS(String s, List<String> wordDict, HashMap<String, LinkedList<String>>map) {
		if (map.containsKey(s))
			return map.get(s);

		LinkedList<String> res = new LinkedList<>();
		if (s.length() == 0) {
			res.add("");
			return res;
		}
		for (String word : wordDict) {
			if (s.startsWith(word)) {
				List<String> sublist = DFS(s.substring(word.length()), wordDict, map);
				for (String sub : sublist)
					res.add(word + (sub.isEmpty() ? "" : " ") + sub);
			}
		}
		map.put(s, res);
		return res;
	}

	public boolean isUgly(int num) {
		for (int i=2; i<6 && num>0; i++)
			while (num % i == 0)
				num /= i;
		return num == 1;
	}

	public boolean isPerfectSquare(int num) {
		int lo = 1, hi = num;
		while(lo <= hi){
			int mid = lo + (hi - lo)/2;
			long result = (long)mid * (long)mid;	//use long because of overflow
			if(result < num){
				lo = mid + 1;
			}else if(result > num){
				hi = mid - 1;
			}else{
				return true;
			}
		}
		return false;
	}

	public boolean canFinish(int numCourses, int[][] prerequisites) {
		int[][] matrix = new int[numCourses][numCourses]; // i -> j
		int[] indegree = new int[numCourses];

		for (int i=0; i<prerequisites.length; i++) {
			int ready = prerequisites[i][0];
			int pre = prerequisites[i][1];
			if (matrix[pre][ready] == 0)
				indegree[ready]++; //duplicate case
			matrix[pre][ready] = 1;
		}

		int count = 0;
		Queue<Integer> queue = new LinkedList();
		for (int i=0; i<indegree.length; i++) {
			if (indegree[i] == 0) queue.offer(i);
		}
		while (!queue.isEmpty()) {
			int course = queue.poll();
			count++;
			for (int i=0; i<numCourses; i++) {
				if (matrix[course][i] != 0) {
					if (--indegree[i] == 0)
						queue.offer(i);
				}
			}
		}
		return count == numCourses;
	}

	public int[] intersection(int[] nums1, int[] nums2) {
		Set<Integer> set = new HashSet<>();
		Set<Integer> intersect = new HashSet<>();
		for (int i = 0; i < nums1.length; i++) {
			set.add(nums1[i]);
		}
		for (int i = 0; i < nums2.length; i++) {
			if (set.contains(nums2[i])) {
				intersect.add(nums2[i]);
			}
		}
		int[] result = new int[intersect.size()];
		int i = 0;
		for (Integer num : intersect) {
			result[i++] = num;
		}
		return result;
	}

	public int integerReplacement(int n) {
		int c = 0;
		while (n != 1) {
			if ((n & 1) == 0) {
				n >>>= 1;
			} else if (n == 3 || ((n >>> 1) & 1) == 0) {
				--n;
			} else {
				++n;
			}
			++c;
		}
		return c;
	}

	public boolean isPowerOfFour(int num) {
		return num > 0 && (num&(num-1)) == 0 && (num & 0x55555555) != 0;
		//0x55555555 is to get rid of those power of 2 but not power of 4
		//so that the single 1 bit always appears at the odd position
	}

	/**
	 * dp[i][j]: the longest palindromic subsequence's length of substring(i, j)
	 * State transition:
	 * dp[i][j] = dp[i+1][j-1] + 2 if s.charAt(i) == s.charAt(j)
	 * otherwise, dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1])
	 * Initialization: dp[i][i] = 1
	 */
	public int longestPalindromeSubseq(String s) {
		int[][] dp = new int[s.length()][s.length()];

		for (int i = s.length() - 1; i >= 0; i--) {
			dp[i][i] = 1;
			for (int j = i+1; j < s.length(); j++) {
				if (s.charAt(i) == s.charAt(j)) {
					dp[i][j] = dp[i+1][j-1] + 2;
				} else {
					dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1]);
				}
			}
		}
		return dp[0][s.length()-1];
	}

	public boolean isValidSerialization(String preorder) {
		String[] nodes = preorder.split(",");
		int diff = 1;
		for (String node: nodes) {
			if (--diff < 0) return false;
			if (!node.equals("#")) diff += 2;
		}
		return diff == 0;
	}

    public ListNode removeElements(ListNode head, int val) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;

        ListNode pre = dummy;
        ListNode cur = head;
        while(cur != null){
            if(cur.val == val){
                pre.next = cur.next;
                cur = pre.next;
            }else{
                pre = cur;
                cur = cur.next;
            }
        }
        return dummy.next;
    }

    public int lastRemaining(int n) {
        boolean left = true;
        int remaining = n;
        int step = 1;
        int head = 1;
        while (remaining > 1) {
            if (left || remaining % 2 ==1) {
                head = head + step;
            }
            remaining = remaining / 2;
            step = step * 2;
            left = !left;
        }
        return head;
    }

	/* The guess API is defined in the parent class GuessGame.
   @param num, your guess
   @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
      int guess(int num); */
	public int guessNumber(int n) {
		int lo = 1, hi = n, mid;
		while(lo < hi){
			mid = lo + (hi-lo)/2;
			int res = guess(mid);
			if(res == 0)
				return mid;
			else if(res == -1){
				hi = mid - 1;
			}else{
				lo = mid + 1;
			}
		}
		return lo;
	}

	private int guess(int num){
		return 0;
	}

    public boolean hasPathSum(TreeNode root, int sum) {
	    if(root == null) return false;
        if(root.left == null && root.right == null)
            return root.val == sum;
        return hasPathSum (root.left, sum - root.val)
               || hasPathSum(root.right, sum - root.val);
    }

    int DIV = 1337;

    List<Integer> findLoop(int a){
        List<Integer> index = new ArrayList<>();
        boolean[] set = new boolean[DIV];
        int rem = a % DIV;
        while ( ! set[rem] ) {
            set[rem]=true;
            index.add(rem);
            rem = (rem*a) % DIV;
        }
        return index;
    }

    int modBy(int[] b, int m){
        int rem = 0;
        for (int i=0; i < b.length; i++) {
            rem = (rem*10+b[i]) % m;
        }
        return rem;
    }

    public int superPow(int a, int[] b) {
        if (a==0 || a==DIV || b==null || b.length == 0) return 0;
        if (a==1) return 1;
        if (a > DIV) return superPow( a % DIV, b);
        List<Integer> index = findLoop(a);
        int loopsize = index.size();
        int rem = modBy(b, loopsize);
        rem = rem==0? loopsize: rem;
        return index.get(rem-1);
    }

	public int countNodes(TreeNode root) {
		int h = height(root);
		return h < 0 ? 0 :
				height(root.right) == h-1 ? (1 << h) + countNodes(root.right)
						: (1 << h-1) + countNodes(root.left);
	}

	int height(TreeNode root) {
		return root == null ? -1 : 1 + height(root.left);
	}

	public int maxPoints(Point[] points) {
		if (points==null) return 0;
		if (points.length<=2) return points.length;

		Map<Integer,Map<Integer,Integer>> map = new HashMap<Integer,Map<Integer,Integer>>();
		int result=0;
		for (int i=0;i<points.length;i++){
			map.clear();
			int overlap=0,max=0;
			for (int j=i+1;j<points.length;j++){
				int x=points[j].x-points[i].x;
				int y=points[j].y-points[i].y;
				if (x==0&&y==0){
					overlap++;
					continue;
				}
				int gcd=generateGCD(x,y);
				if (gcd!=0){
					x/=gcd;
					y/=gcd;
				}

				if (map.containsKey(x)){
					if (map.get(x).containsKey(y)){
						map.get(x).put(y, map.get(x).get(y)+1);
					}else{
						map.get(x).put(y, 1);
					}
				}else{
					Map<Integer,Integer> m = new HashMap<Integer,Integer>();
					m.put(y, 1);
					map.put(x, m);
				}
				max=Math.max(max, map.get(x).get(y));
			}
			result=Math.max(result, max+overlap+1);
		}
		return result;
	}

	/**
	 * 求最大公约数
	 * @param a
	 * @param b
	 * @return
	 */
	private int generateGCD(int a,int b){

		if (b==0) return a;
		else return generateGCD(b,a%b);
	}

	public int[] countBits(int num) {
		int[] f = new int[num + 1];
		for (int i=1; i<=num; i++) f[i] = f[i >> 1] + (i & 1);
		return f;
	}

	public String toHex(int num) {
		if(num == 0) return "0";
		StringBuilder sb = new StringBuilder();
		for(int i=7; i>=0; i--){
			int tmp = (num >>> (i<<2)) & 0xf;
			if(tmp == 0 && sb.length()==0) continue;
			if(tmp < 10){
				sb.append(tmp);
			}else{
				sb.append((char)('a' + tmp - 10));
			}
		}
		return sb.toString();
	}

	public int maxProfit(int k, int[] prices) {
		int len = prices.length;
		if (k >= len / 2) return quickSolve(prices);

		int[][] t = new int[k + 1][len];
		for (int i = 1; i <= k; i++) {
			//tmpMax means the maximum profit of just doing at most i-1 transactions,
			// using at most first j-1 prices, and buying the stock at price[j] - this is used for the next loop.
			int tmpMax =  -prices[0];
			for (int j = 1; j < len; j++) {
				t[i][j] = Math.max(t[i][j - 1], prices[j] + tmpMax);
				tmpMax =  Math.max(tmpMax, t[i - 1][j - 1] - prices[j]);
			}
		}
		return t[k][len - 1];
	}

	private int quickSolve(int[] prices) {
		int len = prices.length, profit = 0;
		for (int i = 1; i < len; i++)
			// as long as there is a price gap, we gain a profit.
			if (prices[i] > prices[i - 1]) profit += prices[i] - prices[i - 1];
		return profit;
	}

	public int strongPasswordChecker(String s) {

		if(s.length()<2) return 6-s.length();

		//Initialize the states, including current ending character(end), existence of lowercase letter(lower), uppercase letter(upper), digit(digit) and number of replicates for ending character(end_rep)
		char end = s.charAt(0);
		boolean upper = end>='A'&&end<='Z', lower = end>='a'&&end<='z', digit = end>='0'&&end<='9';

		//Also initialize the number of modification for repeated characters, total number needed for eliminate all consequnce 3 same character by replacement(change), and potential maximun operation of deleting characters(delete). Note delete[0] means maximum number of reduce 1 replacement operation by 1 deletion operation, delete[1] means maximun number of reduce 1 replacement by 2 deletion operation, delete[2] is no use here.
		int end_rep = 1, change = 0;
		int[] delete = new int[3];

		for(int i = 1;i<s.length();++i){
			if(s.charAt(i)==end) ++end_rep;
			else{
				change+=end_rep/3;
				if(end_rep/3>0) ++delete[end_rep%3];
				//updating the states
				end = s.charAt(i);
				upper = upper||end>='A'&&end<='Z';
				lower = lower||end>='a'&&end<='z';
				digit = digit||end>='0'&&end<='9';
				end_rep = 1;
			}
		}
		change+=end_rep/3;
		if(end_rep/3>0) ++delete[end_rep%3];

		//The number of replcement needed for missing of specific character(lower/upper/digit)
		int check_req = (upper?0:1)+(lower?0:1)+(digit?0:1);

		if(s.length()>20){
			int del = s.length()-20;

			//Reduce the number of replacement operation by deletion
			if(del<=delete[0]) change-=del;
			else if(del-delete[0]<=2*delete[1]) change-=delete[0]+(del-delete[0])/2;
			else change-=delete[0]+delete[1]+(del-delete[0]-2*delete[1])/3;

			return del+Math.max(check_req,change);
		}
		else return Math.max(6-s.length(), Math.max(check_req, change));
	}

	public boolean detectCapitalUse(String word) {
		int cnt = 0;
		for(char c: word.toCharArray()) if('Z' - c >= 0) cnt++;
		return ((cnt==0 || cnt==word.length()) || (cnt==1 && 'Z' - word.charAt(0)>=0));
	}

    public int maximumGap(int[] num) {
        if (num == null || num.length < 2)
            return 0;
        // get the max and min value of the array
        int min = num[0];
        int max = num[0];
        for (int i:num) {
            min = Math.min(min, i);
            max = Math.max(max, i);
        }
        // the minimum possibale gap, ceiling of the integer division
        int gap = (int)Math.ceil((double)(max - min)/(num.length - 1));
        int[] bucketsMIN = new int[num.length - 1]; // store the min value in that bucket
        int[] bucketsMAX = new int[num.length - 1]; // store the max value in that bucket
        Arrays.fill(bucketsMIN, Integer.MAX_VALUE);
        Arrays.fill(bucketsMAX, Integer.MIN_VALUE);
        // put numbers into buckets
        for (int i:num) {
            if (i == min || i == max)
                continue;
            int idx = (i - min) / gap; // index of the right position in the buckets
            bucketsMIN[idx] = Math.min(i, bucketsMIN[idx]);
            bucketsMAX[idx] = Math.max(i, bucketsMAX[idx]);
        }
        // scan the buckets for the max gap
        int maxGap = Integer.MIN_VALUE;
        int previous = min;
        for (int i = 0; i < num.length - 1; i++) {
            if (bucketsMIN[i] == Integer.MAX_VALUE && bucketsMAX[i] == Integer.MIN_VALUE)
                // empty bucket
                continue;
            // min value minus the previous value is the current gap
            maxGap = Math.max(maxGap, bucketsMIN[i] - previous);
            // update previous bucket value
            previous = bucketsMAX[i];
        }
        maxGap = Math.max(maxGap, max - previous); // updata the final max value gap
        return maxGap;
    }

	public String fractionToDecimal(int numerator, int denominator) {
		if (numerator == 0) {
			return "0";
		}
		StringBuilder res = new StringBuilder();
		// "+" or "-"
		res.append(((numerator > 0) ^ (denominator > 0)) ? "-" : "");
		long num = Math.abs((long)numerator);
		long den = Math.abs((long)denominator);

		// integral part
		res.append(num / den);
		num %= den;
		if (num == 0) {
			return res.toString();
		}

		// fractional part
		res.append(".");
		HashMap<Long, Integer> map = new HashMap<Long, Integer>();
		map.put(num, res.length());
		while (num != 0) {
			num *= 10;
			res.append(num / den);
			num %= den;
			if (map.containsKey(num)) {
				int index = map.get(num);
				res.insert(index, "(");
				res.append(")");
				break;
			}
			else {
				map.put(num, res.length());
			}
		}
		return res.toString();
	}

	public int kthSmallest(TreeNode root, int k) {
		int count = countNodes2(root.left);
		if (k <= count) {
			return kthSmallest(root.left, k);
		} else if (k > count + 1) {
			return kthSmallest(root.right, k-1-count); // 1 is counted as current node
		}

		return root.val;
	}

	public int countNodes2(TreeNode n) {
		if (n == null) return 0;

		return 1 + countNodes2(n.left) + countNodes2(n.right);
	}

	public int lengthOfLIS(int[] nums) {
		int[] dp = new int[nums.length];
		int len = 0;

		for(int x : nums) {
			int i = Arrays.binarySearch(dp, 0, len, x);
			if(i < 0) i = -(i + 1);
			dp[i] = x;
			if(i == len) len++;
		}

		return len;
	}

	public void gameOfLife(int[][] board) {
		if (board == null || board.length == 0) return;
		int m = board.length, n = board[0].length;

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				int lives = liveNeighbors(board, m, n, i, j);

				// In the beginning, every 2nd bit is 0;
				// So we only need to care about when will the 2nd bit become 1.
				if (board[i][j] == 1 && lives >= 2 && lives <= 3) {
					board[i][j] = 3; // Make the 2nd bit 1: 01 ---> 11
				}
				if (board[i][j] == 0 && lives == 3) {
					board[i][j] = 2; // Make the 2nd bit 1: 00 ---> 10
				}
			}
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				board[i][j] >>= 1;  // Get the 2nd state.
			}
		}
	}

	public int liveNeighbors(int[][] board, int m, int n, int i, int j) {
		int lives = 0;
		for (int x = Math.max(i - 1, 0); x <= Math.min(i + 1, m - 1); x++) {
			for (int y = Math.max(j - 1, 0); y <= Math.min(j + 1, n - 1); y++) {
				lives += board[x][y] & 1;
			}
		}
		lives -= board[i][j] & 1;
		return lives;
	}

	public TreeNode invertTree(TreeNode root) {
		if (root == null) {
			return null;
		}

		final Queue<TreeNode> queue = new LinkedList<>();
		queue.offer(root);

		while(!queue.isEmpty()) {
			final TreeNode node = queue.poll();
			final TreeNode left = node.left;
			node.left = node.right;
			node.right = left;

			if(node.left != null) {
				queue.offer(node.left);
			}
			if(node.right != null) {
				queue.offer(node.right);
			}
		}
		return root;
	}

	public List<String> fizzBuzz(int n) {
		List<String> res = new ArrayList<>(n);
		for(int i=1; i<=n; i++){
			if(i % 3 == 0 && i % 5 == 0) res.add("FizzBuzz");
			else if(i % 3 == 0) res.add("Fizz");
			else if(i % 5 == 0) res.add("Buzz");
			else res.add(String.valueOf(i));
		}
		return res;
	}

	public int hIndex(int[] citations) {
		int length = citations.length;
		if (length == 0) {
			return 0;
		}

		int[] array2 = new int[length + 1];
		for (int i = 0; i < length; i++) {
			if (citations[i] > length) {
				array2[length] += 1;
			} else {
				array2[citations[i]] += 1;
			}
		}
		int t = 0;

		for (int i = length; i >= 0; i--) {
			t = t + array2[i];
			if (t >= i) {
				return i;
			}
		}
		return 0;
	}

	public String predictPartyVictory(String senate) {
		Queue<Integer> q1 = new LinkedList<>(), q2 = new LinkedList<>();
		int n = senate.length();
		for(int i = 0; i<n; i++){
			if(senate.charAt(i) == 'R')q1.add(i);
			else q2.add(i);
		}
		while(!q1.isEmpty() && !q2.isEmpty()){
			int r_index = q1.poll(), d_index = q2.poll();
			if(r_index < d_index)q1.add(r_index + n);
			else q2.add(d_index + n);
		}
		return (q1.size() > q2.size())? "Radiant" : "Dire";
	}

	public ListNode addTwoNumbers2(ListNode l1, ListNode l2) {
		Stack<Integer> s1 = new Stack<>();
		Stack<Integer> s2 = new Stack<>();

		while(l1 != null) {
			s1.push(l1.val);
			l1 = l1.next;
		}
		while(l2 != null) {
			s2.push(l2.val);
			l2 = l2.next;
		}

		int sum = 0;
		ListNode list = new ListNode(0);
		while (!s1.empty() || !s2.empty()) {
			if (!s1.empty()) sum += s1.pop();
			if (!s2.empty()) sum += s2.pop();
			list.val = sum % 10;
			ListNode head = new ListNode(sum / 10);
			head.next = list;
			list = head;
			sum /= 10;
		}

		return list.val == 0 ? list.next : list;
	}

	public List<List<Integer>> levelOrder(TreeNode root) {
		Queue<TreeNode> queue = new LinkedList<>();
		List<List<Integer>> wrapList = new LinkedList<>();

		if(root == null) return wrapList;

		queue.offer(root);
		while(!queue.isEmpty()){
			int levelNum = queue.size();
			List<Integer> subList = new LinkedList<>();
			for(int i=0; i<levelNum; i++) {
				if(queue.peek().left != null) queue.offer(queue.peek().left);
				if(queue.peek().right != null) queue.offer(queue.peek().right);
				subList.add(queue.poll().val);
			}
			wrapList.add(subList);
		}
		return wrapList;
	}

	public int wiggleMaxLength(int[] nums) {
		if (nums.length == 0 || nums.length == 1) {
			return nums.length;
		}
		int k = 0;
		while (k < nums.length - 1 && nums[k] == nums[k + 1]) {  //Skips all the same numbers from series beginning eg 5, 5, 5, 1
			k++;
		}
		if (k == nums.length - 1) {
			return 1;
		}
		int result = 2;     // This will track the result of result array
		boolean smallReq = nums[k] < nums[k + 1];       //To check series starting pattern
		for (int i = k + 1; i < nums.length - 1; i++) {
			if (smallReq && nums[i + 1] < nums[i]) {
				nums[result] = nums[i + 1];
				result++;
				smallReq = !smallReq;    //Toggle the requirement from small to big number
			} else {
				if (!smallReq && nums[i + 1] > nums[i]) {
					nums[result] = nums[i + 1];
					result++;
					smallReq = !smallReq;    //Toggle the requirement from big to small number
				}
			}
		}
		return result;
	}

	public void rotate(int[] nums, int k) {
		k %= nums.length;
		reverse(nums, 0, nums.length - 1);
		reverse(nums, 0, k - 1);
		reverse(nums, k, nums.length - 1);
	}

	private void reverse(int[] nums, int start, int end) {
		while (start < end) {
			int temp = nums[start];
			nums[start] = nums[end];
			nums[end] = temp;
			start++;
			end--;
		}
	}

	public int islandPerimeter(int[][] grid) {
		int islands = 0, neighbours = 0;

		for (int i = 0; i < grid.length; i++) {
			for (int j = 0; j < grid[i].length; j++) {
				if (grid[i][j] == 1) {
					islands++; // count islands
					if (i < grid.length - 1 && grid[i + 1][j] == 1) neighbours++; // count down neighbours
					if (j < grid[i].length - 1 && grid[i][j + 1] == 1) neighbours++; // count right neighbours
				}
			}
		}

		return islands * 4 - neighbours * 2;
	}

	public static void main(String[] args){
		Solution s = new Solution();
		//int[][] nums = {{0,1,0,0}, {1,1,1,0}, {0,1,0,0}, {1,1,0,0}};
		int[][] nums = {{1,1,1}, {1,0,1}};
		System.out.println(s.islandPerimeter(nums));
	}
}
