package answer;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

/**
 * @author hasee
 * @since 2023/7/8
 */
public class StackOfPlates {

    private int cap;

    private List<Stack<Integer>> stackList = new ArrayList<>();

    public StackOfPlates(int cap) {
        this.cap = cap;
    }

    public void push(int val) {
        if (cap == 0) {
            return;
        }
        if (stackList.isEmpty() || stackList.get(stackList.size()-1).size() == cap) {
            Stack<Integer> stack = new Stack<>();
            stack.push(val);
            stackList.add(stack);
        }else {
            stackList.get(stackList.size()-1).push(val);
        }
    }

    public int pop() {
        if (stackList.isEmpty()) {
            return -1;
        }
        Stack<Integer> stack = stackList.get(stackList.size()-1);
        int val = stack.pop();
        if (stack.empty()) {
            stackList.remove(stackList.size()-1);
        }
        return val;
    }

    public int popAt(int index) {
        if (index < 0 || index > stackList.size() -1) {
            return -1;
        }
        Stack<Integer> stack = stackList.get(index);
        int val = stack.pop();
        if (stack.empty()) {
            stackList.remove(index);
        }
        return val;
    }
}
