package answer;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

public class SolutionTest {
    Solution solution;

    @Before
    public void init(){
        solution = new Solution();
    }

    @Test
    public void test(){
        assertEquals(solution.detectCapitalUse("USA"), true);
        assertEquals(solution.detectCapitalUse("leetcode"), true);
        assertEquals(solution.detectCapitalUse("Google"), true);
    }
}
