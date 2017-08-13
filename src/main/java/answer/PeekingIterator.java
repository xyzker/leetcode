package answer;

import java.util.Iterator;

/**
 * Created by kexi on 2017/3/12.
 */
// Java Iterator interface reference:
// https://docs.oracle.com/javase/8/docs/api/java/util/Iterator.html
public class PeekingIterator implements Iterator<Integer>{
    private final Iterator<Integer> iterator;
    private Integer cur;
    private boolean peekCalled;


    public PeekingIterator(Iterator<Integer> iterator) {
        // initialize any member here.
        this.iterator = iterator;
    }

    // Returns the next element in the iteration without advancing the iterator.
    public Integer peek() {
        if(!peekCalled) {
            cur = iterator.next();
            peekCalled = true;
        }
        return cur;
    }

    // hasNext() and next() should behave the same as in the Iterator interface.
    // Override them if needed.
    @Override
    public Integer next() {
        if(!peekCalled){
            return iterator.next();
        }else{
            peekCalled = false;
            return cur;
        }
    }

    @Override
    public boolean hasNext() {
        if(peekCalled){
            return true;
        }else{
            return iterator.hasNext();
        }
    }
}
