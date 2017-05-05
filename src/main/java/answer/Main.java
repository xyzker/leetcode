package answer;

import java.util.*;
public class Main
{
/*        public static void main(String args[]){
        	Scanner cin = new Scanner(System.in);
        	String str = cin.nextLine();
        	String[] array = str.split(";");
        	List<List<Integer>> matrix = new ArrayList<List<Integer>>();
        	for(int i=0; i<array.length; i++){
        		List<Integer> list = new ArrayList<Integer>();
        		String[] tmp = array[i].split(" ");
        		for(int j=0; j<tmp.length; j++){
        			list.add(Integer.parseInt(tmp[j]));
        		}
        		matrix.add(list);
        	}
        	System.out.println(getMax(matrix));
    }
        
    public static int getMax(List<List<Integer>> matrix){
    	int result = 0;
    	int row = matrix.size();
    	int col = matrix.get(0).size();
    	for(int i=0; i<row-1; i++)
    		for(int j=0; j<col-1; j++){
    			int sum = matrix.get(i).get(j) + matrix.get(i+1).get(j)
    						+ matrix.get(i).get(j+1) + matrix.get(i+1).get(j+1);
    			if(result < sum)
    				result = sum;
    		}
    	return result;
    }*/
	/*    	float x = cin.nextFloat();
	float y = cin.nextFloat();
	float r = cin.nextFloat();
	float x1 = x - r, x2 = x + r, y1 = y - r, y2 = y + r;
	*/
	public static void main(String args[]){
        @SuppressWarnings("resource")
		Scanner s = new Scanner(System.in);
        String a = s.nextLine().trim();
        String b = s.nextLine().trim();
        System.out.format("%.2f%%\n", precision(a, b));
	}
    public static double precision(String a, String b) {
        double n = a.length();
        double counter = 0;
        for (int i=0; i<n; ++i) {
        	char c = a.charAt(i);
            if (Character.isDigit(c) || Character.isLetter(c)) {
                if (b.charAt(i) == '1') counter++;
            } else {
                if (b.charAt(i) == '0') counter++;
            }
        }
        return (counter/n)*100;
    }

    
}
