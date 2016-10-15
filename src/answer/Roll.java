package answer;

import java.util.Random;
public class Roll{
	public static void main (String[] args){
		Random r = new Random();
		System.out.println("ÕÅéó£º" + r.nextInt(100));
		System.out.println("ÂÀê¿³¿£º" + r.nextInt(100));
	}
}