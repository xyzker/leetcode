package answer;

import java.util.Random;
public class Roll{
	public static void main (String[] args){
		Random r = new Random();
		System.out.println("张轶：" + r.nextInt(100));
		System.out.println("吕昕晨：" + r.nextInt(100));
	}
}