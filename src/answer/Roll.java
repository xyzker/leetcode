package answer;

import java.util.Random;
public class Roll{
	public static void main (String[] args){
		Random r = new Random();
		System.out.println("����" + r.nextInt(100));
		System.out.println("��꿳���" + r.nextInt(100));
	}
}