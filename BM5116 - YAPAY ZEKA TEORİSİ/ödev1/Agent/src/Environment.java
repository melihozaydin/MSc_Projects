
import java.io.*;
import java.util.Scanner;
/**
 * Bu s�n�f bir txt dosyas�ndan okudu�u ortam� saklar.
 * Tan�mlad��� static de�i�kenler ve fonksiyonlarla herhangi bir s�n�ftan
 * ortama eri�imi sa�lar.
 */
public class Environment {
	//0->bo� konum, 1->engel 
	/**Ortam�n tutuldu�u matris*/
	private static short region[][];
	/**Ortam�n eni*/
	private static int width;
	/**Ortam�n boyu*/
	private static int height;
	/**Robotun ortamdaki ba�lang�� noktas�*/
	private static int[] start;
	/**Var�lacak hedef noktas�*/
	private static boolean goal;

	/**
	 * Bu s�n�f�n bir �rne�ini envronment.txt dosyas�n� okuyarak olu�turur.
	 */
	public Environment(){
		start = new int[2];

		//Ortam� dosyadan oku
		Scanner scanner=null;
		try {
			scanner = new Scanner(new File("./src/environment.txt"),"UTF-8");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		//Ortam�n boyutunu oku
		scanner.next();
		width = scanner.nextInt();
		scanner.next();
		height = scanner.nextInt();
		//Ortam� oku	
		region = new short [height][width];
		for(int i=0; i < height; i++){
			for(int j=0; j < width; j++){
				region[i][j] = scanner.nextShort();	
			}
		}
		printEnvironment();
		//Ba�lang�� ve hedef konumlar�n� oku
		scanner.next();
		start[0] = scanner.nextInt()-1;
		start[1] = scanner.nextInt()-1;
		
	}
	/**
	 * Bu ortam nesnesinin bilgisini konsola basar
	 */
	public void printEnvironment() {
		System.out.println("Given environment is:");
		for(int i=0; i < height; i++){
			for(int j=0; j < width; j++){
				System.out.print(region[i][j]);
			}
			System.out.println();
		}
		System.out.println();
		
	}
	/**
	 * Ortamda verilen x sat�r� ve y s�tununda engel olup olmad���n� d�ner.
	 * @param x ortam�n x. sat�r�
	 * @param y ortam�n y. s�tunu
	 * @return x. sat�r ve y. s�tunda engel olup olmad���n�
	 */
	public static boolean hasObstacle(int x, int y) {
		if(x<0 || y<0 || x>height-1 || y>width-1 || region[x][y] == 1)
			return true;	
		return false;
	}
	/**
	 * Ortamda verilen x sat�r� ve y s�tununda kurabiye olup olmad���n� d�ner.
	 * @param x ortam�n x. sat�r�
	 * @param y ortam�n y. s�tunu
	 * @return x. sat�r ve y. s�tunda engel olup olmad���n�
	 */
	public static boolean hasCookie(int x, int y) {
		if(x>=0 && y>=0 && x<=height-1 && y<=width-1 && region[x][y] == 2)
			return true;	
		return false;
	}
	/**
	 * Ortamda verilen x sat�r� ve y s�tununda kurabiye olup olmad���n� d�ner.
	 * @param x ortam�n x. sat�r�
	 * @param y ortam�n y. s�tunu
	 * @return x. sat�r ve y. s�tunda engel olup olmad���n�
	 */
	public static void eatCookie(int x, int y) {
		if(x>=0 && y>=0 && x<=height-1 && y<=width-1 && region[x][y] == 2)
			region[x][y] = 0;
	}
	/**
	 * Verilen x sat�r� ve y s�tununun ortam�n hedef durumu olup olmad���n� d�ner.
	 * @param x ortam�n x. sat�r�
	 * @param y ortam�n y. s�tunu
	 * @return x. sat�r ve y. s�tunun hedef durumu olup olmad���n�
	 */
	public static boolean isGoal() {		
		for(int i=0; i < height; i++)
			for(int j=0; j < width; j++)
				if(region[i][j]==2)
					return false;
		goal = true;
		return true;
	}
	// VER�C� FONKS�YONLAR
	public static short[][] getEnvironment(){
		return region;
	}
	public static int getWidth() {
		return width;
	}
	public static int getHeight() {
		return height;
	}	
	public static int[] getStart() {
		return start;
	}
	public static boolean getGoal() {
		return goal;
	}

}
