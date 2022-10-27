
import java.io.*;
import java.util.Scanner;
/**
 * Bu sýnýf bir txt dosyasýndan okuduðu ortamý saklar.
 * Tanýmladýðý static deðiþkenler ve fonksiyonlarla herhangi bir sýnýftan
 * ortama eriþimi saðlar.
 */
public class Environment {
	//0->boþ konum, 1->engel 
	/**Ortamýn tutulduðu matris*/
	private static short region[][];
	/**Ortamýn eni*/
	private static int width;
	/**Ortamýn boyu*/
	private static int height;
	/**Robotun ortamdaki baþlangýç noktasý*/
	private static int[] start;
	/**Varýlacak hedef noktasý*/
	private static boolean goal;

	/**
	 * Bu sýnýfýn bir örneðini envronment.txt dosyasýný okuyarak oluþturur.
	 */
	public Environment(){
		start = new int[2];

		//Ortamý dosyadan oku
		Scanner scanner=null;
		try {
			scanner = new Scanner(new File("./src/environment.txt"),"UTF-8");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		//Ortamýn boyutunu oku
		scanner.next();
		width = scanner.nextInt();
		scanner.next();
		height = scanner.nextInt();
		//Ortamý oku	
		region = new short [height][width];
		for(int i=0; i < height; i++){
			for(int j=0; j < width; j++){
				region[i][j] = scanner.nextShort();	
			}
		}
		printEnvironment();
		//Baþlangýç ve hedef konumlarýný oku
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
	 * Ortamda verilen x satýrý ve y sütununda engel olup olmadýðýný döner.
	 * @param x ortamýn x. satýrý
	 * @param y ortamýn y. sütunu
	 * @return x. satýr ve y. sütunda engel olup olmadýðýný
	 */
	public static boolean hasObstacle(int x, int y) {
		if(x<0 || y<0 || x>height-1 || y>width-1 || region[x][y] == 1)
			return true;	
		return false;
	}
	/**
	 * Ortamda verilen x satýrý ve y sütununda kurabiye olup olmadýðýný döner.
	 * @param x ortamýn x. satýrý
	 * @param y ortamýn y. sütunu
	 * @return x. satýr ve y. sütunda engel olup olmadýðýný
	 */
	public static boolean hasCookie(int x, int y) {
		if(x>=0 && y>=0 && x<=height-1 && y<=width-1 && region[x][y] == 2)
			return true;	
		return false;
	}
	/**
	 * Ortamda verilen x satýrý ve y sütununda kurabiye olup olmadýðýný döner.
	 * @param x ortamýn x. satýrý
	 * @param y ortamýn y. sütunu
	 * @return x. satýr ve y. sütunda engel olup olmadýðýný
	 */
	public static void eatCookie(int x, int y) {
		if(x>=0 && y>=0 && x<=height-1 && y<=width-1 && region[x][y] == 2)
			region[x][y] = 0;
	}
	/**
	 * Verilen x satýrý ve y sütununun ortamýn hedef durumu olup olmadýðýný döner.
	 * @param x ortamýn x. satýrý
	 * @param y ortamýn y. sütunu
	 * @return x. satýr ve y. sütunun hedef durumu olup olmadýðýný
	 */
	public static boolean isGoal() {		
		for(int i=0; i < height; i++)
			for(int j=0; j < width; j++)
				if(region[i][j]==2)
					return false;
		goal = true;
		return true;
	}
	// VERÝCÝ FONKSÝYONLAR
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
