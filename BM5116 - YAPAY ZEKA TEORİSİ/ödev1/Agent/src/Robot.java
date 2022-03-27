
import java.util.Random;
/**
 * Bu sýnýf ortamda hareket eden robotu temsil eder. Robotun konum bilgisi
 * ve ortamdaki hareketiyle ilgili arayüzler saðlar.
 */
public class Robot {
	/**Robotun simülatördeki uzunluðu*/
	private int length;
	/**Robotun simülatördeki eni*/
	private int width;
	/**Robotun simülatördeki konumu, pixel biriminde*/
	private int[] position;
	/**Robotun hareket kümesinin eleman sayýsý*/
	private int numOfActions;
	/**Rastgele sayý üreten nesne*/
	private Random rand;
	/**
	 * Robot sýnýfýnýn bir örneðini oluþturur.
	 * @param learningRate robotun gelecek durumlardan öðrenme oraný
	 */
	public Robot(){
		position = new int[2];
		numOfActions = 4;
		rand = new Random();
	}
	/**
	 * state parametresiyle verilen durumdan action parametresiyle verilen 
	 * hareketi gerçekleþtirir. Eðer hareket bir engel dolayýsýyla 
	 * gerçekleþtirilemiyorsa -1 dönülür ve state 
	 * durum parametresi deðiþtirilmez. Eðer hareket gerçekleþtirilebiliyorsa
	 * ilgili hareket yönünde state durumu deðiþtirilir. Bir sonraki durum hedef 
	 * noktasý ise 1 deðeri çaðýrýcý fonksiyona iletilir. Eðer kurabiye yenmiþse 2
	 * deðeri iletilir. Herhangi bir ödül veya ceza alýnmamasý durumunda ise çaðýrýcýya 0 dönülür.
	 * @param state robotun mevcut durumunu ifade eden dizi
	 * @param action mevcut durumda yapýlacak hareket
	 * @return hareketin sonucunda engel olduðu, kurabiye yediðini veya hedefe ulaþtýðýný döner.
	 * Diðer durumda 0 döner.
	 */
	public int act(int[] state, int action){
		int reward = 0;
		int x = state[0];
		int y = state[1];
		//actions:
		if(action == 0)//north
			x--;
		else if(action == 1)//south
			x++;
		else if(action == 2)//east
			y++;
		else if(action == 3)//west
			y--;			
		if(Environment.hasObstacle(x,y))
			reward = -1;
		
		else{
			if(Environment.hasCookie(x, y)){
				reward = 2;
				Environment.eatCookie(x, y);
			}
			state[0] = x;
			state[1] = y;			
		}	
		
		if(Environment.isGoal())
			reward = 1;
		return reward;		
	}
	/**
	 * Bu robot için aðaç aramasý sýnýfýný çaðýrýr
	 * @param sim Öðrenmenin simüle edileceði simülatör
	 */
	public void startSearching(Simulator sim)
	{
		//parametreler: simülatör, robot
		GraphSearch ts = new GraphSearch(sim, this);
		ts.followOptimalPath();
	}
	/**
	 * Hareketler kümesinden rastgele bir hareket seçer
	 * @return rastgele hareket
	 */
	public int chooseRandomAction() {
		return rand.nextInt(numOfActions);
	}
	/**
	 * Ortamda hedef durumuna eþit olmayan ve engele rastlamayan
	 * rastgele bir durum seçer.
	 * @return rastgele seçilen durum {x,y} dizisi þeklinde.
	 */
	public int[] chooseRandomState() {
		int[] state = new int[2];
		boolean found = false;
		while(!found){
			state[0] =  rand.nextInt(Environment.getHeight());
			state[1] =  rand.nextInt(Environment.getWidth());
			if(!Environment.hasObstacle(state[0], state[1]))
				found = true;
		}
		return state;
	}
	
	/**
	 * Robotun uzunluðunu atar.
	 * @param length robotun uzunluðu
	 */
	public void setLength(int length) {
		this.length = length;
	}
	/**
	 * Robotun enini atar.
	 * @param width robotun eni
	 */
	public void setWidth(int width) {
		this.width = width;
	}
	/**
	 * Robotun uzunluðunu döner.
	 * @return robotun uzunluðu
	 */
	public int getLength() {
		return length;
	}
	/**
	 * Robotun enini döner.
	 * @return robotun eni
	 */
	public int getWidth() {
		return width;
	}
	/**
	 * Robotun hareket kümesindeki eleman sayýsýný döner.
	 * @return robotun hareket kümesindeki eleman sayýsý
	 */
	public int getNumOfActions() {
		return numOfActions;
	}
	/**
	 * Robotun simülatördeki x ve y pozisyonunu dizi olarak döner.
	 * @return
	 */
	public int[] getPosition(){
		return position;		
	}
	/**
	 * Robotun simülatördeki x ve y pozisyonunu dizi olarak atar.
	 * @param pos robotun x ve y pozisyonu
	 */
	public void setPosition(int[] pos) {
		position = pos;		
	}
	/**
	 * Robotun simülatördeki x ve y pozisyonunu atar.
	 * @param x robotun x pozisyonu
	 * @param y robotun y pozisyonu
	 */
	public void setPosition(int x, int y) {
		position[0] = x;
		position[1] = y;	
	}
	/**
	 * Verilen hareket anahtarýna tekabül eden Ýngilizce kelimeyi döner
	 * @param action hareket anahtarý
	 * @return hareket anahtarýna tekabül eden Ýngilizce kelime
	 */	
	public String getActionMap(int action) {
		String s="";
		if(action == 0)//north
			s = "north";
		else if(action == 1)//south
			s = "south";
		else if(action == 2)//east
			s = "east";
		else if(action == 3)//west
			s = "west";
		return s;
	}
}
