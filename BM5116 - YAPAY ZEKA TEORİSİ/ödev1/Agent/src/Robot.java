
import java.util.Random;
/**
 * Bu s�n�f ortamda hareket eden robotu temsil eder. Robotun konum bilgisi
 * ve ortamdaki hareketiyle ilgili aray�zler sa�lar.
 */
public class Robot {
	/**Robotun sim�lat�rdeki uzunlu�u*/
	private int length;
	/**Robotun sim�lat�rdeki eni*/
	private int width;
	/**Robotun sim�lat�rdeki konumu, pixel biriminde*/
	private int[] position;
	/**Robotun hareket k�mesinin eleman say�s�*/
	private int numOfActions;
	/**Rastgele say� �reten nesne*/
	private Random rand;
	/**
	 * Robot s�n�f�n�n bir �rne�ini olu�turur.
	 * @param learningRate robotun gelecek durumlardan ��renme oran�
	 */
	public Robot(){
		position = new int[2];
		numOfActions = 4;
		rand = new Random();
	}
	/**
	 * state parametresiyle verilen durumdan action parametresiyle verilen 
	 * hareketi ger�ekle�tirir. E�er hareket bir engel dolay�s�yla 
	 * ger�ekle�tirilemiyorsa -1 d�n�l�r ve state 
	 * durum parametresi de�i�tirilmez. E�er hareket ger�ekle�tirilebiliyorsa
	 * ilgili hareket y�n�nde state durumu de�i�tirilir. Bir sonraki durum hedef 
	 * noktas� ise 1 de�eri �a��r�c� fonksiyona iletilir. E�er kurabiye yenmi�se 2
	 * de�eri iletilir. Herhangi bir �d�l veya ceza al�nmamas� durumunda ise �a��r�c�ya 0 d�n�l�r.
	 * @param state robotun mevcut durumunu ifade eden dizi
	 * @param action mevcut durumda yap�lacak hareket
	 * @return hareketin sonucunda engel oldu�u, kurabiye yedi�ini veya hedefe ula�t���n� d�ner.
	 * Di�er durumda 0 d�ner.
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
	 * Bu robot i�in a�a� aramas� s�n�f�n� �a��r�r
	 * @param sim ��renmenin sim�le edilece�i sim�lat�r
	 */
	public void startSearching(Simulator sim)
	{
		//parametreler: sim�lat�r, robot
		GraphSearch ts = new GraphSearch(sim, this);
		ts.followOptimalPath();
	}
	/**
	 * Hareketler k�mesinden rastgele bir hareket se�er
	 * @return rastgele hareket
	 */
	public int chooseRandomAction() {
		return rand.nextInt(numOfActions);
	}
	/**
	 * Ortamda hedef durumuna e�it olmayan ve engele rastlamayan
	 * rastgele bir durum se�er.
	 * @return rastgele se�ilen durum {x,y} dizisi �eklinde.
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
	 * Robotun uzunlu�unu atar.
	 * @param length robotun uzunlu�u
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
	 * Robotun uzunlu�unu d�ner.
	 * @return robotun uzunlu�u
	 */
	public int getLength() {
		return length;
	}
	/**
	 * Robotun enini d�ner.
	 * @return robotun eni
	 */
	public int getWidth() {
		return width;
	}
	/**
	 * Robotun hareket k�mesindeki eleman say�s�n� d�ner.
	 * @return robotun hareket k�mesindeki eleman say�s�
	 */
	public int getNumOfActions() {
		return numOfActions;
	}
	/**
	 * Robotun sim�lat�rdeki x ve y pozisyonunu dizi olarak d�ner.
	 * @return
	 */
	public int[] getPosition(){
		return position;		
	}
	/**
	 * Robotun sim�lat�rdeki x ve y pozisyonunu dizi olarak atar.
	 * @param pos robotun x ve y pozisyonu
	 */
	public void setPosition(int[] pos) {
		position = pos;		
	}
	/**
	 * Robotun sim�lat�rdeki x ve y pozisyonunu atar.
	 * @param x robotun x pozisyonu
	 * @param y robotun y pozisyonu
	 */
	public void setPosition(int x, int y) {
		position[0] = x;
		position[1] = y;	
	}
	/**
	 * Verilen hareket anahtar�na tekab�l eden �ngilizce kelimeyi d�ner
	 * @param action hareket anahtar�
	 * @return hareket anahtar�na tekab�l eden �ngilizce kelime
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
