
import java.awt.Color;
import java.awt.Graphics;
import java.awt.event.WindowEvent;
import java.awt.event.WindowStateListener;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
/**
 * Bu s�n�f sadece ortam� sim�le edbilmek amac�yla olu�turulmu�tur. *
 */
public class Simulator extends JFrame {
	/** Bu simulat�r penceresinin geni�li�i*/
	private int width;
	/** Bu simulat�r penceresinin y�ksekli�i*/
	private int height;
	/** Bu simulat�r penceresinin y�ksekli�i*/
	private final  static int heightOffset = 20;
	/** Her bir kutucu�un b�y�kl��� */
	private int boxSize;
	/** Ortamda hareket eden robota ait nesne */
	private Robot robot;
	/** Sim�lat�r�n tamamen ba�tan olu�turulup olu�turulmayaca�� */
	boolean isFirst;
	/** Robotun sim�lat�rde bir �nceki konumu */
	int[] prevLocation;
	/** Robotu temsil eden resim*/
	ImageIcon robotPicture;
	/** ��renme robot hareket ederken mi yoksa arka planda m� olsun */
	boolean isOnline;
	/**
	 * Bu sim�lat�r�n bir nesnesini olu�turur.
	 * @param isOnline ��renmenin robot hareket ederken olup olmayaca��
	 * @param robot robot
	 */
	public Simulator(boolean isOnline, Robot robot){
		super("A* Search Demo");
		this.isOnline = isOnline;
		this.robot = robot;
		boxSize = 60;
		width  = Environment.getWidth()*boxSize;
		height = Environment.getHeight()*boxSize + heightOffset;
		setSize(width,height);
		setResizable(false);
		isFirst=true;
		robotPicture = new ImageIcon(getClass().getResource("cookie_monster.png"));
		robot.setWidth(robotPicture.getIconWidth());
		robot.setLength(robotPicture.getIconHeight());
		robot.setPosition(convertBoxToPixels(Environment.getStart()));
		//Arka plan� pembe renk olarak ayarla
		this.getContentPane().setBackground(new Color(255,236,236));
		//Pencere indirilip kald�r�ld���nda sim�lat�r�n tamamen ba�tan �izilmesi i�in
		this.addWindowStateListener(new WindowStateListener() {			
			@Override
			public void windowStateChanged(WindowEvent e) {
				// TODO Auto-generated method stub
				isFirst = true;
				repaint();				
			}
		});
		
		setDefaultCloseOperation(EXIT_ON_CLOSE);
		setVisible(true);
		robot.startSearching(this);
	}
	/**
	 * Ortamdaki sat�r-s�tun say�s�yla verilen durumu sim�lat�r i�in
	 * ende ve boydaki piksel say�s�yla ifade eder.
	 * @param box sat�r-s�tun say�s�yla verilen durum
	 * @return ende ve boydaki piksel say�s�yla ifade edilmi� durum
	 */
	public int[] convertBoxToPixels(int[] box) {
		int pixels[] = new int[2];
		pixels[0] = box[1]*boxSize+(boxSize-robot.getWidth())/2;
		pixels[1] = heightOffset+box[0]*boxSize+(boxSize-robot.getLength())/2;
		return pixels;
	}
	/**
	 * Bu sim�lat�r� �izdirmek i�in olu�turulmu�tur. Java taraf�ndan otomatik olarak,
	 * ya da repaint() fonksiyonu arac�l���yla �a��r�labilir.
	 */
	@Override
	public void paint(Graphics g) {
		//E�er sim�lat�r ilk olarak �izdirilecekse
		if(isFirst){
			super.paint(g);
			//Ortam� �izdir
			drawEnvironment(g);
			//Robotu ortama �izdir.
			drawRobot(g,robot.getPosition());
			isFirst = false;
		}
		//Sadece robotun konumu g�ncellenecekse
		else if(prevLocation != null){
			//Robotu �nceki konumdan temizle. Takip edilen yolu g�rmek i�in yoruma al�n�r.
			//repaintSubRegion(g,prevLocation);
			//Robotu yeni konumuna �izdir.
			drawRobot(g,robot.getPosition());
		}
	}	
	/**
	 * Verilen grafik objesine robot resmini �izer.
	 * @param g Bu sim�lat�r�n grafik objesi
	 * @param pos robotun konumu {x,y}
	 */
	public void drawRobot(Graphics g,int[] pos){
		robotPicture.paintIcon(this, g, pos[0], pos[1]);
		
	}
	/**
	 * Robotun sim�lat�r ortam�nda mevcut konumundan verilen duruma hareket
	 * ettirilmesini sa�lar. 
	 * @param next ortamdaki sat�r-s�tun say�s�na g�re verilen durum
	 */
	public void moveRobot(int[] next){
		//Durumu piksel cinsinden ifade et
	    next = convertBoxToPixels(next);
	    // Ne y�nde ne kadar ve hangi h�zda gidece�ini hesapla
	    int x_range = Math.abs(next[0]-robot.getPosition()[0]);
		int y_range = Math.abs(next[1]-robot.getPosition()[1]);
		int x_inc = (x_range!=0? (next[0]-robot.getPosition()[0])/x_range :0);
		int y_inc = (y_range!=0? (next[1]-robot.getPosition()[1])/y_range :0);
		
		//Hareket edilecek mesafe kadar robotu hareket ettir.
		for(int i=0; i<y_range || i<x_range;i++){
			try {
				//E�er robotun daha h�zl� hareket etmesi isteniyorsa bu de�er azalt�l�r.
				//Ama �ok azalt�l�rsa Java tekrar �izmek i�in h�z�na yeti�emeyebilir.
				Thread.sleep(10);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			prevLocation = robot.getPosition();
			robot.setPosition(robot.getPosition()[0]+x_inc,robot.getPosition()[1]+y_inc);
			repaint();			
		}
	}
	/**
	 * Robotun hareketi esnas�nda mevcut konumundan �nceki konumunda var olan
	 * robot resmi temizlenir, ortam� ifade eden ilgili renkteki pikseller
	 * grafik nesnesine �izilir. 
	 * @param g Bu sim�lat�r�n grafik nesnesi
	 * @param pos Robotun resminin temizlenece�i konum
	 */
	public void repaintSubRegion(Graphics g, int[] pos) {
		short[][] region = Environment.getEnvironment();
		int h_remainder, w_remainder, h_division, w_division;
		for(int i=(pos[1]<=0?0:pos[1]-1);i<(pos[1]+robot.getLength()>=height?height:pos[1]+robot.getLength()+1);i++)
			for(int j=(pos[0]<=0?0:pos[0]-1);j<(pos[0]+robot.getWidth()>=width?width:pos[0]+robot.getWidth()+1);j++){
				h_remainder = (i-heightOffset) % boxSize;
				w_remainder = j % boxSize;
				h_division = (i-heightOffset) / boxSize;
				w_division = j / boxSize;
				//kutucuklar aras� s�n�r b�lgesinin �art�
				if(h_remainder==0 || h_remainder==1 || w_remainder==0 || w_remainder==1)//border
					g.setColor(new Color(64,0,0));//gri
				//engellerin bulundu�u b�lgelerin �art�
				else if(region[h_division][w_division] == 1) //obstacle
					g.setColor(new Color(128,157,193));//mavi
				//Di�er durumlar gezilebilir alan� ifade eder.
				else //space
					g.setColor(new Color(255,236,236));//pembe
				g.fillRect(j, i, 1, 1);//Rengi ayarlanan pikseli boya
				
			}
	}
	/**
	 * Environment nesnesiyle tan�ml� ortam� pixellere d�kerek bu sim�lat�r�n grafik
	 * nesnesine �izdirir.
	 * @param g Bu sim�lat�r�n grafik nesnesi
	 */
	public void drawEnvironment(Graphics g){
		boolean once = true;
		short[][] region = Environment.getEnvironment();
		//Yatayda t�m durumlar i�in
		for(int i=0;i<Math.floor(height/boxSize);i++){
			//i sat�r�n�n yatay s�n�r �izgisini �izdir. 
			g.setColor(new Color(64,0,0));
			g.fillRect(0, heightOffset+i*boxSize, width, 2);
			//D��eyde t�m durumlar i�in
			for(int j=0;j<width/boxSize;j++){
				//j s�tununun d��ey s�n�r �izgisini bu d�ng�n�n sadece ilk 
				//ilk seferi i�in �izdir. 
				if(once){
					g.setColor(new Color(64,0,0));
					g.fillRect(j*boxSize, 0, 2, height);
				}
				//E�er durumda engel varsa bu durumu mavi renk olarak �izdir.
				if(region[i][j] == 1){
					g.setColor(new Color(128,157,193));
					g.fillRect(j*boxSize+2, heightOffset+i*boxSize+2, boxSize-2, boxSize-2);
				}
				//Kurabiyeleri �izdir.
				else if(region[i][j] == 2){ //space
					g.setColor(new Color(160,80,0));//kahverengi
					g.fillOval(j*boxSize+17, heightOffset+i*boxSize+17, boxSize-34, boxSize-34);
				}
			}
			//D��ey s�n�rlar i� d�ng�de sadece 1 kez �izdirilir.
			if(once)
				once = false;
		}
	}		
}
