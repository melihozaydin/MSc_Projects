
import java.awt.Color;
import java.awt.Graphics;
import java.awt.event.WindowEvent;
import java.awt.event.WindowStateListener;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
/**
 * Bu sýnýf sadece ortamý simüle edbilmek amacýyla oluþturulmuþtur. *
 */
public class Simulator extends JFrame {
	/** Bu simulatör penceresinin geniþliði*/
	private int width;
	/** Bu simulatör penceresinin yüksekliði*/
	private int height;
	/** Bu simulatör penceresinin yüksekliði*/
	private final  static int heightOffset = 20;
	/** Her bir kutucuðun büyüklüðü */
	private int boxSize;
	/** Ortamda hareket eden robota ait nesne */
	private Robot robot;
	/** Simülatörün tamamen baþtan oluþturulup oluþturulmayacaðý */
	boolean isFirst;
	/** Robotun simülatörde bir önceki konumu */
	int[] prevLocation;
	/** Robotu temsil eden resim*/
	ImageIcon robotPicture;
	/** Öðrenme robot hareket ederken mi yoksa arka planda mý olsun */
	boolean isOnline;
	/**
	 * Bu simülatörün bir nesnesini oluþturur.
	 * @param isOnline öðrenmenin robot hareket ederken olup olmayacaðý
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
		//Arka planý pembe renk olarak ayarla
		this.getContentPane().setBackground(new Color(255,236,236));
		//Pencere indirilip kaldýrýldýðýnda simülatörün tamamen baþtan çizilmesi için
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
	 * Ortamdaki satýr-sütun sayýsýyla verilen durumu simülatör için
	 * ende ve boydaki piksel sayýsýyla ifade eder.
	 * @param box satýr-sütun sayýsýyla verilen durum
	 * @return ende ve boydaki piksel sayýsýyla ifade edilmiþ durum
	 */
	public int[] convertBoxToPixels(int[] box) {
		int pixels[] = new int[2];
		pixels[0] = box[1]*boxSize+(boxSize-robot.getWidth())/2;
		pixels[1] = heightOffset+box[0]*boxSize+(boxSize-robot.getLength())/2;
		return pixels;
	}
	/**
	 * Bu simülatörü çizdirmek için oluþturulmuþtur. Java tarafýndan otomatik olarak,
	 * ya da repaint() fonksiyonu aracýlýðýyla çaðýrýlabilir.
	 */
	@Override
	public void paint(Graphics g) {
		//Eðer simülatör ilk olarak çizdirilecekse
		if(isFirst){
			super.paint(g);
			//Ortamý çizdir
			drawEnvironment(g);
			//Robotu ortama çizdir.
			drawRobot(g,robot.getPosition());
			isFirst = false;
		}
		//Sadece robotun konumu güncellenecekse
		else if(prevLocation != null){
			//Robotu önceki konumdan temizle. Takip edilen yolu görmek için yoruma alýnýr.
			//repaintSubRegion(g,prevLocation);
			//Robotu yeni konumuna çizdir.
			drawRobot(g,robot.getPosition());
		}
	}	
	/**
	 * Verilen grafik objesine robot resmini çizer.
	 * @param g Bu simülatörün grafik objesi
	 * @param pos robotun konumu {x,y}
	 */
	public void drawRobot(Graphics g,int[] pos){
		robotPicture.paintIcon(this, g, pos[0], pos[1]);
		
	}
	/**
	 * Robotun simülatör ortamýnda mevcut konumundan verilen duruma hareket
	 * ettirilmesini saðlar. 
	 * @param next ortamdaki satýr-sütun sayýsýna göre verilen durum
	 */
	public void moveRobot(int[] next){
		//Durumu piksel cinsinden ifade et
	    next = convertBoxToPixels(next);
	    // Ne yönde ne kadar ve hangi hýzda gideceðini hesapla
	    int x_range = Math.abs(next[0]-robot.getPosition()[0]);
		int y_range = Math.abs(next[1]-robot.getPosition()[1]);
		int x_inc = (x_range!=0? (next[0]-robot.getPosition()[0])/x_range :0);
		int y_inc = (y_range!=0? (next[1]-robot.getPosition()[1])/y_range :0);
		
		//Hareket edilecek mesafe kadar robotu hareket ettir.
		for(int i=0; i<y_range || i<x_range;i++){
			try {
				//Eðer robotun daha hýzlý hareket etmesi isteniyorsa bu deðer azaltýlýr.
				//Ama çok azaltýlýrsa Java tekrar çizmek için hýzýna yetiþemeyebilir.
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
	 * Robotun hareketi esnasýnda mevcut konumundan önceki konumunda var olan
	 * robot resmi temizlenir, ortamý ifade eden ilgili renkteki pikseller
	 * grafik nesnesine çizilir. 
	 * @param g Bu simülatörün grafik nesnesi
	 * @param pos Robotun resminin temizleneceði konum
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
				//kutucuklar arasý sýnýr bölgesinin þartý
				if(h_remainder==0 || h_remainder==1 || w_remainder==0 || w_remainder==1)//border
					g.setColor(new Color(64,0,0));//gri
				//engellerin bulunduðu bölgelerin þartý
				else if(region[h_division][w_division] == 1) //obstacle
					g.setColor(new Color(128,157,193));//mavi
				//Diðer durumlar gezilebilir alaný ifade eder.
				else //space
					g.setColor(new Color(255,236,236));//pembe
				g.fillRect(j, i, 1, 1);//Rengi ayarlanan pikseli boya
				
			}
	}
	/**
	 * Environment nesnesiyle tanýmlý ortamý pixellere dökerek bu simülatörün grafik
	 * nesnesine çizdirir.
	 * @param g Bu simülatörün grafik nesnesi
	 */
	public void drawEnvironment(Graphics g){
		boolean once = true;
		short[][] region = Environment.getEnvironment();
		//Yatayda tüm durumlar için
		for(int i=0;i<Math.floor(height/boxSize);i++){
			//i satýrýnýn yatay sýnýr çizgisini çizdir. 
			g.setColor(new Color(64,0,0));
			g.fillRect(0, heightOffset+i*boxSize, width, 2);
			//Düþeyde tüm durumlar için
			for(int j=0;j<width/boxSize;j++){
				//j sütununun düþey sýnýr çizgisini bu döngünün sadece ilk 
				//ilk seferi için çizdir. 
				if(once){
					g.setColor(new Color(64,0,0));
					g.fillRect(j*boxSize, 0, 2, height);
				}
				//Eðer durumda engel varsa bu durumu mavi renk olarak çizdir.
				if(region[i][j] == 1){
					g.setColor(new Color(128,157,193));
					g.fillRect(j*boxSize+2, heightOffset+i*boxSize+2, boxSize-2, boxSize-2);
				}
				//Kurabiyeleri çizdir.
				else if(region[i][j] == 2){ //space
					g.setColor(new Color(160,80,0));//kahverengi
					g.fillOval(j*boxSize+17, heightOffset+i*boxSize+17, boxSize-34, boxSize-34);
				}
			}
			//Düþey sýnýrlar iç döngüde sadece 1 kez çizdirilir.
			if(once)
				once = false;
		}
	}		
}
