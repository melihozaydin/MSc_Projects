import java.util.ArrayList;
import java.util.Random;
/**
 * Bu s�n�f d�zg�n bedel aramas� (uniform cost search) veya A* aramas�n� kullanarak
 * a�a� aramas� algoritmas�n� ger�ekler
 */
public class GraphSearch{
		/**��renmenin sim�le edilece�i sim�lat�r*/
		private Simulator simulator;
		/**Ortamdaki robot*/
		private Robot robot;
		/** Arama sonucunda ajan� ba�lang�� konumundan hedefe g�t�ren yol.
		 * Burada sadece g�sterim ama�l� atama yap�lm��t�r. 
		 * Bu de�i�kenin de�erlerini normalde arama sonucu belirleyecek.*/
		private int[] optimalPath = {1,1,1,1,3,3,3,3,3,3,3,3,1,1,1,1,1,2,2,2,2,2,2,2,2};
		/**
		 * Bu TreeSearch s�n�f�n�n bir nesnesini olu�turur. Algoritman�n ger�ekle�tirilebilmesi
		 * i�in gerekli de�i�kenleri tan�mlar.		
		 * @param sim ��renmenin sim�le edilece�i sim�lat�r
		 * @param robot Ortamdaki robot
		 */
		public GraphSearch(Simulator sim, Robot robot){
			simulator = sim;
			this.robot = robot;		
			
		}
		
				
		/**
		 * Hesaplanan en iyi patikay� kullanarak verilen
		 * ba�lang�� konumundan hedef konumuna robotun Sim�lat�rde hareketini sa�lar
		 */
		public void followOptimalPath(){
			
			int steps = 0;
			int[]state = new int[2];
			int action;
			state[0] = Environment.getStart()[0];
			state[1] = Environment.getStart()[1];
			System.out.println("Given start state, I will follow the optimal path now.\n");
			robot.setPosition(simulator.convertBoxToPixels(state));	
			for(int i=0; i<optimalPath.length; i++){
				action = optimalPath[steps];
				steps++;
				robot.act(state, action);
				simulator.moveRobot(state);			
			}			
			System.out.println("Optimal path is followed in "+steps+" number of steps.");				
		}
	}