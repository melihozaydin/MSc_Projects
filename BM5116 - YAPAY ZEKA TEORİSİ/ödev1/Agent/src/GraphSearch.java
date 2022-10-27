import java.util.ArrayList;
import java.util.Random;
/**
 * Bu sýnýf düzgün bedel aramasý (uniform cost search) veya A* aramasýný kullanarak
 * aðaç aramasý algoritmasýný gerçekler
 */
public class GraphSearch{
		/**Öðrenmenin simüle edileceði simülatör*/
		private Simulator simulator;
		/**Ortamdaki robot*/
		private Robot robot;
		/** Arama sonucunda ajaný baþlangýç konumundan hedefe götüren yol.
		 * Burada sadece gösterim amaçlý atama yapýlmýþtýr. 
		 * Bu deðiþkenin deðerlerini normalde arama sonucu belirleyecek.*/
		private int[] optimalPath = {1,1,1,1,3,3,3,3,3,3,3,3,1,1,1,1,1,2,2,2,2,2,2,2,2};
		/**
		 * Bu TreeSearch sýnýfýnýn bir nesnesini oluþturur. Algoritmanýn gerçekleþtirilebilmesi
		 * için gerekli deðiþkenleri tanýmlar.		
		 * @param sim Öðrenmenin simüle edileceði simülatör
		 * @param robot Ortamdaki robot
		 */
		public GraphSearch(Simulator sim, Robot robot){
			simulator = sim;
			this.robot = robot;		
			
		}
		
				
		/**
		 * Hesaplanan en iyi patikayý kullanarak verilen
		 * baþlangýç konumundan hedef konumuna robotun Simülatörde hareketini saðlar
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