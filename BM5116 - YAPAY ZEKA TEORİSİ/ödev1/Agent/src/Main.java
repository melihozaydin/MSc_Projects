/**
 * Uygulamay� ba�latan ana s�n�f
 * 
 */
public class Main {

	public static void main(String[] args) {
		// Ortam� olu�tur.
		Environment e = new Environment();
		//Robotu olu�tur. 
		Robot robot = new Robot();
		//Sim�lat�r� �a��r
		new Simulator(false, robot);
		e.printEnvironment();

	}

}
