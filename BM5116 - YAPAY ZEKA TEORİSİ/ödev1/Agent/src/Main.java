/**
 * Uygulamayý baþlatan ana sýnýf
 * 
 */
public class Main {

	public static void main(String[] args) {
		// Ortamý oluþtur.
		Environment e = new Environment();
		//Robotu oluþtur. 
		Robot robot = new Robot();
		//Simülatörü çaðýr
		new Simulator(false, robot);
		e.printEnvironment();

	}

}
