import java.io.IOException;
import java.io.PrintStream;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        double[] input = new double[] {1, 1};
        double[] target = new double[] {1};
        int[] layerSize = new int[] {input.length, 16, 16, target.length};

        //create a new network
        Network network = new Network(input, target, layerSize);
        //network.train(300);

        new Network("network1.txt").predict(new double[] {0.6, 0.1});

        //network.importSample("sample1.txt");
        //network.trainSample(5000);


/*
        //ask for sample one by one infinitely
        while(true) {
            double[] newInput = new double[input.length];
            double[] newTarget = new double[target.length];



            System.out.println("Input New Sample:");
            //accept new sample (new input and new target)
            for (int i = 0; i < input.length; i++) {
                System.out.print("newInput[" + i + "]: ");
                newInput[i] = scanner.nextDouble();
            }
            for (int i = 0; i < target.length; i++) {
                System.out.print("newTarget[" + i + "]: ");
                newTarget[i] = scanner.nextDouble();
            }

            try {
                network.acceptSample(newInput, newTarget);

                network.train(300);

            } catch(Exception e) {
                System.out.println("Input again...");
            }
        }
*/

    }
}
