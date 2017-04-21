import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * User: tpeng
 * Date: 6/22/12
 * Time: 11:01 PM
 * To change this template use File | Settings | File Templates.
 */
public class Logistic {

    /** the learning rate */
    private double rate;

    /** the weight to learn */
    private double[] weights;

    /** the number of iterations */
    private int ITERATIONS = 3000;

    public Logistic(int n) {
        this.rate = 0.0001;
        weights = new double[n];
    }

    private double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    //SGD
    public void train_GSD(List<Instance> instances) {
        for (int n=0; n<ITERATIONS; n++) {
            double lik = 0.0;
            for (int i=0; i<instances.size(); i++) {
                int[] x = instances.get(i).getX();
                double predicted = classify(x);
                int label = instances.get(i).getLabel();
                for (int j=0; j<weights.length; j++) {
                    weights[j] = weights[j] + rate * (label - predicted) * x[j];
                }
                // not necessary for learning
                lik += label * Math.log(classify(x)) + (1-label) * Math.log(1- classify(x));
            }
            System.out.println("iteration: " + n + " " + Arrays.toString(weights) + " mle: " + lik);
        }
    }
    
    //GD
    public void train_SD(List<Instance> instances) {
        for (int n=0; n<ITERATIONS; n++) {
           double[] gradients = new double[weights.length];
            //loop entire data set to cal gradient
            for (int i=0; i<instances.size(); i++) {
                int[] x = instances.get(i).getX();
                double predicted = classify(x);
                int label = instances.get(i).getLabel();
                for(int j = 0; j < x.length; j++){
                    gradients[j] +=  rate * (label - predicted) * x[j];
                }
            }
            
            //adjust weights
            for (int j=0; j<weights.length; j++) {
                weights[j] = weights[j] + gradients[j];
            }

            System.out.println("iteration: " + n + " " + Arrays.toString(weights));
        }
    }

    private double classify(int[] x) {
    	if (x.length != weights.length){
    		return 0;
    	}
    	
        double logit = .0;
        for (int i=0; i<weights.length;i++)  {
            logit += weights[i] * x[i];
        }
        return sigmoid(logit);
    }


    public static void main(String... args) throws FileNotFoundException {
        List<Instance> instances = DataSet.readDataSet("dataset.txt");
        int x_length = 0;
        if(instances.size() > 0)
        	 x_length = instances.get(0).getX().length;
        Logistic logistic = new Logistic(x_length); //set
        logistic.train_GSD(instances);
        int[] x = {2, 1, 1, 0, 1};
        System.out.println("prob(1|x) = " + logistic.classify(x));

        int[] x2 = {1, 0, 1, 0, 0};
        System.out.println("prob(1|x2) = " + logistic.classify(x2));
    }
}
