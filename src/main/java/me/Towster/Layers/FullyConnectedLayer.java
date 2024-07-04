package me.Towster.Layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer {
    double[][] weights;

    @Override
    public void createRandomWeights(int inNum, int outNum, int seed) {
        weights = new double[outNum][inNum];
        Random randomizer = new Random(seed);
        for (int out = 0; out < outNum; out++) {
            for (int in = 0; in < inNum; in++) {
                weights[out][in] = randomizer.nextGaussian();
            }
        }
    }

    @Override
    public List<double[][]> feedForward(List<double[][]> dataIn) {
        double[] vecData = matrixToVector(dataIn);
        double[] outData = new double[weights[0].length * weights.length];
        for (int outIndex = 0; outIndex < weights[0].length; outIndex++) {
            for (int inIndex = 0; inIndex < weights.length; inIndex++) {
                outData[outIndex] += vecData[inIndex] * weights[outIndex][inIndex];
            }
        }

        List<double[][]> outList = new ArrayList<>();
        double[][] outArray = new double[1][outData.length];
        outArray[0] = outData;
        outList.add(outArray);
        return outList;
    }

    @Override
    public void backProp(List<double[][]> deltas) {

    }
}
