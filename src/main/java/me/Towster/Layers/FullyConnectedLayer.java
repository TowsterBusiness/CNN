package me.Towster.Layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer {
    List<float[][]> weights;

    @Override
    public void createRandomWeights(int rowNum, int colNum, int depthNum, int seed) {
        weights = new ArrayList<>();
        Random randomizer = new Random(seed);
        for (int dep = 0; dep < depthNum; dep++) {
            float[][] xy = new float[colNum][rowNum];
            for (int col = 0; col < colNum; col++) {
                for (int row = 0; row < rowNum; row++) {
                    xy[col][row] = (float) randomizer.nextGaussian();
                }
            }
            weights.add(xy);
        }
    }

    @Override
    public List<double[][]> feedForward(List<double[][]> dataIn) {
        return List.of();
    }

    @Override
    public void backProp(List<double[][]> dataIn) {

    }
}
