package me.Towster.Layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PoolingLayer extends Layer {
    List<double[][]> weights;
    int filterWidth;
    int filterHeight;
    int depth;

    public PoolingLayer(int filterWidth, int filterHeight, int depth, int stepSize) {
        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.depth = depth;

        weights = new ArrayList<>();
        for (int depIndex = 0; depIndex < depth; depIndex ++) {
            double[][] filter = new double[filterHeight][filterWidth];
            weights.add(filter);
        }
    }

    @Override
    public void createRandomWeights(int seed) {
        Random randomizer = new Random(seed);
        for (double[][] xy : weights) {
            for (int y = 0; y < filterHeight; y++) {
                for (int x = 0; x < filterWidth; x++) {
                    xy[y][x] = randomizer.nextGaussian();
                }
            }
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
