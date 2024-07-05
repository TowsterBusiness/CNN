package me.Towster.Layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PoolingLayer extends Layer {
    int poolWidth, poolHeight;
    public PoolingLayer(int poolWidth, int poolHeight) {
        this.poolWidth = poolWidth;
        this.poolHeight = poolHeight;
    }

    @Override
    public void createRandomWeights(int seed) {

    }

    @Override
    public List<double[][]> feedForward(List<double[][]> dataIn) {
        List<double[][]> outList = new ArrayList<>();

        for (int depIndex = 0; depIndex < dataIn.size(); depIndex++) {

            double[][] outXY = new double[dataIn.getFirst()[0].length - poolHeight]
                    [dataIn.getFirst().length - poolWidth];
            for (int slideX = 0; slideX < dataIn.getFirst()[0].length - poolHeight; slideX++) {
                for (int slideY = 0; slideY < dataIn.getFirst().length - poolWidth; slideY++) {
                    double max = 0;
                    for (int yIndex = 0; yIndex < poolHeight; yIndex++) {
                        for (int xIndex = 0; xIndex < poolWidth; xIndex++) {
                            max = Math.max(dataIn.get(depIndex)[yIndex + slideY][xIndex + slideX], max);
                        }
                    }
                    outXY[slideY][slideX] = max;
                }
            }
            outList.add(outXY);
        }

        if (_nextLayer != null) {
            return _nextLayer.feedForward(outList);
        }
        return outList;
    }

    @Override
    public void backProp(List<double[][]> dataIn) {

    }
}
