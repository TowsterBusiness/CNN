package me.Towster.Layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PoolingLayer extends Layer {
    int poolWidth, poolHeight;
    List<int[][][]> lastZ;
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
        lastZ = new ArrayList<>();

        for (int depIndex = 0; depIndex < dataIn.size(); depIndex++) {
            int[][][] cacheLayer = new int[dataIn.getFirst()[0].length][dataIn.getFirst().length][2];
            // indexed by y, x, [x,y]
            double[][] outXY = new double[dataIn.getFirst().length - poolHeight]
                    [dataIn.getFirst()[0].length - poolWidth];
            for (int slideX = 0; slideX < dataIn.getFirst()[0].length - poolHeight; slideX++) {
                for (int slideY = 0; slideY < dataIn.getFirst().length - poolWidth; slideY++) {
                    double max = 0;
                    int[] maxXY = new int[2];
                    for (int yIndex = 0; yIndex < poolHeight; yIndex++) {
                        for (int xIndex = 0; xIndex < poolWidth; xIndex++) {
                            double val = dataIn.get(depIndex)[yIndex + slideY][xIndex + slideX];
                            if (val >= max) {
                                max = val;
                                maxXY[0] = xIndex;
                                maxXY[1] = yIndex;
                            }
                        }
                    }
                    cacheLayer[slideY][slideX] = maxXY;
                    outXY[slideY][slideX] = max;
                }
            }
            outList.add(outXY);
            lastZ.add(cacheLayer);
        }

        if (_nextLayer != null) {
            return _nextLayer.feedForward(outList);
        }
        return outList;
    }

    @Override
    public void backProp(List<double[][]> dataIn) {
        List<double[][]> dataOut = new ArrayList<>();
        for (int depthIndex = 0; depthIndex < dataIn.size(); depthIndex++) {
            double[][] layer = new double[dataIn.getFirst().length * poolHeight][dataIn.getFirst()[0].length * poolWidth];
            for (int yIndex = 0; yIndex < dataIn.getFirst().length; yIndex++) {
                for (int xIndex = 0; xIndex < dataIn.getFirst()[0].length; xIndex++) {
                    layer[lastZ.get(depthIndex)[yIndex][xIndex][1]][lastZ.get(depthIndex)[yIndex][xIndex][0]]
                            += dataIn.get(depthIndex)[yIndex][xIndex];
                }
            }
            dataOut.add(layer);
        }

        if (_previousLayer != null) {
            backProp(dataOut);
        }
    }
}
