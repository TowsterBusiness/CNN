package me.Towster.Layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvolutionalLayer extends Layer {
    List<double[][]> weights;
    int filterWidth;
    int filterHeight;
    int depth;

    public ConvolutionalLayer(int filterWidth, int filterHeight, int depth) {
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
        List<double[][]> outData = new ArrayList<>();
        for (int dataInPointer = 0; dataInPointer < dataIn.size(); dataInPointer++) {
            double[][] dataInXY = dataIn.get(dataInPointer);
            double[][] outXY = new double[dataInXY[0].length - filterHeight][dataInXY.length - filterHeight];
            for (int yPointer = 0; yPointer < dataInXY[0].length - filterHeight; yPointer ++) {
                for (int xPointer = 0; xPointer < dataInXY.length - filterHeight; xPointer ++) {
                    double outSum = 0;
                    for (int filterY = 0; filterY < filterHeight; filterY++) {
                        for (int filterX = 0; filterX < filterWidth; filterX++) {
                            outSum += dataInXY[yPointer + filterY][xPointer + filterX]
                                    * weights.get(dataInPointer)[filterY][filterX];
                        }
                    }
                    outSum /= filterHeight * filterWidth;
                    outXY[yPointer][xPointer] = ReLU(outSum);
                }
            }
            outData.add(outXY);
        }

        if (_nextLayer != null) {
            return _nextLayer.feedForward(outData);
        }
        return outData;
    }

    @Override
    public void backProp(List<double[][]> dataIn) {

    }
}
