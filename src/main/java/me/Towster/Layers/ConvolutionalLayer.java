package me.Towster.Layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvolutionalLayer extends Layer {
    List<List<double[][]>> weights;
    int filterWidth;
    int filterHeight;
    int inDepth;
    int outDepth;

    public ConvolutionalLayer(int filterWidth, int filterHeight, int inDepth, int outDepth) {
        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.inDepth = inDepth;
        this.outDepth = outDepth;

        weights = new ArrayList<>();
        for (int depIndex = 0; depIndex < inDepth; depIndex ++) {
            List<double[][]> outWeights = new ArrayList<>();
            for (int outDepIndex = 0; outDepIndex < outDepth; outDepIndex++ ) {
                double[][] filter = new double[filterHeight][filterWidth];
                outWeights.add(filter);
            }
            weights.add(outWeights);
        }
    }

    @Override
    public void createRandomWeights(int seed) {
        Random randomizer = new Random(seed);
        for (int inWeightIndex = 0; inWeightIndex < inDepth; inDepth++) {
            for (int outWeightIndex = 0; outWeightIndex < outDepth; outDepth++) {
                    for (int y = 0; y < filterHeight; y++) {
                        for (int x = 0; x < filterWidth; x++) {
                            weights.get(inWeightIndex)
                                    .get(outWeightIndex)[y][x] = randomizer.nextGaussian();
                        }
                    }
            }
        }

    }

    @Override
    public List<double[][]> feedForward(List<double[][]> dataIn) {
        List<double[][]> outData = new ArrayList<>();
        for (int dataInPointer = 0; dataInPointer < dataIn.size(); dataInPointer++) {
            for (int outWeightIndex = 0; outWeightIndex < outDepth; outDepth++) {
                double[][] dataInXY = dataIn.get(dataInPointer);
                double[][] outXY = new double[dataInXY[0].length - filterHeight][dataInXY.length - filterHeight];
                for (int yPointer = 0; yPointer < dataInXY[0].length - filterHeight; yPointer++) {
                    for (int xPointer = 0; xPointer < dataInXY.length - filterHeight; xPointer++) {
                        double outSum = 0;
                        for (int filterY = 0; filterY < filterHeight; filterY++) {
                            for (int filterX = 0; filterX < filterWidth; filterX++) {
                                outSum += dataInXY[yPointer + filterY][xPointer + filterX]
                                        * weights.get(outWeightIndex)
                                        .get(dataInPointer)[filterY][filterX];
                            }
                        }
                        outXY[yPointer][xPointer] = ReLU(outSum);
                    }
                }
                outData.add(outXY);
            }

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
