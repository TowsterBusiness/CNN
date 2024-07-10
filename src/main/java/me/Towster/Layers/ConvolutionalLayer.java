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
    double learningRate = 0.01;
    List<double[][]> pastZ;
    List<double[][]> pastA;

    public ConvolutionalLayer(int filterWidth, int filterHeight, int inDepth, int outDepth, double learningRate) {
        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.inDepth = inDepth;
        this.outDepth = outDepth;
        this.learningRate = learningRate;

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
        //Check if dataIn has any data changes
        pastA = dataIn;
        for (int dataInPointer = 0; dataInPointer < dataIn.size(); dataInPointer++) {
            for (int outWeightIndex = 0; outWeightIndex < outDepth; outDepth++) {
                double[][] dataInXY = dataIn.get(dataInPointer);
                double[][] outXY = new double[dataInXY.length - filterHeight][dataInXY[0].length - filterHeight];
                for (int yPointer = 0; yPointer < dataInXY.length - filterHeight; yPointer++) {
                    for (int xPointer = 0; xPointer < dataInXY[0].length - filterHeight; xPointer++) {
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

        pastZ = outData;

        if (_nextLayer != null) {
            return _nextLayer.feedForward(outData);
        }
        return outData;
    }

    @Override
    public void backProp(List<double[][]> dataIn) {
        List<double[][]> dataOut = new ArrayList<>();

        for (int depthInIndex = 0; depthInIndex < inDepth; depthInIndex++) {
            double[][] dataOutLayer = new double[dataIn.getFirst()[0].length][dataIn.getFirst().length];

            for (int depthOutIndex = 0; depthOutIndex < dataIn.size(); depthOutIndex++) {
                int depthIndex = depthInIndex * depthOutIndex;
                double[][] dataInLayer = dataIn.get(depthIndex);
                double[][] filterLayer = weights.get(depthInIndex).get(depthOutIndex);


                int filterMovesX = dataInLayer[0].length - filterLayer[0].length;
                int filterMovesY = dataInLayer.length - filterLayer.length;

                for (int filterYIndex = 0; filterYIndex < filterLayer.length; filterYIndex++) {
                    for (int filterXIndex = 0; filterXIndex < filterLayer[0].length; filterXIndex++) {
                        //It's a little scary using variables but in principle it should work
                        for (int moveY = 0; moveY < filterMovesY; moveY++) {
                            for (int moveX = 0; moveX < filterMovesX; moveX++) {
                                double dLdY = dataInLayer[moveY][moveX];
                                double dYdZ = derivativeReLU(pastZ.get(depthIndex)[moveY][moveX]);
                                double dZdW = pastA.get(depthInIndex)[moveY + filterYIndex][moveX + filterXIndex];
                                double dZdA = filterLayer[filterYIndex][filterXIndex];

                                filterLayer[filterYIndex][filterXIndex] -= dLdY * dYdZ * dZdW * learningRate;
                                dataOutLayer[moveY + filterYIndex][moveX + filterXIndex] += dLdY * dYdZ * dZdA;
                            }
                        }
                    }
                }
                dataOut.add(dataInLayer);

            }

            dataOut.add(dataOutLayer);
        }

        if (_previousLayer != null) {
            backProp(dataOut);
        }
    }
}
