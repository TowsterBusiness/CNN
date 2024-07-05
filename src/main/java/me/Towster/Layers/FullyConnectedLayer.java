package me.Towster.Layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer {
    double[][] weights;
    double[] lastZ;
    double[] previousLayerLastZ;
    double learningRate = 0.01;

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
        previousLayerLastZ = vecData;
        double[] outData = new double[weights[0].length * weights.length];
        for (int outIndex = 0; outIndex < weights[0].length; outIndex++) {
            for (int inIndex = 0; inIndex < weights.length; inIndex++) {
                outData[outIndex] += vecData[inIndex] * weights[outIndex][inIndex];
            }
        }

        lastZ = outData;
        if (_nextLayer != null)
            return _nextLayer.feedForward(vectorToMatrix(outData));
        return vectorToMatrix(outData);
    }

    @Override
    public void backProp(List<double[][]> deltas) {
        double[] deltaVec = matrixToVector(deltas);
        double[] dZdXList = new double[weights[0].length];

        for (int i = 0; i < weights[0].length; i++) {
            double dLdX = 0;
            for (int j = 0; j < weights.length; j++) {
                double dRdZ = derivativeReLU(lastZ[j]);
                double dZdW = previousLayerLastZ[i];
                double dZdX = weights[i][j];
                weights[i][j] -= deltaVec[j] * dRdZ * dZdW * learningRate;
                dLdX += deltaVec[j] * dRdZ * dZdX;
            }
            dZdXList[i] = dLdX;
        }

        if (_previousLayer != null) {
            backProp(vectorToMatrix(dZdXList));
        }
    }
}
