package me.Towster.Layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer {
    double[][] weights;
    double[] lastZ;
    double[] previousLayerLastZ;

    int inNum;
    int outNum;
    double learningRate = 0.01;

    public FullyConnectedLayer (int inNum, int outNum, double learningRate) {
        this.inNum = inNum;
        this.outNum = outNum;
        this.learningRate = learningRate;

        weights = new double[outNum][inNum];
    }

    @Override
    public void createRandomWeights(int seed) {
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
        double[] outData = new double[inNum * outNum];
        for (int outIndex = 0; outIndex < inNum; outIndex++) {
            for (int inIndex = 0; inIndex < outNum; inIndex++) {
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
        double[] dZdXList = new double[inNum];

        for (int i = 0; i < inNum; i++) {
            double dLdX = 0;
            for (int j = 0; j < outNum; j++) {
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
