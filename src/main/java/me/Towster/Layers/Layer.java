package me.Towster.Layers;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {
    protected Layer _previousLayer;
    protected Layer _nextLayer;

    public abstract void createRandomWeights(int row, int col, int seed);
    public abstract List<double[][]> feedForward(List<double[][]> dataIn);
    public abstract void backProp(List<double[][]> dataIn);

    public double ReLU(double a) {
        if (a < 0) return 0;
        else return a;
    }

    public double[] matrixToVector(List<double[][]> inData) {
        List<Double> outDataList = new ArrayList<>();
        for (double[][] xy : inData) {
            for (double[] row : xy) {
                for (double cell : row) {
                    outDataList.add(cell);
                }
            }
        }
        double[] outData = new double[outDataList.size()];
        int i = 0;
        for (double a : outDataList) {
            outData[i] = a;
            i++;
        }
        return outData;
    }

    public Layer get_previousLayer() {
        return _previousLayer;
    }

    public void set_previousLayer(Layer _previousLayer) {
        this._previousLayer = _previousLayer;
    }

    public Layer get_nextLayer() {
        return _nextLayer;
    }

    public void set_nextLayer(Layer _nextLayer) {
        this._nextLayer = _nextLayer;
    }
}
