package me.Towster.Layers;

import java.util.List;

public abstract class Layer {
    protected Layer _previousLayer;
    protected Layer _nextLayer;

    public abstract void createRandomWeights(int row, int col, int dep, int seed);
    public abstract List<double[][]> feedForward(List<double[][]> dataIn);

    public abstract void backProp(List<double[][]> dataIn);

    public float ReLU(float a) {
        if (a < 0) return 0;
        else return a;
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
