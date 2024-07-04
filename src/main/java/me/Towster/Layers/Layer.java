package me.Towster.Layers;

import java.util.List;

public abstract class Layer {
    public abstract double[][] feedForward(double[][] dataIn);
    public abstract List<double[][]> feedForward(List<double[][]> dataIn);

    public abstract double[][] backProp(double[][] dataIn);
    public abstract List<double[][]> backProp(List<double[][]> dataIn);
}
