package me.Towster;

public class Image {

    private float[][] data;
    private int label;

    public Image(float[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    public float[][] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }
}
