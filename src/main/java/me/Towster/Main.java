package me.Towster;

import java.io.IOException;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
        List<Image> imageList = ImageProcessing.read("src/main/resources/mnist_train.csv", 5);

        for (float[] row : imageList.getFirst().getData()) {
            for (float col : row) {
                System.out.print(col + ",");
            }
            System.out.println();
        }

    }
}