package me.Towster;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ImageProcessing {
    public static int getImageWidth() {
        return imageWidth;
    }

    public static int getImageHeight() {
        return imageHeight;
    }

    private static int imageWidth = 28;
    private static int imageHeight = 28;

    public static List<Image> read(String filePath, int count) throws FileNotFoundException {
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        List<Image> imageList = new ArrayList<>();

        if (count == -1) {
            count = Integer.MAX_VALUE;
        }

        try {
            String imageLine = reader.readLine();

            int limitCounter = 0;
            while (imageLine != null && limitCounter <= count) {
                float[][] imageData = new float[imageHeight][imageWidth];
                int label = 0;
                String[] stringData = imageLine.split(",");
                int i = 0;
                for (String s : stringData) {
                    int sData = Integer.parseInt(s);
                    if (i == 0) {
                        label = sData;
                    } else {
                        imageData[(i-1) / imageHeight][(i-1) % imageWidth] = sData;
                    }
                    i++;
                }
                imageList.add(new Image(imageData, label));
                imageLine = reader.readLine();
            }
        } catch (IOException e) {
            System.out.println("Error!");
        }

        return imageList;
    }
}
