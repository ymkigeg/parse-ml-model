import java.io.*;
import java.util.InputMismatchException;
import java.util.List;
import java.util.Map;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class FMModel {
    private int featureSize;
    private int k;
    private double bias;
    private double [] coefficients;
    private double [][] embeddings;
    private String dt;

    private FMModel(int featureSize) {
        this.k = 0;
        this.featureSize = featureSize;
        this.coefficients = new double[featureSize];
        // this.embeddings = new double[featureSize][k];
        this.bias = 0;
    }

    public static FMModel createByList(List<String> rows, int featureSize) {
        if (CollectionUtils.isEmpty(rows) || rows.size() <=  featureSize) {
            return null;
        }

        FMModel fmModel = new FMModel(featureSize);

        for (String str : rows) {
            try {
                parseRow(str, fmModel);
            } catch (Exception e) {
                System.out.println(str);
            }
        }
        return fmModel;
    }

    private static int parseRow(String str, FMModel fmModel) {
        if (StringUtils.isEmpty(str)) {
            return -1;
        }

        String[] colArray = str.trim().split(":");
        if ("dt".equals(colArray[0])) {
            fmModel.dt = colArray[1].trim();
        } else if ("bias".equals(colArray[0])) {
            fmModel.bias = Double.valueOf(colArray[1].trim());
        } else if (colArray[0].startsWith("i_")) {
            int index = Integer.valueOf(colArray[0].split("_")[1].trim());
            fmModel.checkFeautreSize(index);

            fmModel.coefficients[index] = Double.valueOf(colArray[1]);
        } else if (colArray[0].startsWith("v_")) {
            int index = Integer.valueOf(colArray[0].split("_")[1].trim());
            String[] embs = colArray[1].trim().split(" ");
            fmModel.checkFeautreSize(index);
            if (0 == fmModel.k) {
                fmModel.k = embs.length;
                fmModel.embeddings = new double[fmModel.featureSize][fmModel.k];
            }

            fmModel.checkEmbeddingSize(embs.length);

            double[] emb = new double[embs.length];
            for (int i = 0; i < embs.length; i++) {
                emb[i] = Double.valueOf(embs[i]);
            }
            fmModel.embeddings[index] = emb;
        } else {
            return -2;
        }
        return 0;
    }

    public static FMModel createByText(InputStream in, int featureSize) throws IOException, InputMismatchException {
        FMModel fmModel = new FMModel(featureSize);
        return create(in, fmModel);
    }

    public int getFeatureSize() { return this.featureSize; }
    public int getEmbeddingSize() { return this.k; }
    public String getDt() { return this.dt; }

    public double [] getFeatureEmbedding(int featId) {
        return this.embeddings[featId];
    }

    public double predict(Map<Integer, Double> data) {
        double [] coefficients = new double[data.size()];
        double [][] embeddings = new double[data.size()][this.k];
        double [] featValue = new double[data.size()];

        int i = 0;
        for (Integer key : data.keySet()) {
            coefficients[i] = this.coefficients[key];
            embeddings[i] = this.embeddings[key];
            featValue[i] = data.get(key);
            i += 1;
        }

        double firstOrder = calcFirstOrder(featValue, coefficients);
        double secondOrder = calcSecondOrder(featValue, embeddings);

        return 1 / (1 + Math.exp(-(this.bias + firstOrder + secondOrder)));
    }

    public double predict(int [] featIndex, double [] featValue) throws Exception{
        if (featIndex.length != featValue.length) {
            throw new Exception("fm input index's length must equals to value's length");
        }

        double [] coefficients = new double[featIndex.length];
        double [][] embeddings = new double[featIndex.length][this.k];
        for (int i = 0; i < featIndex.length; i++) {
            coefficients[i] = this.coefficients[featIndex[i]];
            embeddings[i] = this.embeddings[featIndex[i]];
        }

        double firstOrder = calcFirstOrder(featValue, coefficients);
        double secondOrder = calcSecondOrder(featValue, embeddings);

        return 1 / (1 + Math.exp(-(this.bias + firstOrder + secondOrder)));
    }

    private double calcFirstOrder(double [] featValue, double[] coefficients) {
        RealVector vectorA = new ArrayRealVector(featValue, false);
        RealVector vectorB = new ArrayRealVector(coefficients, false);
        return vectorA.dotProduct(vectorB);
    }

    private double calcSecondOrder(double [] featValue, double[][] embeddings) {
        RealMatrix matrix = new Array2DRowRealMatrix(embeddings, false);
        RealVector featVector = new ArrayRealVector(featValue, false);
        double secondOrder = 0;
        for (int f  = 0; f < this.k; f++) {
            RealVector vif = matrix.getColumnVector(f);
            double dot = featVector.dotProduct(vif);
            double sumSquare = dot * dot;

            double squareSum = 0;
            for (int i = 0; i < featValue.length; i++) {
                squareSum += featValue[i] * featValue[i] * vif.getEntry(i) * vif.getEntry(i);
            }

            secondOrder += (sumSquare - squareSum);
        }
        return 0.5 * secondOrder;
    }

    private static FMModel create(InputStream in, FMModel fmModel) throws IOException, InputMismatchException {
        InputStreamReader input = new InputStreamReader(in);
        BufferedReader bf = new BufferedReader(input);
        // 按行读取字符串
        String str;
        // bf.readLine();
        while ((str = bf.readLine()) != null) {
            try {
                parseRow(str, fmModel);
            } catch (Exception e) {
                System.out.println(str);
            }
        }

        return fmModel;
    }

    private void checkEmbeddingSize(int k) throws InputMismatchException{
        if (k != this.k) {
            throw new InputMismatchException("the embedding size is mismatch!");
        }
    }

    private void checkFeautreSize(int index) throws InputMismatchException{
        if (index >= this.featureSize) {
            throw new InputMismatchException("feat_id from model is greater than feature_size");
        }
    }
}

