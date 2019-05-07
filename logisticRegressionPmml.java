package com.ymkigeg.ml.pmml;


import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;

import java.io.File;
import java.io.InputStream;
import java.net.URL;
import java.util.*;
import org.apache.commons.math3.linear.ArrayRealVector;


public class LogisticRegressionModel {

    private String modelName;
    private String algorithmName;
    private String functionName;
    private String normalizationMethod;

    private Integer numberOfFields;
    private String targetCategory;

    private double intercept;
    private double [] coefficients;


    private LogisticRegressionModel() {}

    public static LogisticRegressionModel createByPmml(String modelFile) throws DocumentException{
        SAXReader reader = new SAXReader();
        Document document = reader.read(new File(modelFile));
        return create(document);
    }

    public static LogisticRegressionModel createByPmml(InputStream in) throws DocumentException{
        SAXReader reader = new SAXReader();
        Document document = reader.read(in);
        return create(document);
    }

    public static LogisticRegressionModel createByPmml(File file) throws DocumentException{
        SAXReader reader = new SAXReader();
        Document document = reader.read(file);
        return create(document);
    }

    public static LogisticRegressionModel createByPmml(URL url) throws DocumentException{
        SAXReader reader = new SAXReader();
        Document document = reader.read(url);
        return create(document);
    }

    public int getNumberOfFields() { return this.numberOfFields; }

    public double predict(double [] data){
        // TODO 这里可以将sigmoid函数存成字典，这样就减少计算量
        return 1 / (1 + Math.exp(-(vectorDot(coefficients, data) + intercept)));
    }

    private static LogisticRegressionModel create(Document document) {
        LogisticRegressionModel lrModel = new LogisticRegressionModel();
        Element root = document.getRootElement();
        lrModel.numberOfFields = Integer.valueOf(root.element("DataDictionary").attribute("numberOfFields").getValue())-1;
        lrModel.parseAttributes(root.element("RegressionModel"));

        Element modelElement = LogisticRegressionModel.findModelElement(root.element("RegressionModel"));
        lrModel.intercept = Double.valueOf(modelElement.attribute("intercept").getValue());
        lrModel.targetCategory="1";
        List<Element> coefficientElements = modelElement.elements("NumericPredictor");
        if (!lrModel.numberOfFields.equals(coefficientElements.size())) {
            // TODO 抛出异常
            return null;
        }

        lrModel.coefficients = new double[lrModel.numberOfFields];
        for (Element element : coefficientElements) {
            double coef = Double.valueOf(element.attributeValue("coefficient"));
            int index = Integer.valueOf(element.attributeValue("name"));
            lrModel.coefficients[index] = coef;
        }
        return lrModel;
    }

    private double vectorDot(double [] a, double [] b) {
        ArrayRealVector vectorA = new ArrayRealVector(a, false);
        ArrayRealVector vectorB = new ArrayRealVector(b, false);
        return vectorA.dotProduct(vectorB);
//        double result = 0.0;
//        for (int i=0; i<a.length; i++) {
//            result += a[i] * b[i];
//        }
//        return result;
    }

    public String getModelName() { return this.modelName; }
    public String getAlgorithmName() { return this.algorithmName; }
    public String getFunctionName() { return this.functionName; }
    public String getNormalizationMethod() { return this.normalizationMethod; }
    public String getTargetCategory() { return this.targetCategory; }

    private void parseAttributes(Element rootModel) {
        this.modelName = rootModel.attribute("modelName").getValue();
        this.algorithmName = rootModel.attribute("algorithmName").getValue();
        this.functionName = rootModel.attribute("functionName").getValue();
        this.normalizationMethod = rootModel.attribute("normalizationMethod").getValue();
    }

    private static Element findModelElement(Element segmentation) {
        List<Element> elements = segmentation.elements("RegressionTable");
        for (Element element : elements) {
            if ("1".equals(element.attributeValue("targetCategory"))) {
                return element;
            }
        }
        return null;
    }

}
