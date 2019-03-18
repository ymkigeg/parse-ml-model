package com.ymkigeg.ml.pmml;

import java.io.File;
import java.io.InputStream;
import java.net.URL;
import java.util.*;

import lombok.Data;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;


public class GBDTModel {

    private String modelName;
    private String algorithmName;
    private String functionName;
    private List<GBDTTree> treeInfo = new ArrayList<>();
    private Map<String, FieldType> fieldDict;
    private int treeCount;

    private GBDTModel() {}

    public static GBDTModel createByPmml(String modelFile) throws DocumentException{
        SAXReader reader = new SAXReader();
        Document document = reader.read(new File(modelFile));
        return create(document);
    }

    public static GBDTModel createByPmml(InputStream in) throws DocumentException{
        SAXReader reader = new SAXReader();
        Document document = reader.read(in);
        return create(document);
    }

    public static GBDTModel createByPmml(File file) throws DocumentException{
        SAXReader reader = new SAXReader();
        Document document = reader.read(file);
        return create(document);
    }

    public static GBDTModel createByPmml(URL url) throws DocumentException{
        SAXReader reader = new SAXReader();
        Document document = reader.read(url);
        return create(document);
    }

    private static GBDTModel create(Document document) {
        GBDTModel gbdtModel = new GBDTModel();
        Element root = document.getRootElement();
        gbdtModel.parseFieldType(root.element("DataDictionary"));
        gbdtModel.parseAttributes(root.element("MiningModel"));

        List<Element> treeElements = GBDTModel.findTreeElement(root.element("MiningModel").element("Segmentation"));
        gbdtModel.buildGBDTTree(treeElements);

        // 对叶子节点进行特征编码
        gbdtModel.leafEncoding();
        return gbdtModel;
    }

    public double predict(Map<String, Double> data){

        double score = 0.0;
        for (GBDTTree tree : treeInfo) {
            score = score + predictTree(data, tree.getTree());
        }

        // TODO 这里可以将sigmoid函数存成字典，这样就减少计算量
        score = 1 / (Math.exp(-score) + 1);
        return score;
    }

    private double predictTree(Map<String, Double> data, DecisionTree tree) {
        double preds = 0.0;
        while(true) {
            tree = tree.decison(data);
            if (tree.getLeftChild() == null){
                preds += tree.getLeafValue();
                break;
            }
        }
        return preds;
    }

    /**
     * 对数据进行gbdt特征编码
     * @param data
     * @return 特征编号列表
     */
    public int [] gbdtEncoding(Map<String, Double> data){

        int [] result = new int[this.treeCount];
        int i=0;
        for (GBDTTree tree : treeInfo) {
            int nodeId = predictLeaf(data, tree.getTree());
            result[i] = tree.getLeafEncodingMap().get(nodeId);
            i++;
        }

        return result;
    }

    public int [] predictLeaf(Map<String, Double> data){

        int [] result = new int[this.treeCount];
        int i=0;
        for (GBDTTree tree : treeInfo) {
            result[i] = predictLeaf(data, tree.getTree());
            i++;
        }

        return result;
    }

    private int predictLeaf(Map<String, Double> data, DecisionTree tree) {
        while(true) {
            tree = tree.decison(data);
            if (tree.getLeftChild() == null){
                return tree.getNodeId();
            }
        }
    }

    private void leafEncoding() {
        int coding = 0;
        for (GBDTTree tree : treeInfo) {
            List<Integer> order = leafOrder(tree.getTree());
            tree.leafEncodingMap = new HashMap<>();
            for(Integer id : order) {
                tree.leafEncodingMap.put(id, coding);
                coding++;
            }
        }
    }

    private List<Integer> leafOrder(DecisionTree decisionTree) {
        List<Integer> leafs = new ArrayList<>();
        if(!decisionTree.getIsLeaf()) {
            leafs.addAll(leafOrder(decisionTree.getLeftChild()));
            leafs.addAll(leafOrder(decisionTree.getRightChild()));
        } else {
            leafs.add(decisionTree.getNodeId());
        }
        return leafs;
    }


    public String getModelName() { return this.modelName; }
    public String getAlgorithmName() { return this.algorithmName; }
    public String getFunctionName() { return this.functionName; }
    public Map<String, FieldType> getFieldDict() { return this.fieldDict; }
    public int getTreeCount() { return this.treeCount; }

    private void parseFieldType(Element dict) {
        List<Element> dataFields = dict.elements("DataField");
        Map<String, FieldType> result = new HashMap<>();
        for (Element element : dataFields) {
            FieldType fieldType = new FieldType();
            fieldType.setOptype(element.attribute("optype").getValue());
            fieldType.setDataType(element.attribute("dataType").getValue());
            result.put(element.attribute("name").getValue(), fieldType);
        }
        this.fieldDict = result;
    }

    private void parseAttributes(Element rootModel) {
        this.modelName = rootModel.attribute("modelName").getValue();
        this.algorithmName = rootModel.attribute("algorithmName").getValue();
        this.functionName = rootModel.attribute("functionName").getValue();
    }

    private void buildGBDTTree(List<Element> treeElements) {
        int treeCount = 0;
        for (Element treeElement : treeElements) {
            Element rootNode = treeElement.element("TreeModel").element("Node");

            DecisionTree decisionTree = recursiveBuildTree(rootNode);
            decisionTree.splitThrehold();

            GBDTTree gbdtTree = new GBDTTree();
            // gbdtTree.setNumLeaves(numLeaves);
            gbdtTree.setTree(decisionTree);
            treeInfo.add(gbdtTree);
            treeCount++;
        }
        this.treeCount = treeCount;
    }

    private DecisionTree recursiveBuildTree(Element parentNode) {

        DecisionTree decisionTree = new DecisionTree();
        decisionTree.setNodeId(Integer.valueOf(parentNode.attribute("id").getValue()));

        if (parentNode.attribute("score") != null) {
            String score = parentNode.attribute("score").getValue();
            decisionTree.setLeafValue(Double.valueOf(score));
            decisionTree.setIsLeaf(true);
        } else {
            decisionTree.setIsLeaf(false);
        }

        List<Element> childNodes = parentNode.elements("Node");
        if (CollectionUtils.isEmpty(childNodes)) {
            return decisionTree;
        }

        for (Element childNode : childNodes) {
            String splitField = childNode.element("SimplePredicate").attribute("field").getValue();
            decisionTree.setFeatureName(splitField);

            String threshold = childNode.element("SimplePredicate").attribute("value").getValue();
            decisionTree.setThreshold(threshold);

            String decisionType = childNode.element("SimplePredicate").attribute("operator").getValue();

            if ("lessOrEqual".equals(decisionType)) {
                decisionTree.setDecisionType("<=");
                decisionTree.setLeftChild(recursiveBuildTree(childNode));
            } else if ("greaterThan".equals(decisionType)) {
                decisionTree.setDecisionType("<=");
                decisionTree.setRightChild(recursiveBuildTree(childNode));
            }
        }

        return decisionTree;
    }

    private static List<Element> findTreeElement(Element segmentation) {
        List<Element> elements = segmentation.elements("Segment");
        Element treeSegment = null;
        for (Element element : elements) {
            if (element.element("MiningModel") != null
                    && "regression".equals(element.element("MiningModel").attribute("functionName").getValue())) {
                treeSegment = element.element("MiningModel").element("Segmentation");
            }
        }
        return treeSegment.elements("Segment");
    }

    @Data
    public class FieldType {
        String optype;
        String dataType;
    }

    @Data
    private class GBDTTree {
        private DecisionTree tree;
        private Integer numLeaves;
        private Map<Integer, Integer> leafEncodingMap;
        private Double score;
        private Integer j;
    }

    @Data
    private class DecisionTree {
        private Integer nodeId;
        private String featureName;
        private DecisionTree leftChild;
        private DecisionTree rightChild;
        private String decisionType;
        private String threshold;
        private Set<Double> thresholdSet;
        private Double leafValue;
        private Boolean isLeaf;
//        private String missing_type;
//        private Integer internal_value;
//        private Double split_gain;
//        private Integer internal_count;
//        private Integer split_index;
        private Boolean defaultLeft;


        public void splitThrehold() {
            if (StringUtils.isEmpty(this.threshold)) {
                return;
            }
            String [] v = this.threshold.split("\\|\\|");
            this.thresholdSet = new HashSet<>();
            for (String a : v) {
                this.thresholdSet.add(Double.valueOf(a));
            }

            if (this.leftChild != null) {
                this.leftChild.splitThrehold();
            }

            if (this.rightChild != null) {
                this.rightChild.splitThrehold();
            }
        }

        public DecisionTree decison(Map<String, Double> feature) {
            double data = feature.getOrDefault(featureName, 0.0);
            if (data != data) {
                if (this.defaultLeft) return this.getLeftChild();
                else return this.getRightChild();
            } else if (this.decisionType.equals("<=")
                    && data <= this.thresholdSet.iterator().next()) {
                return this.getLeftChild();
            } else if (this.decisionType.equals("==") && this.thresholdSet.contains(data)) {
                return this.getLeftChild();
            } else {
                return this.getRightChild();
            }
        }
    }



}
