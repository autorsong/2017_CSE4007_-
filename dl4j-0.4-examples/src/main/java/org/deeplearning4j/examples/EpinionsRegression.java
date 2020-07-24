package org.deeplearning4j.examples;

import au.com.bytecode.opencsv.CSVReader;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.Collections;

/**
 * Created by songjisu on 2017. 6. 10..
 */
public class EpinionsRegression {
    public static void main(String[] args) throws Exception {
        int seed = 123;
        double learningRate = 0.0001;
        int batchSize = 50;
        int nEpochs = 2;

        int numInputs = 2;
        int numOutputs = 5;

        createNewCSVFile();

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("src/main/resources/EpinionDataset_for_Training_new.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,2,5);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader(0);
        rrTest.initialize(new FileSplit(new File("src/main/resources/EpinionDataset_for_Testing_new.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .regularization(true)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list(7)
            .layer(0, new DenseLayer.Builder()
                .nIn(numInputs)
                .nOut(5)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .build())
            .layer(1, new DenseLayer.Builder()
                .nIn(5)
                .nOut(10)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .build())
            .layer(2, new DenseLayer.Builder()
                .nIn(10)
                .nOut(20)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .build())
            .layer(3, new DenseLayer.Builder()
                .nIn(20)
                .nOut(40)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .build())
            .layer(4, new DenseLayer.Builder()
                .nIn(40)
                .nOut(20)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .build())
            .layer(5, new DenseLayer.Builder()
                .nIn(20)
                .nOut(10)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .build())
            .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .weightInit(WeightInit.XAVIER)
                .activation("softmax")
                .weightInit(WeightInit.XAVIER)
                .nIn(10)
                .nOut(numOutputs).build())
            .pretrain(false).backprop(true).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));

        for (int n = 0; n < nEpochs; n++) {
            model.fit(trainIter);
        }

        // Evaluation and Test
        System.out.println("Evaluate model....");

        String testingFilename = "src/main/resources/EpinionDataset_for_Testing.csv";
        String testDataWriteFilename = "src/main/resources/EpinionDataset_for_Testing_2013011457.csv";

        CSVReader testingFileReader = new CSVReader(new FileReader(testingFilename));
        BufferedWriter testingFileWriter = new BufferedWriter(new FileWriter(testDataWriteFilename));

        String[] splitLine;
        String[] firstLine = testingFileReader.readNext();
        testingFileWriter.write(firstLine[0] + "," + firstLine[1] + "," + firstLine[2] + ","
            + firstLine[3] + "," + firstLine[4] + "," + firstLine[5]);
        testingFileWriter.newLine();

        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray predicted = model.output(features);

            for(int n = 0; n < batchSize; n++){
                double rating = 0.0;

                rating = 1 * predicted.getDouble(n, 0) +
                        2 * predicted.getDouble(n, 1) +
                        3 * predicted.getDouble(n, 2) +
                        4 * predicted.getDouble(n, 3) +
                        5 * predicted.getDouble(n, 4);

                splitLine = testingFileReader.readNext();
                testingFileWriter.write(splitLine[0] + "," + splitLine[1] + "," + splitLine[2] + ",\""
                                    + splitLine[3] + "\"," + rating + "," + splitLine[5]);
                testingFileWriter.newLine();
            }
        }
        testingFileWriter.close();
        testingFileReader.close();
    }

    public static void createNewCSVFile() {
        String trainingFilename = "src/main/resources/EpinionDataset_for_Training.csv";
        String testingFilename = "src/main/resources/EpinionDataset_for_Testing.csv";
        String newTrainingFilename = "src/main/resources/EpinionDataset_for_Training_new.csv";
        String newTestingFilename = "src/main/resources/EpinionDataset_for_Testing_new.csv";

        int trainingCaseNumber = 40000;
        int testingCaseNumber = 10000;
        int trainingCaseCount = 0;
        int testingCaseCount = 0;

        BufferedWriter trainingFileWriter = null;
        BufferedWriter testingFileWriter = null;

        try {
            CSVReader trainingFileReader = new CSVReader(new FileReader(trainingFilename));
            CSVReader testingFileReader = new CSVReader(new FileReader(testingFilename));

            trainingFileWriter = new BufferedWriter(new FileWriter(newTrainingFilename));
            testingFileWriter = new BufferedWriter(new FileWriter(newTestingFilename));

            String[] splitLine;
            trainingFileReader.readNext();
            while ((splitLine = trainingFileReader.readNext()) != null) {
                String writeLine = splitLine[0] + "," + splitLine[2] + "," + splitLine[4];
                trainingFileWriter.write(writeLine);
                trainingFileWriter.newLine();

                if(++trainingCaseCount == trainingCaseNumber) {
                    break;
                }
            }
            trainingFileReader.close();
            trainingFileWriter.close();

            testingFileReader.readNext();
            while ((splitLine = testingFileReader.readNext()) != null) {
                String writeLine = splitLine[0] + "," + splitLine[2];
                testingFileWriter.write(writeLine);
                testingFileWriter.newLine();

                if(++testingCaseCount == testingCaseNumber) {
                    break;
                }
            }
            testingFileReader.close();
            testingFileWriter.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
